from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import Tensor
from tqdm.auto import tqdm

from common import (
    ExponentialMovingAverage,
    atomic_torch_save,
    capture_random_state,
    configure_float32,
    latest_checkpoint,
    move_optimizer_state,
    resolve_device,
    restore_random_state,
    seed_everything,
    set_optimizer_lr,
)
from dataset import sample_checkerboard
from metrics import (
    distance_to_support,
    histogram_distances,
    mode_statistics,
    sliced_wasserstein_2,
    support_membership,
    within_cell_uniformity_jsd,
)
from models import FlowMapNet
from sampling import flow_map_sample


GAP_BUCKETS = (
    (0.0, 0.05),
    (0.05, 0.1),
    (0.1, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
)


@dataclass
class LsdTrainingBatch:
    diagonal_x: Tensor
    diagonal_time: Tensor
    diagonal_target: Tensor
    off_diagonal_x: Tensor
    start_time: Tensor
    target_time: Tensor

    @property
    def num_diagonal(self) -> int:
        return self.diagonal_x.shape[0]

    @property
    def num_off_diagonal(self) -> int:
        return self.off_diagonal_x.shape[0]

    @property
    def batch_size(self) -> int:
        return self.num_diagonal + self.num_off_diagonal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a FlowMapNet with Lagrangian self-distillation.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lsd"))
    parser.add_argument("--steps", type=int, default=130_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=2_048,
        help="Physical chunk size; the logical batch size never changes.",
    )
    parser.add_argument("--diag-fraction", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=35_000,
        help="Keep the initial LR through this step, then use inverse sqrt decay.",
    )
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--bucket-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument(
        "--late-checkpoint-start",
        type=int,
        default=100_000,
    )
    parser.add_argument("--late-checkpoint-every", type=int, default=5_000)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5_000,
        help="Light-evaluation interval; use 0 to disable evaluation.",
    )
    parser.add_argument("--eval-samples", type=int, default=10_000)
    parser.add_argument("--eval-batch-size", type=int, default=10_000)
    parser.add_argument(
        "--eval-nfes",
        nargs="+",
        type=int,
        default=(1, 2, 4),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or an LSD training checkpoint path.",
    )
    parser.add_argument(
        "--pilot-check",
        action="store_true",
        help="Write and enforce the 2k-step pilot trend checks.",
    )
    return parser.parse_args()


def branch_sizes(batch_size: int, diagonal_fraction: float) -> tuple[int, int]:
    num_diagonal_float = batch_size * diagonal_fraction
    num_diagonal = int(round(num_diagonal_float))
    if not math.isclose(num_diagonal_float, num_diagonal, abs_tol=1e-9):
        raise ValueError(
            "--batch-size times --diag-fraction must be an integer "
            "so the branch split is exact"
        )
    return num_diagonal, batch_size - num_diagonal


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "steps",
        "batch_size",
        "microbatch_size",
        "lr_decay_steps",
        "log_every",
        "bucket_every",
        "checkpoint_every",
        "late_checkpoint_every",
        "eval_samples",
        "eval_batch_size",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.microbatch_size > args.batch_size:
        raise ValueError("--microbatch-size cannot exceed --batch-size")
    if not 0.0 < args.diag_fraction < 1.0:
        raise ValueError("--diag-fraction must satisfy 0 < value < 1")
    branch_sizes(args.batch_size, args.diag_fraction)
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must satisfy 0 <= decay < 1")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    if args.late_checkpoint_start < 0:
        raise ValueError("--late-checkpoint-start must be non-negative")
    if not args.eval_nfes or any(nfe <= 0 for nfe in args.eval_nfes):
        raise ValueError("--eval-nfes values must be positive")
    if 1 not in args.eval_nfes:
        raise ValueError("--eval-nfes must include 1 for checkpoint selection")


def lsd_learning_rate(
    step: int,
    *,
    initial_lr: float,
    decay_start: int,
) -> float:
    return initial_lr / math.sqrt(max(step / decay_start, 1.0))


def sample_off_diagonal_times(
    num_samples: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    values = torch.rand(
        num_samples,
        2,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    start_time = values.max(dim=-1).values
    target_time = values.min(dim=-1).values

    equal = start_time == target_time
    while equal.any():
        replacements = torch.rand(
            int(equal.sum().item()),
            2,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        start_time[equal] = replacements.max(dim=-1).values
        target_time[equal] = replacements.min(dim=-1).values
        equal = start_time == target_time
    return start_time, target_time


def sample_lsd_training_batch(
    batch_size: int,
    *,
    diagonal_fraction: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> LsdTrainingBatch:
    num_diagonal, num_off_diagonal = branch_sizes(
        batch_size,
        diagonal_fraction,
    )

    diagonal_x0 = sample_checkerboard(
        num_diagonal,
        generator=generator,
        device=device,
    ).to(dtype=dtype)
    diagonal_noise = torch.randn(
        num_diagonal,
        2,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    diagonal_time = torch.rand(
        num_diagonal,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    diagonal_x = (
        (1.0 - diagonal_time[:, None]) * diagonal_x0
        + diagonal_time[:, None] * diagonal_noise
    )
    diagonal_target = diagonal_noise - diagonal_x0

    off_diagonal_x0 = sample_checkerboard(
        num_off_diagonal,
        generator=generator,
        device=device,
    ).to(dtype=dtype)
    off_diagonal_noise = torch.randn(
        num_off_diagonal,
        2,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    start_time, target_time = sample_off_diagonal_times(
        num_off_diagonal,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    off_diagonal_x = (
        (1.0 - start_time[:, None]) * off_diagonal_x0
        + start_time[:, None] * off_diagonal_noise
    )
    return LsdTrainingBatch(
        diagonal_x=diagonal_x,
        diagonal_time=diagonal_time,
        diagonal_target=diagonal_target,
        off_diagonal_x=off_diagonal_x,
        start_time=start_time,
        target_time=target_time,
    )


def lsd_outputs(
    student: FlowMapNet,
    x_s: Tensor,
    s: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    mapped, dt_flow = torch.func.jvp(
        lambda target_t: student.flow(x_s, s, target_t),
        (t,),
        (torch.ones_like(t),),
    )
    with torch.no_grad():
        target_velocity = student(mapped.detach(), t, t)
    return mapped, dt_flow, target_velocity


def resolve_resume_path(args: argparse.Namespace) -> Optional[Path]:
    resume = str(args.resume)
    if resume.lower() == "none":
        return None
    if resume.lower() == "auto":
        return latest_checkpoint(args.output_dir)
    path = Path(resume).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def append_history(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), sort_keys=True) + "\n")


def atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def checkpoint_payload(
    *,
    student: FlowMapNet,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    generator: torch.Generator,
    step: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
    monitor_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "lsd_training_checkpoint",
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "model_config": student.config,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "random_state": capture_random_state(generator=generator),
        "monitor_state": dict(monitor_state),
        "train_config": vars(args),
    }


def save_training_checkpoint(
    *,
    student: FlowMapNet,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    generator: torch.Generator,
    step: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
    monitor_state: Mapping[str, Any],
) -> Path:
    path = args.output_dir / f"checkpoint_{step:06d}.pt"
    atomic_torch_save(
        checkpoint_payload(
            student=student,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=step,
            elapsed_seconds=elapsed_seconds,
            args=args,
            monitor_state=monitor_state,
        ),
        path,
    )
    return path


def load_training_checkpoint(
    path: Path,
    *,
    student: FlowMapNet,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, float, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "lsd_training_checkpoint":
        raise ValueError(f"unexpected checkpoint kind: {checkpoint.get('kind')}")
    if checkpoint.get("model_config") != student.config:
        raise ValueError("checkpoint model configuration does not match student")
    student.load_state_dict(checkpoint["student"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state(optimizer, device)
    ema.load_state_dict(checkpoint["ema"])
    restore_random_state(checkpoint["random_state"], generator=generator)
    return (
        int(checkpoint["step"]),
        float(checkpoint.get("elapsed_seconds", 0.0)),
        dict(checkpoint.get("monitor_state", {})),
    )


def should_checkpoint(
    completed_step: int,
    target_steps: int,
    *,
    checkpoint_every: int,
    late_start: int,
    late_every: int,
) -> bool:
    regular = completed_step % checkpoint_every == 0
    late = completed_step >= late_start and completed_step % late_every == 0
    return completed_step == target_steps or regular or late


def residual_summary(values: Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0}
    values = values.float()
    quantiles = torch.quantile(
        values,
        torch.tensor((0.5, 0.95, 0.99), dtype=values.dtype),
    )
    return {
        "mean": float(values.mean().item()),
        "median": float(quantiles[0].item()),
        "p95": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
    }


def gap_bucket_statistics(gaps: Tensor, losses: Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for lower, upper in GAP_BUCKETS:
        mask = (gaps > lower) & (gaps <= upper)
        result[f"({lower:g},{upper:g}]"] = {
            "count": int(mask.sum().item()),
            **residual_summary(losses[mask]),
        }
    return result


@torch.no_grad()
def light_distribution_metrics(
    samples: Tensor,
    reference: Tensor,
) -> dict[str, float | int | list[float]]:
    membership, _ = support_membership(samples)
    distances = distance_to_support(samples)
    outside = distances[~membership]
    metrics: dict[str, float | int | list[float]] = {
        "num_samples": samples.shape[0],
        "in_support_rate": float(membership.float().mean().item()),
        "outside_distance_mean": (
            float(outside.mean().item()) if outside.numel() else 0.0
        ),
        "outside_distance_p95": (
            float(torch.quantile(outside, 0.95).item())
            if outside.numel()
            else 0.0
        ),
        "within_cell_jsd": within_cell_uniformity_jsd(samples),
        "sw2": sliced_wasserstein_2(
            samples,
            reference,
            num_projections=128,
            num_quantiles=min(10_000, samples.shape[0]),
        ),
    }
    metrics.update(mode_statistics(samples))
    metrics.update(histogram_distances(samples))
    return metrics


@torch.no_grad()
def self_consistency_diagnostics(
    student: FlowMapNet,
    noise: Tensor,
    reference: Tensor,
    start_time: Tensor,
    target_time: Tensor,
) -> dict[str, float]:
    count = min(
        noise.shape[0],
        reference.shape[0],
        start_time.shape[0],
        target_time.shape[0],
    )
    noise = noise[:count]
    reference = reference[:count]
    start_time = start_time[:count]
    target_time = target_time[:count]
    x_s = (
        (1.0 - start_time[:, None]) * reference
        + start_time[:, None] * noise
    )
    mapped, dt_flow, target_velocity = lsd_outputs(
        student,
        x_s,
        start_time,
        target_time,
    )
    residual = (dt_flow - target_velocity).norm(dim=-1)

    midpoint = student.flow(noise, 1.0, 0.5)
    composed = student.flow(midpoint, 0.5, 0.0)
    direct = student.flow(noise, 1.0, 0.0)
    composition_error = (composed - direct).norm(dim=-1)
    return {
        "lsd_residual_mean": float(residual.mean().item()),
        "lsd_residual_p95": float(torch.quantile(residual, 0.95).item()),
        "composition_error_mean": float(composition_error.mean().item()),
        "composition_error_p95": float(
            torch.quantile(composition_error, 0.95).item()
        ),
        "mapped_norm_mean": float(mapped.norm(dim=-1).mean().item()),
    }


@torch.no_grad()
def evaluate_light(
    student: FlowMapNet,
    noise: Tensor,
    reference: Tensor,
    diagnostic_start: Tensor,
    diagnostic_target: Tensor,
    *,
    nfes: list[int],
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    metrics_by_nfe: dict[str, Any] = {}
    for nfe in nfes:
        samples = flow_map_sample(
            student,
            noise,
            nfe=nfe,
            batch_size=batch_size,
        )
        metrics_by_nfe[str(nfe)] = light_distribution_metrics(
            samples,
            reference,
        )
    diagnostics = self_consistency_diagnostics(
        student,
        noise,
        reference,
        diagnostic_start,
        diagnostic_target,
    )
    return metrics_by_nfe, diagnostics


def parameters_are_finite(model: FlowMapNet) -> bool:
    return all(
        bool(torch.isfinite(parameter).all())
        for parameter in model.parameters()
    )


def summarize_pilot(output_dir: Path, completed_step: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    history_path = output_dir / "history.jsonl"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if (
                    record.get("event") == "train"
                    and int(record.get("step", -1)) <= completed_step
                ):
                    records.append(record)
    if len(records) < 4:
        return {
            "passed": False,
            "reason": "not enough logged train records for pilot trend checks",
        }

    window = max(2, len(records) // 5)
    early = records[:window]
    late = records[-window:]

    def average(key: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    nonfinite = max(
        int(row.get("nonfinite_loss_count", 0))
        + int(row.get("nonfinite_gradient_count", 0))
        + int(row.get("nonfinite_parameter_count", 0))
        for row in records
    )
    late_clip_rate = sum(bool(row["gradient_clipped"]) for row in late) / len(late)
    early_reserved = max(
        int(row.get("cuda_peak_reserved_bytes", 0))
        for row in early
    )
    late_reserved = max(
        int(row.get("cuda_peak_reserved_bytes", 0))
        for row in late
    )
    memory_tolerance = max(int(early_reserved * 0.10), 64 * 1024 * 1024)
    checks = {
        "finite": nonfinite == 0,
        "fm_decreased": average("fm_loss", late) < average("fm_loss", early),
        "lsd_decreased": average("lsd_loss", late) < average("lsd_loss", early),
        "gradient_clipping_not_saturated": late_clip_rate < 0.80,
        "reserved_memory_stable": (
            early_reserved == 0
            or late_reserved <= early_reserved + memory_tolerance
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "early_fm_loss": average("fm_loss", early),
        "late_fm_loss": average("fm_loss", late),
        "early_lsd_loss": average("lsd_loss", early),
        "late_lsd_loss": average("lsd_loss", late),
        "late_gradient_clip_rate": late_clip_rate,
        "early_peak_reserved_bytes": early_reserved,
        "late_peak_reserved_bytes": late_reserved,
        "logged_records": len(records),
    }


def write_oom_report(
    args: argparse.Namespace,
    device: torch.device,
    completed_step: int,
    error: BaseException,
) -> Path:
    report = {
        "event": "cuda_oom",
        "completed_step": completed_step,
        "device": str(device),
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "message": str(error),
        "instruction": (
            "Rerun with an explicitly smaller --microbatch-size. "
            "The logical --batch-size remains unchanged."
        ),
    }
    if device.type == "cuda":
        report.update(
            {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            }
        )
    path = args.output_dir / "oom_report.json"
    atomic_json_save(report, path)
    return path


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    seed_everything(args.seed)
    configure_float32(device)

    student = FlowMapNet().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(student, decay=args.ema_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    eval_model = FlowMapNet(**student.config).to(
        device=device,
        dtype=torch.float32,
    )
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(args.seed + 1)
    fixed_noise = torch.randn(
        args.eval_samples,
        2,
        generator=eval_generator,
        device=device,
        dtype=torch.float32,
    )
    fixed_reference = sample_checkerboard(
        args.eval_samples,
        generator=eval_generator,
        device=device,
    )
    diagnostic_start, diagnostic_target = sample_off_diagonal_times(
        args.eval_samples,
        generator=eval_generator,
        device=device,
    )

    monitor_state: dict[str, Any] = {
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "nonfinite_parameter_count": 0,
    }
    start_step = 0
    previous_elapsed = 0.0
    resume_path = resolve_resume_path(args)
    if resume_path is not None:
        start_step, previous_elapsed, saved_monitor = load_training_checkpoint(
            resume_path,
            student=student,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            device=device,
        )
        for name in monitor_state:
            if name in saved_monitor:
                monitor_state[name] = saved_monitor[name]
        print(f"Resumed from {resume_path} at step {start_step:,}", flush=True)

    if start_step >= args.steps:
        print(
            f"Checkpoint is already at step {start_step:,}; "
            f"target is {args.steps:,}.",
            flush=True,
        )

    student.train()
    run_started = time.perf_counter()
    completed_step = start_step
    last_checkpoint_step = start_step if resume_path is not None else -1
    progress = tqdm(
        range(start_step, args.steps),
        initial=start_step,
        total=args.steps,
        desc="LSD",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
        file=sys.stdout,
    )

    try:
        for step_index in progress:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            learning_rate = lsd_learning_rate(
                step_index,
                initial_lr=args.learning_rate,
                decay_start=args.lr_decay_steps,
            )
            set_optimizer_lr(optimizer, learning_rate)
            batch = sample_lsd_training_batch(
                args.batch_size,
                diagonal_fraction=args.diag_fraction,
                generator=generator,
                device=device,
            )
            collect_diagnostics = (
                step_index + 1 == args.steps
                or (step_index + 1) % args.bucket_every == 0
            )

            optimizer.zero_grad(set_to_none=True)
            diagonal_losses: list[Tensor] = []
            off_diagonal_losses: list[Tensor] = []
            dt_norms: list[Tensor] = []
            target_norms: list[Tensor] = []
            average_velocity_norms: list[Tensor] = []

            for start in range(
                0,
                batch.num_diagonal,
                args.microbatch_size,
            ):
                stop = min(start + args.microbatch_size, batch.num_diagonal)
                prediction = student(
                    batch.diagonal_x[start:stop],
                    batch.diagonal_time[start:stop],
                    batch.diagonal_time[start:stop],
                )
                sample_losses = (
                    prediction.sub(batch.diagonal_target[start:stop])
                    .square()
                    .sum(dim=-1)
                )
                (sample_losses.sum() / args.batch_size).backward()
                diagonal_losses.append(sample_losses.detach())

            for start in range(
                0,
                batch.num_off_diagonal,
                args.microbatch_size,
            ):
                stop = min(
                    start + args.microbatch_size,
                    batch.num_off_diagonal,
                )
                x_s = batch.off_diagonal_x[start:stop]
                s = batch.start_time[start:stop]
                t = batch.target_time[start:stop]
                _, dt_flow, target_velocity = lsd_outputs(
                    student,
                    x_s,
                    s,
                    t,
                )
                sample_losses = (
                    dt_flow.sub(target_velocity).square().sum(dim=-1)
                )
                (sample_losses.sum() / args.batch_size).backward()
                off_diagonal_losses.append(sample_losses.detach())
                if collect_diagnostics:
                    dt_norms.append(dt_flow.detach().norm(dim=-1))
                    target_norms.append(
                        target_velocity.detach().norm(dim=-1)
                    )
                    with torch.no_grad():
                        average_velocity_norms.append(
                            student(x_s, s, t).norm(dim=-1)
                        )

            fm_losses = torch.cat(diagonal_losses)
            lsd_losses = torch.cat(off_diagonal_losses)
            total_loss = (
                fm_losses.sum() + lsd_losses.sum()
            ) / args.batch_size
            if not torch.isfinite(total_loss):
                monitor_state["nonfinite_loss_count"] += 1
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite LSD loss at step {step_index + 1}"
                )

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                args.grad_clip,
            )
            if not torch.isfinite(gradient_norm):
                monitor_state["nonfinite_gradient_count"] += 1
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite gradient norm at step {step_index + 1}"
                )
            was_clipped = float(gradient_norm.item()) > args.grad_clip
            optimizer.step()
            ema.update(student)
            completed_step = step_index + 1

            should_log = (
                completed_step == 1
                or completed_step % args.log_every == 0
                or completed_step == args.steps
            )
            if should_log and not parameters_are_finite(student):
                monitor_state["nonfinite_parameter_count"] += 1
                raise FloatingPointError(
                    f"non-finite model parameter at step {completed_step}"
                )

            step_seconds = time.perf_counter() - step_started
            elapsed = previous_elapsed + time.perf_counter() - run_started
            peak_allocated = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )
            peak_reserved = (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda"
                else 0
            )
            fm_loss = fm_losses.mean()
            lsd_loss = lsd_losses.mean()

            if completed_step == 1 or completed_step % 10 == 0:
                progress.set_postfix(
                    loss=f"{total_loss.item():.5f}",
                    fm=f"{fm_loss.item():.5f}",
                    lsd=f"{lsd_loss.item():.5f}",
                    lr=f"{learning_rate:.2e}",
                    refresh=False,
                )

            if should_log:
                record: dict[str, Any] = {
                    "event": "train",
                    "step": completed_step,
                    "loss": float(total_loss.item()),
                    "fm_loss": float(fm_loss.item()),
                    "lsd_loss": float(lsd_loss.item()),
                    "num_diagonal": batch.num_diagonal,
                    "num_off_diagonal": batch.num_off_diagonal,
                    "diag_fraction": args.diag_fraction,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm.item()),
                    "gradient_clipped": was_clipped,
                    "samples_per_second": (
                        args.batch_size / max(step_seconds, 1e-12)
                    ),
                    "average_samples_per_second": (
                        completed_step
                        * args.batch_size
                        / max(elapsed, 1e-12)
                    ),
                    "elapsed_seconds": elapsed,
                    "cuda_peak_allocated_bytes": peak_allocated,
                    "cuda_peak_reserved_bytes": peak_reserved,
                    "nonfinite_loss_count": monitor_state[
                        "nonfinite_loss_count"
                    ],
                    "nonfinite_gradient_count": monitor_state[
                        "nonfinite_gradient_count"
                    ],
                    "nonfinite_parameter_count": monitor_state[
                        "nonfinite_parameter_count"
                    ],
                }
                if collect_diagnostics:
                    gaps = (
                        batch.start_time - batch.target_time
                    ).detach().cpu()
                    record["gap_buckets"] = gap_bucket_statistics(
                        gaps,
                        lsd_losses.detach().cpu(),
                    )
                    record["dt_flow_norm_mean"] = float(
                        torch.cat(dt_norms).mean().item()
                    )
                    record["instantaneous_target_norm_mean"] = float(
                        torch.cat(target_norms).mean().item()
                    )
                    record["average_velocity_norm_mean"] = float(
                        torch.cat(average_velocity_norms).mean().item()
                    )
                append_history(args.output_dir, record)

            should_evaluate = args.eval_every > 0 and (
                completed_step % args.eval_every == 0
                or completed_step == args.steps
            )
            if should_evaluate:
                ema.copy_to(eval_model)
                eval_model.eval()
                metrics_by_nfe, diagnostics = evaluate_light(
                    eval_model,
                    fixed_noise,
                    fixed_reference,
                    diagnostic_start,
                    diagnostic_target,
                    nfes=sorted(set(args.eval_nfes)),
                    batch_size=args.eval_batch_size,
                )
                append_history(
                    args.output_dir,
                    {
                        "event": "eval",
                        "step": completed_step,
                        "metrics_by_nfe": metrics_by_nfe,
                        "diagnostics": diagnostics,
                    },
                )
                one_nfe = metrics_by_nfe["1"]
                progress.write(
                    f"eval step={completed_step:,} "
                    f"1-NFE sw2={one_nfe['sw2']:.5f} "
                    f"in_support={one_nfe['in_support_rate']:.4f}"
                )
                eval_model.train()

            if should_checkpoint(
                completed_step,
                args.steps,
                checkpoint_every=args.checkpoint_every,
                late_start=args.late_checkpoint_start,
                late_every=args.late_checkpoint_every,
            ):
                checkpoint_path = save_training_checkpoint(
                    student=student,
                    optimizer=optimizer,
                    ema=ema,
                    generator=generator,
                    step=completed_step,
                    elapsed_seconds=elapsed,
                    args=args,
                    monitor_state=monitor_state,
                )
                last_checkpoint_step = completed_step
                progress.write(f"saved {checkpoint_path}")
    except torch.OutOfMemoryError as error:
        progress.close()
        report_path = write_oom_report(
            args,
            device,
            completed_step,
            error,
        )
        print(
            "\nCUDA out of memory. The logical batch was not changed. "
            f"See {report_path} and rerun with an explicit "
            "--microbatch-size.",
            flush=True,
        )
        raise
    except (KeyboardInterrupt, FloatingPointError):
        progress.close()
        elapsed = previous_elapsed + time.perf_counter() - run_started
        checkpoint_path = save_training_checkpoint(
            student=student,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=completed_step,
            elapsed_seconds=elapsed,
            args=args,
            monitor_state=monitor_state,
        )
        print(f"\nStopped; saved {checkpoint_path}", flush=True)
        if sys.exc_info()[0] is FloatingPointError:
            raise
        return
    finally:
        progress.close()

    elapsed = previous_elapsed + time.perf_counter() - run_started
    if last_checkpoint_step != completed_step:
        checkpoint_path = save_training_checkpoint(
            student=student,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=completed_step,
            elapsed_seconds=elapsed,
            args=args,
            monitor_state=monitor_state,
        )
        print(f"saved {checkpoint_path}", flush=True)

    if args.pilot_check:
        pilot_summary = summarize_pilot(args.output_dir, completed_step)
        atomic_json_save(pilot_summary, args.output_dir / "pilot_summary.json")
        print(
            f"Pilot check: {'PASS' if pilot_summary['passed'] else 'FAIL'}",
            flush=True,
        )
        if not pilot_summary["passed"]:
            raise RuntimeError(
                "pilot trend checks failed; see "
                f"{args.output_dir / 'pilot_summary.json'}"
            )
    print(
        f"Finished step {completed_step:,}; checkpoints are in {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
