from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
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
    cosine_learning_rate,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a Flow Matching teacher with stop-gradient LMD.",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("outputs/teacher/teacher.pt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lmd"))
    parser.add_argument("--steps", type=int, default=70_000)
    parser.add_argument("--batch-size", type=int, default=32_768)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        default=2_048,
        help="Memory-bounded chunk size; gradients accumulate to --batch-size.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--lr-decay-steps", type=int, default=90_000)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--bucket-every", type=int, default=1_000)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save a training checkpoint at this interval.",
    )
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
        help="'auto', 'none', or a checkpoint path",
    )
    parser.add_argument(
        "--pilot-check",
        action="store_true",
        help="Write and enforce a short-run loss trend summary.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "steps",
        "batch_size",
        "microbatch_size",
        "lr_decay_steps",
        "log_every",
        "bucket_every",
        "checkpoint_every",
        "eval_samples",
        "eval_batch_size",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.microbatch_size > args.batch_size:
        raise ValueError("--microbatch-size cannot exceed --batch-size")
    if args.learning_rate <= 0.0 or args.final_learning_rate < 0.0:
        raise ValueError("learning rates must be non-negative")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("--final-learning-rate cannot exceed --learning-rate")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must satisfy 0 <= decay < 1")
    if not args.eval_nfes or any(nfe <= 0 for nfe in args.eval_nfes):
        raise ValueError("--eval-nfes values must be positive")
    if 1 not in args.eval_nfes:
        raise ValueError("--eval-nfes must include 1 for checkpoint selection")


def diagonal_probability(step: int) -> float:
    if step < 5_000:
        return 0.10
    if step < 30_000:
        return 0.10 + 0.15 * (step - 5_000) / 25_000
    return 0.25


def maximum_time_gap(step: int) -> float:
    if step < 5_000:
        return 0.10
    if step < 30_000:
        return 0.10 + 0.90 * (step - 5_000) / 25_000
    return 1.0


def sample_lmd_times(
    batch_size: int,
    *,
    step: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor, Tensor]:
    probability = diagonal_probability(step)
    max_gap = maximum_time_gap(step)
    num_diagonal = int(round(batch_size * probability))
    num_off_diagonal = batch_size - num_diagonal

    diagonal_time = torch.rand(
        num_diagonal,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    if num_off_diagonal == 0:
        return diagonal_time, diagonal_time.clone(), torch.ones(
            batch_size,
            device=device,
            dtype=torch.bool,
        )

    u = torch.rand(
        num_off_diagonal,
        generator=generator,
        device=device,
        dtype=dtype,
    ).clamp_min(torch.finfo(dtype).eps)
    r = torch.rand(
        num_off_diagonal,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    area_coordinate = u * (2.0 * max_gap - max_gap * max_gap)
    gap = area_coordinate / (
        1.0 + torch.sqrt((1.0 - area_coordinate).clamp_min(0.0))
    )
    target_time = r * (1.0 - gap)
    start_time = target_time + gap

    s = torch.cat((diagonal_time, start_time))
    t = torch.cat((diagonal_time, target_time))
    diagonal_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    diagonal_mask[:num_diagonal] = True
    return s, t, diagonal_mask


def lmd_outputs(
    student: FlowMapNet,
    teacher: FlowMapNet,
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
        target_velocity = teacher(mapped.detach(), t, t)
    return mapped, dt_flow, target_velocity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_teacher(
    path: Path,
    *,
    device: torch.device,
) -> tuple[FlowMapNet, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "flow_matching_teacher":
        raise ValueError(f"unexpected teacher checkpoint kind: {payload.get('kind')}")
    config = dict(payload["model_config"])
    teacher = FlowMapNet(**config).to(device=device, dtype=torch.float32)
    teacher.load_state_dict(payload["model"], strict=True)
    teacher.eval()
    teacher.requires_grad_(False)
    metadata = {
        "path": str(path),
        "sha256": file_sha256(path),
        "step": int(payload["step"]),
        "model_config": config,
        "train_config": payload.get("train_config", {}),
    }
    return teacher, metadata


def resolve_resume_path(args: argparse.Namespace) -> Optional[Path]:
    if args.resume.lower() == "none":
        return None
    if args.resume.lower() == "auto":
        return latest_checkpoint(args.output_dir)
    path = Path(args.resume).expanduser().resolve()
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
    teacher_metadata: Mapping[str, Any],
    monitor_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "sg_lmd_training_checkpoint",
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "model_config": student.config,
        "student": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "random_state": capture_random_state(generator=generator),
        "curriculum": {
            "next_step": step,
            "p_diag": diagonal_probability(step),
            "d_max": maximum_time_gap(step),
        },
        "teacher": dict(teacher_metadata),
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
    teacher_metadata: Mapping[str, Any],
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
            teacher_metadata=teacher_metadata,
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
    teacher_metadata: Mapping[str, Any],
) -> tuple[int, float, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "sg_lmd_training_checkpoint":
        raise ValueError(f"unexpected checkpoint kind: {checkpoint.get('kind')}")
    if checkpoint.get("model_config") != student.config:
        raise ValueError("checkpoint model configuration does not match student")
    saved_teacher = checkpoint.get("teacher", {})
    if saved_teacher.get("sha256") != teacher_metadata["sha256"]:
        raise ValueError("teacher SHA-256 does not match the resume checkpoint")
    student.load_state_dict(checkpoint["student"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state(optimizer, device)
    ema.load_state_dict(checkpoint["ema"])
    restore_random_state(checkpoint["random_state"], generator=generator)
    monitor_state = dict(checkpoint.get("monitor_state", {}))
    return (
        int(checkpoint["step"]),
        float(checkpoint.get("elapsed_seconds", 0.0)),
        monitor_state,
    )


def should_checkpoint(
    completed_step: int,
    target_steps: int,
    checkpoint_every: int = 10_000,
) -> bool:
    return (
        completed_step in {5_000, 30_000, target_steps}
        or completed_step % checkpoint_every == 0
    )


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
        label = f"({lower:g},{upper:g}]"
        result[label] = {
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
def consistency_diagnostics(
    student: FlowMapNet,
    teacher: FlowMapNet,
    noise: Tensor,
    reference: Tensor,
    times: Tensor,
) -> dict[str, float]:
    count = min(noise.shape[0], reference.shape[0], times.shape[0])
    noise = noise[:count]
    x0 = reference[:count]
    times = times[:count]
    x_t = (1.0 - times[:, None]) * x0 + times[:, None] * noise
    boundary_error = (
        student(x_t, times, times) - teacher(x_t, times, times)
    ).norm(dim=-1)

    midpoint = student.flow(noise, 1.0, 0.5)
    composed = student.flow(midpoint, 0.5, 0.0)
    direct = student.flow(noise, 1.0, 0.0)
    composition_error = (composed - direct).norm(dim=-1)
    return {
        "boundary_error_mean": float(boundary_error.mean().item()),
        "boundary_error_p95": float(
            torch.quantile(boundary_error, 0.95).item()
        ),
        "composition_error_mean": float(composition_error.mean().item()),
        "composition_error_p95": float(
            torch.quantile(composition_error, 0.95).item()
        ),
    }


def save_ema_model(
    path: Path,
    *,
    kind: str,
    ema: ExponentialMovingAverage,
    model: FlowMapNet,
    step: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
    teacher_metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in ema.shadow.items()
    }
    atomic_torch_save(
        {
            "format_version": 1,
            "kind": kind,
            "step": step,
            "elapsed_seconds": elapsed_seconds,
            "model_config": model.config,
            "model": state_dict,
            "ema_decay": ema.decay,
            "teacher": dict(teacher_metadata),
            "train_config": vars(args),
            "metrics": dict(metrics),
        },
        path,
    )
    return path


@torch.no_grad()
def evaluate_light(
    student: FlowMapNet,
    teacher: FlowMapNet,
    noise: Tensor,
    reference: Tensor,
    diagnostic_times: Tensor,
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
    diagnostics = consistency_diagnostics(
        student,
        teacher,
        noise,
        reference,
        diagnostic_times,
    )
    return metrics_by_nfe, diagnostics


def summarize_pilot(output_dir: Path, completed_step: int) -> dict[str, Any]:
    records = []
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
    preceding = records[-2 * window : -window]
    late = records[-window:]

    def average(key: str, rows: list[dict[str, Any]]) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    early_off = average("off_diagonal_loss", early)
    late_off = average("off_diagonal_loss", late)
    early_diag = average("diagonal_loss", early)
    preceding_diag = average("diagonal_loss", preceding)
    late_diag = average("diagonal_loss", late)
    nonfinite = max(
        int(row.get("nonfinite_loss_count", 0))
        + int(row.get("nonfinite_gradient_count", 0))
        for row in records
    )
    checks = {
        "finite": nonfinite == 0,
        "off_diagonal_decreased": late_off < early_off,
        "diagonal_not_sustained_growth": (
            late_diag <= preceding_diag * 1.25 + 1e-8
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "early_off_diagonal_loss": early_off,
        "late_off_diagonal_loss": late_off,
        "early_diagonal_loss": early_diag,
        "preceding_diagonal_loss": preceding_diag,
        "late_diagonal_loss": late_diag,
        "logged_records": len(records),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.teacher = args.teacher.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    seed_everything(args.seed)
    configure_float32(device)
    teacher, teacher_metadata = load_teacher(args.teacher, device=device)

    student = FlowMapNet(**teacher.config).to(device=device, dtype=torch.float32)
    student.load_state_dict(teacher.state_dict(), strict=True)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(student, decay=args.ema_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    eval_model = FlowMapNet(**teacher.config).to(device=device, dtype=torch.float32)
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
    diagnostic_times = torch.rand(
        args.eval_samples,
        generator=eval_generator,
        device=device,
        dtype=torch.float32,
    )

    monitor_state: dict[str, Any] = {
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "best_step": None,
        "best_sw2": math.inf,
        "best_in_support": -math.inf,
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
            teacher_metadata=teacher_metadata,
        )
        monitor_state.update(saved_monitor)
        print(f"Resumed from {resume_path} at step {start_step:,}", flush=True)

    if start_step >= args.steps:
        print(
            f"Checkpoint is already at step {start_step:,}; "
            f"target is {args.steps:,}.",
            flush=True,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    student.train()
    run_started = time.perf_counter()
    completed_step = start_step
    last_checkpoint_step = start_step if resume_path is not None else -1
    progress = tqdm(
        range(start_step, args.steps),
        initial=start_step,
        total=args.steps,
        desc="SG-LMD",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
        file=sys.stdout,
    )

    try:
        for step_index in progress:
            step_started = time.perf_counter()
            learning_rate = cosine_learning_rate(
                step_index,
                decay_steps=args.lr_decay_steps,
                initial_lr=args.learning_rate,
                final_lr=args.final_learning_rate,
            )
            set_optimizer_lr(optimizer, learning_rate)
            p_diag = diagonal_probability(step_index)
            d_max = maximum_time_gap(step_index)

            x0 = sample_checkerboard(
                args.batch_size,
                generator=generator,
                device=device,
            )
            noise = torch.randn(
                args.batch_size,
                2,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            s, t, diagonal_mask = sample_lmd_times(
                args.batch_size,
                step=step_index,
                generator=generator,
                device=device,
            )
            x_s = (1.0 - s[:, None]) * x0 + s[:, None] * noise

            optimizer.zero_grad(set_to_none=True)
            per_sample_losses = []
            for start in range(0, args.batch_size, args.microbatch_size):
                stop = min(start + args.microbatch_size, args.batch_size)
                _, dt_flow, target_velocity = lmd_outputs(
                    student,
                    teacher,
                    x_s[start:stop],
                    s[start:stop],
                    t[start:stop],
                )
                residual_loss = (
                    dt_flow.sub(target_velocity).square().sum(dim=-1)
                )
                (residual_loss.sum() / args.batch_size).backward()
                per_sample_losses.append(residual_loss.detach())

            losses = torch.cat(per_sample_losses)
            if not torch.isfinite(losses).all():
                monitor_state["nonfinite_loss_count"] += 1
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"non-finite SG-LMD loss at step {step_index + 1}"
                )

            loss = losses.mean()
            diagonal_loss = losses[diagonal_mask].mean()
            off_diagonal_loss = losses[~diagonal_mask].mean()
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
            step_seconds = time.perf_counter() - step_started
            elapsed = previous_elapsed + time.perf_counter() - run_started
            peak_memory = (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            )

            if completed_step == 1 or completed_step % 10 == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.5f}",
                    diag=f"{diagonal_loss.item():.5f}",
                    off=f"{off_diagonal_loss.item():.5f}",
                    gap=f"{d_max:.2f}",
                    lr=f"{learning_rate:.2e}",
                    refresh=False,
                )

            should_log = (
                completed_step == 1
                or completed_step % args.log_every == 0
                or completed_step == args.steps
            )
            if should_log:
                record: dict[str, Any] = {
                    "event": "train",
                    "step": completed_step,
                    "sg_lmd_loss": float(loss.item()),
                    "diagonal_loss": float(diagonal_loss.item()),
                    "off_diagonal_loss": float(off_diagonal_loss.item()),
                    "p_diag": p_diag,
                    "actual_p_diag": float(diagonal_mask.float().mean().item()),
                    "d_max": d_max,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm.item()),
                    "gradient_clipped": was_clipped,
                    "samples_per_second": args.batch_size / max(step_seconds, 1e-12),
                    "average_samples_per_second": (
                        completed_step * args.batch_size / max(elapsed, 1e-12)
                    ),
                    "elapsed_seconds": elapsed,
                    "cuda_peak_memory_bytes": peak_memory,
                    "nonfinite_loss_count": monitor_state[
                        "nonfinite_loss_count"
                    ],
                    "nonfinite_gradient_count": monitor_state[
                        "nonfinite_gradient_count"
                    ],
                }
                if (
                    completed_step % args.bucket_every == 0
                    or completed_step == args.steps
                ):
                    off_mask = ~diagonal_mask
                    record["gap_buckets"] = gap_bucket_statistics(
                        (s[off_mask] - t[off_mask]).detach().cpu(),
                        losses[off_mask].detach().cpu(),
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
                    teacher,
                    fixed_noise[: args.eval_samples],
                    fixed_reference[: args.eval_samples],
                    diagnostic_times[: args.eval_samples],
                    nfes=sorted(set(args.eval_nfes)),
                    batch_size=args.eval_batch_size,
                )
                one_nfe = metrics_by_nfe.get("1")
                if one_nfe is None:
                    first_nfe = str(min(args.eval_nfes))
                    one_nfe = metrics_by_nfe[first_nfe]
                score = (
                    float(one_nfe["sw2"]),
                    -float(one_nfe["in_support_rate"]),
                )
                best_score = (
                    float(monitor_state["best_sw2"]),
                    -float(monitor_state["best_in_support"]),
                )
                if score < best_score:
                    monitor_state.update(
                        {
                            "best_step": completed_step,
                            "best_sw2": score[0],
                            "best_in_support": -score[1],
                        }
                    )
                    save_ema_model(
                        args.output_dir / "best_lmd_sg.pt",
                        kind="sg_lmd_best",
                        ema=ema,
                        model=student,
                        step=completed_step,
                        elapsed_seconds=elapsed,
                        args=args,
                        teacher_metadata=teacher_metadata,
                        metrics={
                            "metrics_by_nfe": metrics_by_nfe,
                            "diagnostics": diagnostics,
                        },
                    )
                append_history(
                    args.output_dir,
                    {
                        "event": "eval",
                        "step": completed_step,
                        "metrics_by_nfe": metrics_by_nfe,
                        "diagnostics": diagnostics,
                        "best_step": monitor_state["best_step"],
                    },
                )
                progress.write(
                    f"eval step={completed_step:,} "
                    f"1-NFE sw2={one_nfe['sw2']:.5f} "
                    f"in_support={one_nfe['in_support_rate']:.4f} "
                    f"best={monitor_state['best_step']}"
                )
                eval_model.train()

            if should_checkpoint(
                completed_step,
                args.steps,
                args.checkpoint_every,
            ):
                checkpoint_path = save_training_checkpoint(
                    student=student,
                    optimizer=optimizer,
                    ema=ema,
                    generator=generator,
                    step=completed_step,
                    elapsed_seconds=elapsed,
                    args=args,
                    teacher_metadata=teacher_metadata,
                    monitor_state=monitor_state,
                )
                last_checkpoint_step = completed_step
                progress.write(f"saved {checkpoint_path}")
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
            teacher_metadata=teacher_metadata,
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
            teacher_metadata=teacher_metadata,
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
                f"pilot trend checks failed; see {args.output_dir / 'pilot_summary.json'}"
            )
    print(
        f"Finished step {completed_step:,}; checkpoints are in {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
