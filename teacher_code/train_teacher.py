from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

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
from metrics import histogram_distances, mode_statistics, support_membership
from models import FlowMapNet
from sampling import euler_sample_teacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the checkerboard Flow Matching teacher.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/teacher"))
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--final-learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-decay-steps", type=int, default=120_000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--checkpoint-every", type=int, default=15_000)
    parser.add_argument("--eval-every", type=int, default=15_000)
    parser.add_argument("--eval-samples", type=int, default=10_000)
    parser.add_argument("--eval-nfe", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or a checkpoint path",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "steps",
        "batch_size",
        "lr_decay_steps",
        "checkpoint_every",
        "eval_every",
        "eval_samples",
        "eval_nfe",
        "eval_batch_size",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0.0 or args.final_learning_rate < 0.0:
        raise ValueError("learning rates must be non-negative, with a positive initial rate")
    if args.final_learning_rate > args.learning_rate:
        raise ValueError("--final-learning-rate cannot exceed --learning-rate")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")
def resolve_resume_path(args: argparse.Namespace) -> Optional[Path]:
    if args.resume.lower() == "none":
        return None
    if args.resume.lower() == "auto":
        return latest_checkpoint(args.output_dir)
    path = Path(args.resume).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def checkpoint_payload(
    *,
    model: FlowMapNet,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    generator: torch.Generator,
    step: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "teacher_training_checkpoint",
        "step": step,
        "elapsed_seconds": elapsed_seconds,
        "model_config": model.config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
        "random_state": capture_random_state(generator=generator),
        "train_config": vars(args),
    }


def save_training_checkpoint(
    *,
    model: FlowMapNet,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    generator: torch.Generator,
    step: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
) -> Path:
    path = args.output_dir / f"checkpoint_{step:06d}.pt"
    atomic_torch_save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=step,
            elapsed_seconds=elapsed_seconds,
            args=args,
        ),
        path,
    )
    return path


def load_training_checkpoint(
    path: Path,
    *,
    model: FlowMapNet,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_config") != model.config:
        raise ValueError("checkpoint model configuration does not match FlowMapNet")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    move_optimizer_state(optimizer, device)
    ema.load_state_dict(checkpoint["ema"])
    restore_random_state(checkpoint["random_state"], generator=generator)
    return int(checkpoint["step"]), float(checkpoint.get("elapsed_seconds", 0.0))


def append_history(output_dir: Path, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate_teacher(
    model: FlowMapNet,
    noise: Tensor,
    *,
    nfe: int,
    batch_size: int,
) -> dict[str, float | int | list[float]]:
    was_training = model.training
    model.eval()
    samples = euler_sample_teacher(
        model,
        noise,
        nfe=nfe,
        batch_size=batch_size,
    )
    membership, _ = support_membership(samples)
    results: dict[str, float | int | list[float]] = {
        "in_support_rate": float(membership.float().mean().item()),
    }
    results.update(mode_statistics(samples))
    results.update(histogram_distances(samples))
    model.train(was_training)
    return results


def save_teacher(
    *,
    ema: ExponentialMovingAverage,
    model: FlowMapNet,
    step: int,
    elapsed_seconds: float,
    args: argparse.Namespace,
    metrics: dict[str, float | int | list[float]],
) -> Path:
    path = args.output_dir / "teacher.pt"
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in ema.shadow.items()
    }
    atomic_torch_save(
        {
            "format_version": 1,
            "kind": "flow_matching_teacher",
            "step": step,
            "elapsed_seconds": elapsed_seconds,
            "model_config": model.config,
            "model": state_dict,
            "ema_decay": ema.decay,
            "train_config": vars(args),
            "eval_nfe": args.eval_nfe,
            "metrics": metrics,
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    seed_everything(args.seed)
    configure_float32(device)

    model = FlowMapNet().to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(args.seed + 1)
    fixed_eval_noise = torch.randn(
        args.eval_samples,
        2,
        generator=eval_generator,
        device=device,
        dtype=torch.float32,
    )
    eval_model = FlowMapNet(**model.config).to(device=device, dtype=torch.float32)

    start_step = 0
    previous_elapsed = 0.0
    resume_path = resolve_resume_path(args)
    if resume_path is not None:
        start_step, previous_elapsed = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            device=device,
        )
        print(f"Resumed from {resume_path} at step {start_step:,}", flush=True)

    if start_step >= args.steps:
        print(
            f"Checkpoint is already at step {start_step:,}; "
            f"target is {args.steps:,}.",
            flush=True,
        )

    model.train()
    run_started = time.perf_counter()
    completed_step = start_step
    progress = tqdm(
        range(start_step, args.steps),
        initial=start_step,
        total=args.steps,
        desc="Teacher FM",
        unit="step",
        dynamic_ncols=True,
        mininterval=0.2,
        file=sys.stdout,
    )

    try:
        for step_index in progress:
            learning_rate = cosine_learning_rate(
                step_index,
                decay_steps=args.lr_decay_steps,
                initial_lr=args.learning_rate,
                final_lr=args.final_learning_rate,
            )
            set_optimizer_lr(optimizer, learning_rate)

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
            time_batch = torch.rand(
                args.batch_size,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            xt = (1.0 - time_batch[:, None]) * x0 + time_batch[:, None] * noise
            target_velocity = noise - x0

            optimizer.zero_grad(set_to_none=True)
            predicted_velocity = model(xt, time_batch, time_batch)
            loss = (
                predicted_velocity.sub(target_velocity)
                .square()
                .sum(dim=-1)
                .mean()
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            optimizer.step()
            ema.update(model)
            completed_step = step_index + 1

            if completed_step == 1 or completed_step % 10 == 0:
                progress.set_postfix(
                    loss=f"{loss.item():.5f}",
                    lr=f"{learning_rate:.2e}",
                    grad=f"{float(gradient_norm):.3f}",
                    refresh=False,
                )

            should_evaluate = (
                completed_step % args.eval_every == 0
                or completed_step == args.steps
            )
            if should_evaluate:
                ema.copy_to(eval_model)
                metrics = evaluate_teacher(
                    eval_model,
                    fixed_eval_noise,
                    nfe=args.eval_nfe,
                    batch_size=args.eval_batch_size,
                )
                elapsed = previous_elapsed + time.perf_counter() - run_started
                record = {
                    "step": completed_step,
                    "loss": float(loss.item()),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": elapsed,
                    "eval_nfe": args.eval_nfe,
                    **metrics,
                }
                append_history(args.output_dir, record)
                progress.write(
                    f"eval step={completed_step:,} nfe={args.eval_nfe} "
                    f"in_support={metrics['in_support_rate']:.4f} "
                    f"hist_tv={metrics['histogram_tv']:.4f}"
                )

            should_checkpoint = (
                completed_step % args.checkpoint_every == 0
                or completed_step == args.steps
            )
            if should_checkpoint:
                elapsed = previous_elapsed + time.perf_counter() - run_started
                checkpoint_path = save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    generator=generator,
                    step=completed_step,
                    elapsed_seconds=elapsed,
                    args=args,
                )
                progress.write(f"saved {checkpoint_path}")
    except KeyboardInterrupt:
        progress.close()
        elapsed = previous_elapsed + time.perf_counter() - run_started
        checkpoint_path = save_training_checkpoint(
            model=model,
            optimizer=optimizer,
            ema=ema,
            generator=generator,
            step=completed_step,
            elapsed_seconds=elapsed,
            args=args,
        )
        print(f"\nInterrupted; saved {checkpoint_path}", flush=True)
        return
    finally:
        progress.close()

    elapsed = previous_elapsed + time.perf_counter() - run_started
    ema.copy_to(eval_model)
    metrics = evaluate_teacher(
        eval_model,
        fixed_eval_noise,
        nfe=args.eval_nfe,
        batch_size=args.eval_batch_size,
    )
    teacher_path = save_teacher(
        ema=ema,
        model=model,
        step=completed_step,
        elapsed_seconds=elapsed,
        args=args,
        metrics=metrics,
    )
    print(
        f"Finished step {completed_step:,}; saved EMA teacher to {teacher_path}",
        flush=True,
    )
    print(
        f"NFE={args.eval_nfe} in-support rate: "
        f"{float(metrics['in_support_rate']):.4f} "
        f"({'PASS' if float(metrics['in_support_rate']) > 0.99 else 'BELOW TARGET'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
