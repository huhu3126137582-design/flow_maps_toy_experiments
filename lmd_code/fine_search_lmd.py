from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from common import atomic_torch_save, resolve_device
from dataset import sample_checkerboard
from metrics import evaluate_distribution, sliced_wasserstein_2, support_membership
from models import FlowMapNet
from sampling import flow_map_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best SG-LMD EMA checkpoint near 70k.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/lmd"),
    )
    parser.add_argument("--candidate-min-step", type=int, default=65_000)
    parser.add_argument("--candidate-max-step", type=int, default=75_000)
    parser.add_argument("--short-samples", type=int, default=10_000)
    parser.add_argument("--full-samples", type=int, default=50_000)
    parser.add_argument("--reference-samples", type=int, default=200_000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.candidate_min_step > args.candidate_max_step:
        raise ValueError("candidate step range is empty")
    for name in ("short_samples", "full_samples", "reference_samples", "top_k"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.short_samples > args.full_samples:
        raise ValueError("--short-samples cannot exceed --full-samples")
    if args.full_samples > args.reference_samples:
        raise ValueError("--full-samples cannot exceed --reference-samples")


def load_ema(path: Path, device: torch.device) -> FlowMapNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FlowMapNet(**checkpoint["model_config"]).to(
        device=device,
        dtype=torch.float32,
    )
    model.load_state_dict(checkpoint["ema"]["shadow"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def ranking_metrics(samples: Tensor, reference: Tensor) -> dict[str, float]:
    membership, _ = support_membership(samples)
    return {
        "sw2": sliced_wasserstein_2(
            samples,
            reference,
            num_projections=128,
            num_quantiles=min(10_000, samples.shape[0]),
        ),
        "in_support_rate": float(membership.float().mean().item()),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, float]:
    one_nfe = row["metrics_by_nfe"]["1"]
    return float(one_nfe["sw2"]), -float(one_nfe["in_support_rate"])


def portable_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: portable_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_json_value(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return candidate.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                return candidate.name
    return value


def atomic_json_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(portable_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def save_visualizations(
    model: FlowMapNet,
    noise: Tensor,
    reference: Tensor,
    *,
    step: int,
    output_dir: Path,
) -> None:
    series = [("True data", reference[: noise.shape[0]].detach().cpu().numpy())]
    for nfe in (1, 2, 4, 8):
        samples = flow_map_sample(
            model,
            noise,
            nfe=nfe,
            batch_size=10_000,
        )
        series.append((f"NFE={nfe}", samples.detach().cpu().numpy()))

    stem = f"best_checkpoint_{step:06d}_nfe_1_2_4_8"
    scatter_path = output_dir / f"{stem}_scatter.png"
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, (title, values) in zip(axes, series):
        axis.scatter(
            values[:, 0],
            values[:, 1],
            s=0.35,
            alpha=0.28,
            rasterized=True,
            color="#1769aa",
        )
        axis.set_title(title)
        axis.set_xlim(-1.2, 1.2)
        axis.set_ylim(-1.2, 1.2)
        axis.set_aspect("equal")
        axis.set_xlabel("x1")
        axis.grid(alpha=0.12)
    axes[0].set_ylabel("x2")
    figure.suptitle(
        f"True data and SG-LMD {step // 1000}k EMA - fixed 50k samples",
        fontsize=15,
    )
    figure.savefig(scatter_path, dpi=180)
    plt.close(figure)

    histograms = [
        np.histogram2d(
            values[:, 0],
            values[:, 1],
            bins=96,
            range=((-1.2, 1.2), (-1.2, 1.2)),
        )[0]
        for _, values in series
    ]
    maximum = max(float(histogram.max()) for histogram in histograms)
    histogram_path = output_dir / f"{stem}_hist2d.png"
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(21, 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for axis, (title, _), histogram in zip(axes, series, histograms):
        image = axis.imshow(
            histogram.T,
            origin="lower",
            extent=(-1.2, 1.2, -1.2, 1.2),
            cmap="viridis",
            vmin=0,
            vmax=maximum,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(
            title if title == "True data" else f"SG-LMD {step // 1000}k EMA, {title}"
        )
        axis.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    figure.colorbar(
        image,
        ax=axes,
        label="Sample count per bin",
        shrink=0.92,
    )
    figure.savefig(histogram_path, dpi=180)
    plt.close(figure)


def save_selected_model(
    model: FlowMapNet,
    best: dict[str, Any],
    output_dir: Path,
) -> Path:
    path = output_dir / "best_lmd_sg.pt"
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    atomic_torch_save(
        {
            "format_version": 1,
            "kind": "sg_lmd_selected",
            "step": int(best["step"]),
            "source_checkpoint": Path(best["path"]).name,
            "model_config": model.config,
            "model": state_dict,
            "metrics_by_nfe": best["metrics_by_nfe"],
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    validate_args(args)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    device = resolve_device(args.device)

    candidates = []
    for path in sorted(checkpoint_dir.glob("checkpoint_[0-9]*.pt")):
        step = int(path.stem.split("_")[-1])
        if args.candidate_min_step <= step <= args.candidate_max_step:
            candidates.append((step, path))
    if not candidates:
        raise FileNotFoundError("no checkpoints found in the requested range")
    if args.top_k > len(candidates):
        raise ValueError("--top-k exceeds the number of candidate checkpoints")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        args.full_samples,
        2,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    reference = sample_checkerboard(
        args.reference_samples,
        generator=generator,
        device=device,
    )

    short_ranked = []
    for step, path in candidates:
        model = load_ema(path, device)
        metrics_by_nfe = {}
        for nfe in (1, 2, 4):
            samples = flow_map_sample(
                model,
                noise[: args.short_samples],
                nfe=nfe,
                batch_size=10_000,
            )
            metrics_by_nfe[str(nfe)] = ranking_metrics(
                samples,
                reference[: args.short_samples],
            )
        short_ranked.append(
            {
                "step": step,
                "path": str(path),
                "metrics_by_nfe": metrics_by_nfe,
            }
        )
        del model
    short_ranked.sort(key=rank_key)

    full_ranked = []
    for candidate in short_ranked[: args.top_k]:
        model = load_ema(Path(candidate["path"]), device)
        metrics_by_nfe = {}
        for nfe in (1, 2, 4):
            samples = flow_map_sample(
                model,
                noise,
                nfe=nfe,
                batch_size=10_000,
            )
            metrics_by_nfe[str(nfe)] = evaluate_distribution(
                samples,
                reference,
            )
        full_ranked.append(
            {
                "step": candidate["step"],
                "path": candidate["path"],
                "metrics_by_nfe": metrics_by_nfe,
            }
        )
        del model
    full_ranked.sort(key=rank_key)

    result = {
        "selection_rule": ["1-NFE SW2", "-1-NFE in-support"],
        "best": full_ranked[0],
        "full_ranked": full_ranked,
        "short_ranked": short_ranked,
    }
    atomic_json_save(result, checkpoint_dir / "selection_metrics.json")

    best = full_ranked[0]
    best_model = load_ema(Path(best["path"]), device)
    save_selected_model(best_model, best, checkpoint_dir)
    save_visualizations(
        best_model,
        noise,
        reference,
        step=int(best["step"]),
        output_dir=checkpoint_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
