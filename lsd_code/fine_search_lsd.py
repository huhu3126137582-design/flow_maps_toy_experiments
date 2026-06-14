from __future__ import annotations

import argparse
import json
import os
import shutil
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

from common import resolve_device
from dataset import sample_checkerboard
from metrics import histogram_distances, sliced_wasserstein_2, support_membership
from models import FlowMapNet
from sampling import flow_map_sample


NFES = (1, 2, 4, 8)
RANK_METRICS = (
    ("sw2", False),
    ("histogram_jsd", False),
    ("in_support_rate", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best LSD EMA checkpoint across NFE=1/2/4/8.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/lsd"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Directory for metrics, plots, and the selected checkpoint.",
    )
    parser.add_argument("--candidate-min-step", type=int, default=0)
    parser.add_argument("--candidate-max-step", type=int, default=130_000)
    parser.add_argument("--short-samples", type=int, default=10_000)
    parser.add_argument("--full-samples", type=int, default=50_000)
    parser.add_argument("--reference-samples", type=int, default=200_000)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--robust-top-k", type=int, default=4)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(20260614, 20260615, 20260616),
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.candidate_min_step > args.candidate_max_step:
        raise ValueError("candidate step range is empty")
    for name in (
        "short_samples",
        "full_samples",
        "reference_samples",
        "top_k",
        "robust_top_k",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.short_samples > args.full_samples:
        raise ValueError("--short-samples cannot exceed --full-samples")
    if args.full_samples > args.reference_samples:
        raise ValueError("--full-samples cannot exceed --reference-samples")
    if args.robust_top_k > args.top_k:
        raise ValueError("--robust-top-k cannot exceed --top-k")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds values must be unique")


def load_ema(path: Path, device: torch.device) -> FlowMapNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "lsd_training_checkpoint":
        raise ValueError(f"not an LSD checkpoint: {path}")
    model = FlowMapNet(**checkpoint["model_config"]).to(
        device=device,
        dtype=torch.float32,
    )
    model.load_state_dict(checkpoint["ema"]["shadow"], strict=True)
    model.eval()
    return model


def make_evaluation_data(
    *,
    seed: int,
    sample_count: int,
    reference_count: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        sample_count,
        2,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    reference = sample_checkerboard(
        reference_count,
        generator=generator,
        device=device,
    )
    return noise, reference


@torch.no_grad()
def distribution_metrics(
    samples: Tensor,
    reference: Tensor,
    *,
    num_projections: int,
) -> dict[str, float]:
    membership, _ = support_membership(samples)
    metrics = {
        "sw2": sliced_wasserstein_2(
            samples,
            reference,
            num_projections=num_projections,
            num_quantiles=min(10_000, samples.shape[0]),
        ),
        "in_support_rate": float(membership.float().mean().item()),
    }
    metrics.update(histogram_distances(samples))
    return metrics


@torch.no_grad()
def evaluate_checkpoint(
    path: Path,
    noise: Tensor,
    reference: Tensor,
    *,
    device: torch.device,
    num_projections: int,
) -> dict[str, dict[str, float]]:
    model = load_ema(path, device)
    metrics_by_nfe: dict[str, dict[str, float]] = {}
    for nfe in NFES:
        samples = flow_map_sample(
            model,
            noise,
            nfe=nfe,
            batch_size=10_000,
        )
        metrics_by_nfe[str(nfe)] = distribution_metrics(
            samples,
            reference,
            num_projections=num_projections,
        )
    return metrics_by_nfe


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows

    rank_totals = np.zeros(len(rows), dtype=np.float64)
    criteria = [
        (str(nfe), metric, descending)
        for nfe in NFES
        for metric, descending in RANK_METRICS
    ]
    for nfe, metric, descending in criteria:
        values = np.asarray(
            [float(row["metrics_by_nfe"][nfe][metric]) for row in rows],
            dtype=np.float64,
        )
        order = np.argsort(-values if descending else values, kind="stable")
        ranks = np.empty(len(rows), dtype=np.float64)
        ranks[order] = np.arange(len(rows), dtype=np.float64)
        if len(rows) > 1:
            ranks /= len(rows) - 1
        rank_totals += ranks

    for index, row in enumerate(rows):
        metrics = row["metrics_by_nfe"]
        row["composite_score"] = float(rank_totals[index] / len(criteria))
        row["mean_sw2"] = float(
            np.mean([float(metrics[str(nfe)]["sw2"]) for nfe in NFES])
        )
        row["mean_histogram_jsd"] = float(
            np.mean(
                [
                    float(metrics[str(nfe)]["histogram_jsd"])
                    for nfe in NFES
                ]
            )
        )
        row["mean_in_support_rate"] = float(
            np.mean(
                [
                    float(metrics[str(nfe)]["in_support_rate"])
                    for nfe in NFES
                ]
            )
        )

    rows.sort(
        key=lambda row: (
            float(row["composite_score"]),
            float(row["mean_sw2"]),
            float(row["mean_histogram_jsd"]),
            -float(row["mean_in_support_rate"]),
        )
    )
    return rows


def average_seed_metrics(
    runs: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for nfe in map(str, NFES):
        result[nfe] = {
            metric: float(np.mean([run[nfe][metric] for run in runs]))
            for metric in runs[0][nfe]
        }
    return result


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(portable_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


@torch.no_grad()
def save_distribution_visualizations(
    model: FlowMapNet,
    noise: Tensor,
    reference: Tensor,
    *,
    step: int,
    output_dir: Path,
) -> None:
    series = [("True data", reference[: noise.shape[0]].detach().cpu().numpy())]
    for nfe in NFES:
        samples = flow_map_sample(
            model,
            noise,
            nfe=nfe,
            batch_size=10_000,
        )
        series.append((f"NFE={nfe}", samples.detach().cpu().numpy()))

    stem = "best_checkpoint_nfe_1_2_4_8"
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
        f"True data and LSD {step // 1000}k EMA - "
        f"fixed {noise.shape[0] // 1000}k samples",
        fontsize=15,
    )
    figure.savefig(output_dir / f"{stem}_scatter.png", dpi=180)
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
            title if title == "True data" else f"LSD {step // 1000}k EMA, {title}"
        )
        axis.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    figure.colorbar(
        image,
        ax=axes,
        label="Sample count per bin",
        shrink=0.92,
    )
    figure.savefig(output_dir / f"{stem}_hist2d.png", dpi=180)
    plt.close(figure)


def copy_selected_checkpoint(path: Path, output_dir: Path) -> Path:
    destination = output_dir / path.name
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return destination


def main() -> None:
    args = parse_args()
    validate_args(args)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    artifact_dir = (
        checkpoint_dir
        if args.artifact_dir is None
        else args.artifact_dir.expanduser().resolve()
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
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

    first_seed = args.seeds[0]
    full_noise, full_reference = make_evaluation_data(
        seed=first_seed,
        sample_count=args.full_samples,
        reference_count=args.reference_samples,
        device=device,
    )

    short_ranked = []
    for step, path in candidates:
        metrics_by_nfe = evaluate_checkpoint(
            path,
            full_noise[: args.short_samples],
            full_reference[: args.short_samples],
            device=device,
            num_projections=128,
        )
        short_ranked.append(
            {
                "step": step,
                "path": str(path),
                "metrics_by_nfe": metrics_by_nfe,
            }
        )
        print(f"short step={step:,}", flush=True)
    rank_rows(short_ranked)

    full_ranked = []
    for candidate in short_ranked[: args.top_k]:
        metrics_by_nfe = evaluate_checkpoint(
            Path(candidate["path"]),
            full_noise,
            full_reference,
            device=device,
            num_projections=512,
        )
        full_ranked.append(
            {
                "step": candidate["step"],
                "path": candidate["path"],
                "metrics_by_nfe": metrics_by_nfe,
            }
        )
        print(f"full step={candidate['step']:,}", flush=True)
    rank_rows(full_ranked)

    robust_candidates = full_ranked[: args.robust_top_k]
    runs_by_step = {
        int(row["step"]): [row["metrics_by_nfe"]]
        for row in robust_candidates
    }
    for seed in args.seeds[1:]:
        noise, reference = make_evaluation_data(
            seed=seed,
            sample_count=args.full_samples,
            reference_count=args.reference_samples,
            device=device,
        )
        for candidate in robust_candidates:
            step = int(candidate["step"])
            runs_by_step[step].append(
                evaluate_checkpoint(
                    Path(candidate["path"]),
                    noise,
                    reference,
                    device=device,
                    num_projections=512,
                )
            )
            print(f"robust seed={seed} step={step:,}", flush=True)

    robust_ranked = []
    for candidate in robust_candidates:
        step = int(candidate["step"])
        robust_ranked.append(
            {
                "step": step,
                "path": candidate["path"],
                "metrics_by_nfe": average_seed_metrics(runs_by_step[step]),
                "per_seed": runs_by_step[step],
            }
        )
    rank_rows(robust_ranked)

    best = robust_ranked[0]
    selected_path = copy_selected_checkpoint(
        Path(best["path"]),
        artifact_dir,
    )
    best["selected_checkpoint"] = str(selected_path)
    coarse_steps = [int(row["step"]) for row in short_ranked[: args.top_k]]
    result = {
        "selection_rule": {
            "nfes": list(NFES),
            "metrics": [
                "SW2 ascending",
                "histogram JSD ascending",
                "in-support rate descending",
            ],
            "aggregation": (
                "equal-weight mean normalized rank over per-seed metric means"
            ),
        },
        "seeds": list(args.seeds),
        "short_samples": args.short_samples,
        "full_samples_per_seed": args.full_samples,
        "coarse_candidate_interval": [min(coarse_steps), max(coarse_steps)],
        "best": best,
        "robust_ranked": robust_ranked,
        "full_ranked": full_ranked,
        "short_ranked": short_ranked,
    }
    atomic_json_save(
        result,
        artifact_dir / "best_checkpoint_nfe_1_2_4_8_metrics.json",
    )

    best_model = load_ema(selected_path, device)
    save_distribution_visualizations(
        best_model,
        full_noise,
        full_reference,
        step=int(best["step"]),
        output_dir=artifact_dir,
    )
    print(f"Selected {selected_path}", flush=True)
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
