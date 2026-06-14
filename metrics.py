from __future__ import annotations

import math
from typing import Optional, Sequence
import numpy as np
import torch
from torch import Tensor


BOUNDS = (-1.0, 1.0)
NUM_CELLS = 4
CELL_SIZE = 0.5


def dark_cell_indices(
    num_cells: int = NUM_CELLS,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    rows, columns = torch.meshgrid(
        torch.arange(num_cells, device=device),
        torch.arange(num_cells, device=device),
        indexing="ij",
    )
    mask = (rows + columns).remainder(2) == 0
    return torch.stack((columns[mask], rows[mask]), dim=-1)


def support_membership(
    samples: Tensor,
    *,
    bounds: tuple[float, float] = BOUNDS,
    num_cells: int = NUM_CELLS,
) -> tuple[Tensor, Tensor]:
    lower, upper = bounds
    cell_size = (upper - lower) / num_cells
    in_bounds = ((samples >= lower) & (samples <= upper)).all(dim=-1)
    cells = torch.floor((samples - lower) / cell_size).long()
    cells = cells.clamp(0, num_cells - 1)
    is_dark = cells.sum(dim=-1).remainder(2) == 0
    return is_dark & in_bounds, cells


def distance_to_support(
    samples: Tensor,
    *,
    bounds: tuple[float, float] = BOUNDS,
    num_cells: int = NUM_CELLS,
) -> Tensor:
    lower, upper = bounds
    cell_size = (upper - lower) / num_cells
    cells = dark_cell_indices(num_cells, device=samples.device).to(samples.dtype)
    rectangle_lower = lower + cells * cell_size
    rectangle_upper = rectangle_lower + cell_size
    points = samples[:, None, :]
    coordinate_distance = torch.maximum(
        torch.maximum(rectangle_lower[None] - points, points - rectangle_upper[None]),
        torch.zeros((), device=samples.device, dtype=samples.dtype),
    )
    return coordinate_distance.square().sum(dim=-1).sqrt().min(dim=1).values


def _kl_to_uniform(probabilities: Tensor) -> float:
    positive = probabilities > 0
    if not positive.any():
        return float("inf")
    values = probabilities[positive]
    uniform_probability = 1.0 / probabilities.numel()
    return float((values * (values / uniform_probability).log()).sum().item())


def mode_statistics(
    samples: Tensor,
    *,
    bounds: tuple[float, float] = BOUNDS,
    num_cells: int = NUM_CELLS,
) -> dict[str, float | int | list[float]]:
    membership, cells = support_membership(
        samples,
        bounds=bounds,
        num_cells=num_cells,
    )
    dark_cells = dark_cell_indices(num_cells, device=samples.device)
    counts = torch.zeros(
        dark_cells.shape[0],
        device=samples.device,
        dtype=torch.float64,
    )
    for index, cell in enumerate(dark_cells):
        counts[index] = (
            membership & (cells == cell).all(dim=-1)
        ).sum()
    proportions_all = counts / samples.shape[0]
    supported_count = counts.sum()
    conditional = (
        counts / supported_count
        if supported_count > 0
        else torch.zeros_like(counts)
    )
    return {
        "mode_coverage": int((proportions_all > 0.01).sum().item()),
        "mode_kl": _kl_to_uniform(conditional),
        "mode_proportions": [float(value) for value in proportions_all.cpu()],
    }


def _js_divergence(probabilities: Tensor, target: Tensor) -> Tensor:
    eps = torch.finfo(probabilities.dtype).eps
    probabilities = probabilities.clamp_min(eps)
    target = target.clamp_min(eps)
    midpoint = 0.5 * (probabilities + target)
    return 0.5 * (
        (probabilities * (probabilities / midpoint).log()).sum()
        + (target * (target / midpoint).log()).sum()
    )


def within_cell_uniformity_jsd(
    samples: Tensor,
    *,
    subdivisions: int = 8,
    bounds: tuple[float, float] = BOUNDS,
    num_cells: int = NUM_CELLS,
) -> float:
    lower, upper = bounds
    cell_size = (upper - lower) / num_cells
    membership, cells = support_membership(
        samples,
        bounds=bounds,
        num_cells=num_cells,
    )
    dark_cells = dark_cell_indices(num_cells, device=samples.device)
    uniform = torch.full(
        (subdivisions * subdivisions,),
        1.0 / (subdivisions * subdivisions),
        device=samples.device,
        dtype=torch.float64,
    )
    divergences = []
    for cell in dark_cells:
        mask = membership & (cells == cell).all(dim=-1)
        if not mask.any():
            divergences.append(torch.tensor(math.log(2.0), device=samples.device))
            continue
        local = (samples[mask] - (lower + cell * cell_size)) / cell_size
        subcells = torch.floor(local * subdivisions).long().clamp(
            0,
            subdivisions - 1,
        )
        flat = subcells[:, 1] * subdivisions + subcells[:, 0]
        counts = torch.bincount(
            flat,
            minlength=subdivisions * subdivisions,
        ).to(torch.float64)
        probabilities = counts / counts.sum()
        divergences.append(_js_divergence(probabilities, uniform))
    return float(torch.stack(divergences).mean().item())


def _analytic_histogram_probabilities(
    edges: np.ndarray,
    *,
    bounds: tuple[float, float] = BOUNDS,
    num_cells: int = NUM_CELLS,
) -> np.ndarray:
    lower, upper = bounds
    cell_size = (upper - lower) / num_cells
    dark_cells = dark_cell_indices(num_cells).cpu().numpy()
    probabilities = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.float64)
    for column, row in dark_cells:
        rect_x0 = lower + column * cell_size
        rect_y0 = lower + row * cell_size
        rect_x1 = rect_x0 + cell_size
        rect_y1 = rect_y0 + cell_size
        x_overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], rect_x1) - np.maximum(edges[:-1], rect_x0),
        )
        y_overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], rect_y1) - np.maximum(edges[:-1], rect_y0),
        )
        probabilities += 0.5 * np.outer(x_overlap, y_overlap)
    return probabilities


def histogram_distances(
    samples: Tensor,
    *,
    histogram_bounds: tuple[float, float] = (-1.2, 1.2),
    bins: int = 96,
) -> dict[str, float]:
    values = samples.detach().cpu().numpy()
    edges = np.linspace(histogram_bounds[0], histogram_bounds[1], bins + 1)
    counts, _, _ = np.histogram2d(values[:, 0], values[:, 1], bins=(edges, edges))
    inside = (
        (values[:, 0] >= histogram_bounds[0])
        & (values[:, 0] <= histogram_bounds[1])
        & (values[:, 1] >= histogram_bounds[0])
        & (values[:, 1] <= histogram_bounds[1])
    )
    generated = np.concatenate(
        (counts.reshape(-1), np.array([np.count_nonzero(~inside)]))
    )
    generated /= generated.sum()
    target_inside = _analytic_histogram_probabilities(edges).reshape(-1)
    target = np.concatenate((target_inside, np.array([0.0])))
    target /= target.sum()

    total_variation = 0.5 * np.abs(generated - target).sum()
    midpoint = 0.5 * (generated + target)

    def relative_entropy(probability: np.ndarray, reference: np.ndarray) -> float:
        mask = probability > 0
        return float(
            np.sum(probability[mask] * np.log(probability[mask] / reference[mask]))
        )

    jsd = 0.5 * (
        relative_entropy(generated, midpoint)
        + relative_entropy(target, midpoint)
    )
    return {
        "histogram_tv": float(total_variation),
        "histogram_jsd": float(jsd),
        "overflow_rate": float(1.0 - inside.mean()),
    }


def _sample_sorted_quantiles(sorted_values: Tensor, count: int) -> Tensor:
    if sorted_values.shape[0] == count:
        return sorted_values
    positions = torch.linspace(
        0,
        sorted_values.shape[0] - 1,
        count,
        device=sorted_values.device,
    )
    lower = positions.floor().long()
    upper = positions.ceil().long()
    weight = (positions - lower).unsqueeze(-1)
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


@torch.no_grad()
def sliced_wasserstein_2(
    samples: Tensor,
    reference: Tensor,
    *,
    num_projections: int = 512,
    seed: int = 0,
    projection_batch_size: int = 32,
    num_quantiles: int = 10_000,
) -> float:
    if samples.shape[1] != 2 or reference.shape[1] != 2:
        raise ValueError("samples and reference must have shape [n, 2]")
    generator = torch.Generator(device=samples.device)
    generator.manual_seed(seed)
    directions = torch.randn(
        num_projections,
        2,
        generator=generator,
        device=samples.device,
        dtype=samples.dtype,
    )
    directions /= directions.norm(dim=-1, keepdim=True)
    quantile_count = min(num_quantiles, samples.shape[0], reference.shape[0])
    squared_distances = []
    for start in range(0, num_projections, projection_batch_size):
        direction_batch = directions[start : start + projection_batch_size]
        generated_projection = (samples @ direction_batch.T).sort(dim=0).values
        reference_projection = (reference @ direction_batch.T).sort(dim=0).values
        generated_quantiles = _sample_sorted_quantiles(
            generated_projection,
            quantile_count,
        )
        reference_quantiles = _sample_sorted_quantiles(
            reference_projection,
            quantile_count,
        )
        squared_distances.append(
            (generated_quantiles - reference_quantiles).square().mean(dim=0)
        )
    return float(torch.cat(squared_distances).mean().sqrt().item())


def _kernel_sums(
    first: Tensor,
    second: Tensor,
    bandwidths: Tensor,
    *,
    block_size: int,
    exclude_diagonal: bool,
) -> Tensor:
    sums = torch.zeros(
        bandwidths.numel(),
        device=first.device,
        dtype=torch.float64,
    )
    same_tensor = first.data_ptr() == second.data_ptr()
    for first_start in range(0, first.shape[0], block_size):
        first_block = first[first_start : first_start + block_size]
        for second_start in range(0, second.shape[0], block_size):
            second_block = second[second_start : second_start + block_size]
            distance_squared = torch.cdist(first_block, second_block).square()
            kernels = torch.exp(
                -distance_squared.unsqueeze(0)
                / (2.0 * bandwidths[:, None, None].square())
            )
            if exclude_diagonal and same_tensor and first_start == second_start:
                diagonal = torch.arange(
                    min(first_block.shape[0], second_block.shape[0]),
                    device=first.device,
                )
                kernels[:, diagonal, diagonal] = 0.0
            sums += kernels.sum(dim=(1, 2), dtype=torch.float64)
    return sums


@torch.no_grad()
def mmd_rbf(
    samples: Tensor,
    reference: Tensor,
    *,
    bandwidths: Sequence[float] = (0.05, 0.1, 0.2, 0.5),
    max_samples: int = 10_000,
    block_size: int = 1024,
) -> tuple[float, list[float]]:
    first = samples[:max_samples]
    second = reference[:max_samples]
    sigma = torch.tensor(
        bandwidths,
        device=first.device,
        dtype=first.dtype,
    )
    xx = _kernel_sums(
        first,
        first,
        sigma,
        block_size=block_size,
        exclude_diagonal=True,
    )
    yy = _kernel_sums(
        second,
        second,
        sigma,
        block_size=block_size,
        exclude_diagonal=True,
    )
    xy = _kernel_sums(
        first,
        second,
        sigma,
        block_size=block_size,
        exclude_diagonal=False,
    )
    n = first.shape[0]
    m = second.shape[0]
    squared = xx / (n * (n - 1)) + yy / (m * (m - 1)) - 2.0 * xy / (n * m)
    squared = squared.clamp_min(0.0)
    values = squared.sqrt()
    return float(values.mean().item()), [float(value) for value in values.cpu()]


@torch.no_grad()
def evaluate_distribution(
    samples: Tensor,
    reference: Tensor,
    *,
    metric_device: Optional[torch.device] = None,
    sw2_projections: int = 512,
    mmd_samples: int = 10_000,
) -> dict[str, float | int | list[float]]:
    if metric_device is None:
        metric_device = samples.device
    generated = samples.detach().to(metric_device, dtype=torch.float32)
    target = reference.detach().to(metric_device, dtype=torch.float32)
    membership, _ = support_membership(generated)
    distances = distance_to_support(generated)
    outside_distances = distances[~membership]
    metrics: dict[str, float | int | list[float]] = {
        "num_samples": generated.shape[0],
        "in_support_rate": float(membership.float().mean().item()),
        "outside_distance_mean": (
            float(outside_distances.mean().item())
            if outside_distances.numel()
            else 0.0
        ),
        "outside_distance_p95": (
            float(torch.quantile(outside_distances, 0.95).item())
            if outside_distances.numel()
            else 0.0
        ),
        "within_cell_jsd": within_cell_uniformity_jsd(generated),
    }
    metrics.update(mode_statistics(generated))
    metrics.update(histogram_distances(generated))
    metrics["sw2"] = sliced_wasserstein_2(
        generated,
        target,
        num_projections=sw2_projections,
    )
    mmd_mean, mmd_values = mmd_rbf(
        generated,
        target,
        max_samples=mmd_samples,
    )
    metrics["mmd_rbf"] = mmd_mean
    metrics["mmd_by_bandwidth"] = mmd_values
    return metrics
