from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def _validate_sampling_inputs(
    noise: Tensor,
    *,
    nfe: int,
    batch_size: Optional[int],
) -> int:
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    if noise.ndim != 2 or noise.shape[1] != 2:
        raise ValueError("noise must have shape [num_samples, 2]")
    sample_batch_size = noise.shape[0] if batch_size is None else batch_size
    if sample_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return sample_batch_size


@torch.no_grad()
def euler_sample_teacher(
    model: nn.Module,
    noise: Tensor,
    *,
    nfe: int,
    batch_size: Optional[int] = None,
) -> Tensor:
    """Integrate the teacher velocity field from noise at t=1 to data at t=0."""
    sample_batch_size = _validate_sampling_inputs(
        noise,
        nfe=nfe,
        batch_size=batch_size,
    )

    samples = noise.clone()
    time_step = -1.0 / nfe
    for index in range(nfe):
        time = 1.0 - index / nfe
        for start in range(0, samples.shape[0], sample_batch_size):
            stop = min(start + sample_batch_size, samples.shape[0])
            chunk = samples[start:stop]
            velocity = model(chunk, time, time)
            samples[start:stop] = chunk + time_step * velocity
    return samples


@torch.no_grad()
def flow_map_sample(
    model: nn.Module,
    noise: Tensor,
    *,
    nfe: int,
    batch_size: Optional[int] = None,
) -> Tensor:
    """Sample from t=1 to t=0 using equal flow-map time segments."""
    sample_batch_size = _validate_sampling_inputs(
        noise,
        nfe=nfe,
        batch_size=batch_size,
    )
    samples = noise.clone()
    for index in range(nfe):
        start_time = 1.0 - index / nfe
        end_time = 1.0 - (index + 1) / nfe
        for start in range(0, samples.shape[0], sample_batch_size):
            stop = min(start + sample_batch_size, samples.shape[0])
            samples[start:stop] = model.flow(
                samples[start:stop],
                start_time,
                end_time,
            )
    return samples
