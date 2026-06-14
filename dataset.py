from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

DeviceLike = Union[str, torch.device]


def sample_checkerboard(
    num_samples: int,
    *,
    num_cells: int = 4,
    bounds: Tuple[float, float] = (-1.0, 1.0),
    generator: Optional[torch.Generator] = None,
    device: Optional[DeviceLike] = None,
) -> Tensor:
    """Sample points uniformly from the dark cells of a 2D checkerboard."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_cells <= 0:
        raise ValueError("num_cells must be positive")

    lower, upper = bounds
    if lower >= upper:
        raise ValueError("bounds must satisfy lower < upper")
    target_device = torch.device("cpu" if device is None else device)

    rows, columns = torch.meshgrid(
        torch.arange(num_cells, device=target_device),
        torch.arange(num_cells, device=target_device),
        indexing="ij",
    )
    is_dark = (rows + columns).remainder(2) == 0
    dark_cells = torch.stack(
        (columns[is_dark], rows[is_dark]),
        dim=1,
    )
    selected = torch.randint(
        len(dark_cells),
        (num_samples,),
        generator=generator,
        device=target_device,
    )
    offsets = torch.rand(
        num_samples,
        2,
        generator=generator,
        device=target_device,
    )
    cell_size = (upper - lower) / num_cells
    return (lower + (dark_cells[selected] + offsets) * cell_size).float()
