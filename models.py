from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor, nn

TimeLike = Union[float, Tensor]


class GaussianFourierTimeEmbedding(nn.Module):
    """Fixed Gaussian Fourier features followed by a learned projection."""

    def __init__(self, embedding_dim: int = 64, scale: float = 16.0) -> None:
        super().__init__()
        if embedding_dim <= 0 or embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be a positive even number")
        frequencies = torch.randn(embedding_dim // 2) * scale
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.SiLU()

    def forward(self, time: Tensor) -> Tensor:
        angles = 2.0 * math.pi * time.unsqueeze(-1) * self.frequencies
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.activation(self.projection(features))


class FlowMapNet(nn.Module):
    """Shared average-velocity network used by all flow-map experiments."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        num_hidden_layers: int = 6,
        time_embedding_dim: int = 64,
        fourier_scale: float = 16.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")

        self.config = {
            "hidden_dim": hidden_dim,
            "num_hidden_layers": num_hidden_layers,
            "time_embedding_dim": time_embedding_dim,
            "fourier_scale": fourier_scale,
        }
        self.start_time_embedding = GaussianFourierTimeEmbedding(
            time_embedding_dim,
            fourier_scale,
        )
        self.end_time_embedding = GaussianFourierTimeEmbedding(
            time_embedding_dim,
            fourier_scale,
        )

        layers: list[nn.Module] = []
        input_dim = 2 + 2 * time_embedding_dim
        for layer_index in range(num_hidden_layers):
            layers.append(
                nn.Linear(
                    input_dim if layer_index == 0 else hidden_dim,
                    hidden_dim,
                )
            )
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    @staticmethod
    def _broadcast_time(time: TimeLike, x: Tensor) -> Tensor:
        value = torch.as_tensor(time, device=x.device, dtype=x.dtype)
        target_shape = x.shape[:-1]
        if value.shape == target_shape + (1,):
            value = value.squeeze(-1)
        try:
            return torch.broadcast_to(value, target_shape)
        except RuntimeError as error:
            raise ValueError(
                f"time shape {tuple(value.shape)} cannot broadcast to "
                f"x batch shape {tuple(target_shape)}"
            ) from error

    def forward(self, x: Tensor, s: TimeLike, t: TimeLike) -> Tensor:
        if x.shape[-1] != 2:
            raise ValueError("x must have final dimension 2")
        start = self._broadcast_time(s, x)
        end = self._broadcast_time(t, x)
        inputs = torch.cat(
            (
                x,
                self.start_time_embedding(start),
                self.end_time_embedding(end),
            ),
            dim=-1,
        )
        return self.network(inputs)

    def flow(self, x: Tensor, s: TimeLike, t: TimeLike) -> Tensor:
        start = self._broadcast_time(s, x)
        end = self._broadcast_time(t, x)
        average_velocity = self(x, start, end)
        return x + (end - start).unsqueeze(-1) * average_velocity
