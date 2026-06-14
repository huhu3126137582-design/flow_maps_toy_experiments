from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn

DeviceLike = Union[str, torch.device]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def configure_float32(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def cosine_learning_rate(
    step: int,
    *,
    decay_steps: int,
    initial_lr: float,
    final_lr: float,
) -> float:
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive")
    if not 0.0 <= final_lr <= initial_lr:
        raise ValueError("learning rates must satisfy 0 <= final <= initial")
    if decay_steps == 1:
        return final_lr
    progress = min(max(step, 0) / (decay_steps - 1), 1.0)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_lr + (initial_lr - final_lr) * multiplier


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")
        self.decay = decay
        self.num_updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        if model_state.keys() != self.shadow.keys():
            raise ValueError("model state does not match EMA state")
        for name, value in model_state.items():
            shadow_value = self.shadow[name]
            if torch.is_floating_point(shadow_value):
                shadow_value.lerp_(value.detach(), 1.0 - self.decay)
            else:
                shadow_value.copy_(value.detach())
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        shadow = state_dict["shadow"]
        if shadow.keys() != self.shadow.keys():
            raise ValueError("checkpoint EMA state does not match model")
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict.get("num_updates", 0))
        for name, value in shadow.items():
            self.shadow[name].copy_(value)


def capture_random_state(
    *,
    generator: Optional[torch.Generator] = None,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if generator is not None:
        state["generator"] = generator.get_state()
    return state


def restore_random_state(
    state: Mapping[str, Any],
    *,
    generator: Optional[torch.Generator] = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [cuda_state.detach().cpu() for cuda_state in state["cuda"]]
        )
    if generator is not None and "generator" in state:
        try:
            generator.set_state(state["generator"].detach().cpu())
        except RuntimeError as error:
            raise ValueError(
                "checkpoint generator state is incompatible with the selected "
                f"device ({generator.device}); resume on the original device type"
            ) from error


def atomic_torch_save(payload: Any, path: Union[str, Path]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def latest_checkpoint(output_dir: Union[str, Path]) -> Optional[Path]:
    checkpoints = sorted(Path(output_dir).glob("checkpoint_[0-9]*.pt"))
    return checkpoints[-1] if checkpoints else None


def move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)
