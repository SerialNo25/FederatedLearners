"""Utilities for selecting the best available torch device."""

from __future__ import annotations

import torch


class DeviceSelector:
    """Selects the preferred compute device in priority order CUDA > MPS > CPU."""

    _PRIORITY: tuple[str, ...] = ("cuda", "mps", "cpu")

    def available_devices(self) -> list[str]:
        devices: list[str] = []
        if torch.cuda.is_available():
            devices.append("cuda")

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            devices.append("mps")

        devices.append("cpu")
        return devices

    def select_best_device(self) -> str:
        available = set(self.available_devices())
        for device in self._PRIORITY:
            if device in available:
                return device
        return "cpu"
