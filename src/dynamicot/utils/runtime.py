from __future__ import annotations

import random
import subprocess

import numpy as np
import torch


def _macos_version():
    try:
        return subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def describe_accelerators():
    lines = [
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA build: {torch.version.cuda}",
        f"CUDA devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
    ]
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            lines.append(f"  cuda:{index}: {torch.cuda.get_device_name(index)}")

    lines.extend(
        [
            f"MPS built: {torch.backends.mps.is_built()}",
            f"MPS available: {torch.backends.mps.is_available()}",
            f"macOS: {_macos_version()}",
        ]
    )
    return "\n".join(lines)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(name: str):
    name = str(name or "cpu").lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("GPU was requested, but no CUDA or MPS device is available.\n" + describe_accelerators())

    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch reports CUDA is unavailable.\n" + describe_accelerators())
        device = torch.device(name)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"{name} was requested, but only {torch.cuda.device_count()} CUDA device(s) are visible.")
        return device

    if name == "mps":
        if not torch.backends.mps.is_built():
            raise RuntimeError("PyTorch was not built with MPS support.")
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested, but PyTorch reports it is unavailable. "
                f"macOS={_macos_version()}, mps_built={torch.backends.mps.is_built()}, "
                f"mps_available={torch.backends.mps.is_available()}."
            )
        return torch.device("mps")

    try:
        return torch.device(name)
    except RuntimeError as error:
        raise ValueError(f"Unsupported device: {name}") from error


def resolve_dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]
