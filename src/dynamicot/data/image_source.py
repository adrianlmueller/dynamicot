from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def build_image_source_data(
    source_cfg: dict[str, Any] | None,
    *,
    n_samples: int,
    width: int,
    input_channels: int,
    device: torch.device,
    dtype: torch.dtype,
):
    source_cfg = source_cfg or {}
    source_type = str(source_cfg.get("type", "uniform")).lower()

    if source_type == "uniform":
        return torch.rand(n_samples, input_channels * width * width, device=device, dtype=dtype)

    if source_type == "upsampled_uniform":
        base_width = int(source_cfg.get("base_width", 8))
        resize_mode = str(source_cfg.get("resize_mode", "bicubic")).lower()
        antialias = bool(source_cfg.get("antialias", True))

        if base_width <= 0:
            raise ValueError("source.base_width must be positive")

        low_res = torch.rand(n_samples, input_channels, base_width, base_width, device=device, dtype=dtype)
        if resize_mode == "nearest":
            upsampled = F.interpolate(low_res, size=(width, width), mode=resize_mode)
        elif resize_mode in {"bilinear", "bicubic"}:
            upsampled = F.interpolate(
                low_res,
                size=(width, width),
                mode=resize_mode,
                align_corners=False,
                antialias=antialias,
            )
        else:
            raise ValueError(
                f"Unsupported source.resize_mode '{resize_mode}'. "
                "Use 'nearest', 'bilinear', or 'bicubic'."
            )
        return upsampled.view(n_samples, input_channels * width * width)

    raise ValueError(
        f"Unsupported source.type '{source_type}'. "
        "Use 'uniform' or 'upsampled_uniform'."
    )
