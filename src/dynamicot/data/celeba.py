from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .image_source import build_image_source_data


def _center_crop_tensor(image: torch.Tensor, crop_size: int):
    height, width = image.shape[-2:]
    if crop_size > min(height, width):
        raise ValueError(f"center_crop={crop_size} exceeds image size {height}x{width}.")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[..., top : top + crop_size, left : left + crop_size]


def _downsample_batch(images: torch.Tensor, *, width: int, resize_mode: str, antialias: bool):
    if resize_mode == "area":
        return F.adaptive_avg_pool2d(images, (width, width))
    if resize_mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            images,
            size=(width, width),
            mode=resize_mode,
            align_corners=False,
            antialias=antialias,
        )
    raise ValueError(f"Unsupported CelebA resize_mode '{resize_mode}'. Use 'area', 'bilinear', or 'bicubic'.")


# Decoding, center-cropping and downsampling a full CelebA split is done once and cached
# to disk; every run with the same preprocessing settings reuses the cache. The random
# subset selection and the source sampling happen after loading the cache.


def _cache_path(config: dict[str, Any], root: str, split: str):
    width = int(config.get("width", 32))
    resize_mode = str(config.get("resize_mode", "area")).lower()
    antialias = bool(config.get("antialias", True))
    crop_size = config.get("center_crop")
    crop_tag = str(int(crop_size)) if crop_size is not None else "full"
    cache_dir = Path(config.get("cache_dir") or Path(root) / "celeba" / "preprocessed")
    return cache_dir / f"celeba_{split}_crop{crop_tag}_w{width}_{resize_mode}_aa{int(antialias)}.pt"


def _build_cache(config: dict[str, Any], root: str, split: str):
    dataset = datasets.CelebA(
        root=root,
        split=split,
        target_type="attr",
        download=bool(config.get("download", True)),
        transform=transforms.ToTensor(),
    )
    width = int(config.get("width", 32))
    resize_mode = str(config.get("resize_mode", "area")).lower()
    antialias = bool(config.get("antialias", True))
    requested_crop_size = config.get("center_crop")

    print(f"Preprocessing {len(dataset)} CelebA '{split}' images into the dataset cache...", flush=True)
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=int(config.get("cache_workers", 8)),
        generator=torch.Generator().manual_seed(0),
    )
    images = torch.empty((len(dataset), 3, width, width), dtype=torch.float32)
    offset = 0
    for batch_images, _ in loader:
        batch_images = batch_images.to(dtype=torch.float32)
        crop_size = int(requested_crop_size or min(batch_images.shape[-2], batch_images.shape[-1]))
        cropped = _center_crop_tensor(batch_images, crop_size)
        images[offset : offset + batch_images.shape[0]] = _downsample_batch(
            cropped,
            width=width,
            resize_mode=resize_mode,
            antialias=antialias,
        )
        offset += batch_images.shape[0]
    return {"images": images}


def _load_or_build_cache(config: dict[str, Any], root: str, split: str):
    cache_path = _cache_path(config, root, split)
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    payload = _build_cache(config, root, split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, cache_path)
    print(f"Saved CelebA dataset cache: {cache_path}", flush=True)
    return payload


def load_celeba_vectors(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    root = config.get("root", "./data")
    split = str(config.get("split", "train")).lower()
    n_samples = int(config["n_samples"])
    width = int(config.get("width", 32))

    images = _load_or_build_cache(config, root, split)["images"]
    if n_samples > images.shape[0]:
        raise ValueError(f"Requested {n_samples} CelebA samples, but only {images.shape[0]} are available")

    selected_indices = torch.randperm(images.shape[0])[:n_samples]
    target_data = images[selected_indices].to(device=device, dtype=dtype).view(n_samples, 3 * width * width)
    source_data = build_image_source_data(
        config.get("source"),
        n_samples=n_samples,
        width=width,
        input_channels=3,
        device=device,
        dtype=dtype,
    )
    metadata = {"width": width, "input_channels": 3, "split": split}
    return source_data, target_data, metadata
