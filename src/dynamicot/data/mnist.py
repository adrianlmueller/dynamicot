from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from .image_source import build_image_source_data


EMNIST_LETTER_LABELS = {chr(ord("A") + idx): idx + 1 for idx in range(26)}


def _normalize_emnist_letter(letter: Any):
    if isinstance(letter, int):
        label = int(letter)
    else:
        text = str(letter).strip()
        if len(text) == 1 and text.isalpha():
            label = ord(text.upper()) - ord("A") + 1
        elif text.isdigit():
            label = int(text)
        else:
            raise ValueError(f"Invalid EMNIST letter '{letter}'. Use letters A-Z or labels 1-26.")

    if not 1 <= label <= 26:
        raise ValueError(f"Invalid EMNIST letter label {label}. Use labels 1-26.")
    return label, chr(ord("A") + label - 1)


def _resolve_emnist_letters(source_config: dict[str, Any]):
    requested = source_config.get("letters")
    if requested is None:
        requested = list(EMNIST_LETTER_LABELS)
    elif isinstance(requested, str):
        text = requested.strip()
        if "," in text:
            requested = [letter.strip() for letter in text.split(",") if letter.strip()]
        else:
            requested = list(text) if len(text) > 1 and text.isalpha() else [text]
    elif isinstance(requested, int):
        requested = [requested]

    labels_to_letters = {}
    for letter in requested:
        label, normalized = _normalize_emnist_letter(letter)
        labels_to_letters[label] = normalized

    labels = sorted(labels_to_letters)
    if not labels:
        raise ValueError("data.source.letters must contain at least one EMNIST letter")
    return labels, [labels_to_letters[label] for label in labels]


def _orient_emnist_letters(images: torch.Tensor):
    return torch.rot90(images, k=-1, dims=(-2, -1)).flip(-1).contiguous()


def _select_mnist_targets(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    root = config.get("root", "./data")
    train = bool(config.get("train", True))
    download = bool(config.get("download", True))
    digits = set(int(digit) for digit in config.get("digits", [0, 1]))
    n_samples = int(config["n_samples"])

    dataset = datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )

    mask = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for digit in digits:
        mask = mask | (dataset.targets == digit)
    indices = mask.nonzero(as_tuple=True)[0]

    mnist_data = dataset.data[indices].to(dtype=torch.float32) / 255.0
    if n_samples > mnist_data.shape[0]:
        raise ValueError(f"Requested {n_samples} MNIST samples, but only {mnist_data.shape[0]} are available")

    perm = torch.randperm(mnist_data.shape[0])
    target_original = mnist_data[perm][:n_samples].to(device=device, dtype=dtype)
    return target_original, sorted(digits)


def load_emnist_letter_vectors(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    source_config = config.get("source") or {}
    root = source_config.get("root", config.get("root", "./data"))
    train = bool(source_config.get("train", config.get("train", True)))
    download = bool(source_config.get("download", config.get("download", True)))
    n_samples = int(config["n_samples"])
    width = int(config.get("width", 28))
    labels, letters = _resolve_emnist_letters(source_config)
    fix_orientation = bool(source_config.get("fix_orientation", True))

    dataset = datasets.EMNIST(
        root=root,
        split="letters",
        train=train,
        download=download,
        transform=transforms.ToTensor(),
    )

    mask = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for label in labels:
        mask = mask | (dataset.targets == label)
    indices = mask.nonzero(as_tuple=True)[0]

    emnist_data = dataset.data[indices].to(dtype=torch.float32) / 255.0
    if fix_orientation:
        emnist_data = _orient_emnist_letters(emnist_data)
    if emnist_data.shape[0] == 0:
        split_name = "train" if train else "test"
        raise ValueError(
            f"No EMNIST {split_name} samples are available for letters {letters}"
        )

    if n_samples <= emnist_data.shape[0]:
        sample_indices = torch.randperm(emnist_data.shape[0])[:n_samples]
        sampled_with_replacement = False
    else:
        sample_indices = torch.randint(emnist_data.shape[0], (n_samples,))
        sampled_with_replacement = True
    source_original = emnist_data[sample_indices].to(device=device, dtype=dtype)
    source_images = source_original.reshape(n_samples, 1, 28, 28)
    source_downsampled = F.adaptive_avg_pool2d(source_images, (width, width))
    source_data = source_downsampled.reshape(n_samples, width * width)

    metadata = {
        "source_original": source_original.reshape(n_samples, 28 * 28),
        "source_letters": letters,
        "source_letter_labels": labels,
        "emnist_split": "train" if train else "test",
        "source_fix_orientation": fix_orientation,
        "source_sampled_with_replacement": sampled_with_replacement,
    }
    return source_data, metadata


def load_emnist_mnist_vectors(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    n_samples = int(config["n_samples"])
    width = int(config.get("width", 10))

    target_original, digits = _select_mnist_targets(config, device=device, dtype=dtype)
    target_images = target_original.view(n_samples, 1, 28, 28)
    target_downsampled = F.adaptive_avg_pool2d(target_images, (width, width))
    target_data = target_downsampled.view(n_samples, width * width)
    source_data, source_metadata = load_emnist_letter_vectors(config, device=device, dtype=dtype)
    source_metadata.pop("source_original", None)

    metadata = {
        "width": width,
        "original_width": 28,
        "target_original": target_original.view(n_samples, 28 * 28),
        "digits": digits,
        **source_metadata,
    }
    return source_data, target_data, metadata


def load_mnist_upsample_vectors(config: dict[str, Any], *, device: torch.device, dtype: torch.dtype):
    n_samples = int(config["n_samples"])
    width = int(config.get("width", 28))
    d_small = int(config["d_small"])
    original_width = 28

    if width != original_width:
        raise ValueError("MNIST upsample experiment expects data.width to be 28")
    if d_small <= 0:
        raise ValueError("data.d_small must be positive")
    if d_small > original_width:
        raise ValueError("data.d_small must be at most 28 for MNIST")

    target_original, digits = _select_mnist_targets(config, device=device, dtype=dtype)
    target_images = target_original.view(n_samples, 1, original_width, original_width)
    source_small = F.interpolate(
        target_images,
        size=(d_small, d_small),
        mode="bilinear",
        align_corners=False,
    )
    source_images = F.interpolate(
        source_small,
        size=(original_width, original_width),
        mode="bilinear",
        align_corners=False,
    )

    source_data = source_images.view(n_samples, width * width)
    target_data = target_images.view(n_samples, width * width)
    metadata = {
        "width": width,
        "original_width": original_width,
        "target_original": target_data,
        "digits": digits,
        "d_small": d_small,
    }
    return source_data, target_data, metadata
