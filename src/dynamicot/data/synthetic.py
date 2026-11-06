from __future__ import annotations

import math

import numpy as np
import torch


DATASETS_2D = ("moons", "scurve", "8gaussians", "moons_8gaussians")


def _eight_gaussians(n_samples: int, *, std_dev, device: torch.device, dtype: torch.dtype):
    num_clusters = 8
    if n_samples % num_clusters != 0:
        raise ValueError("n_samples must be divisible by 8 for the 8gaussians data set")

    clusters = []
    for i in range(num_clusters):
        angle = 2 * math.pi * i / num_clusters
        mean = torch.tensor([np.cos(angle), np.sin(angle)], device=device, dtype=dtype) * 5
        noise = torch.randn(n_samples // num_clusters, 2, device=device, dtype=dtype)
        clusters.append(noise * std_dev + mean)
    target = torch.cat(clusters, dim=0)
    return target[torch.randperm(target.shape[0], device=device)]


def build_2d_dataset(
    name: str,
    *,
    n_samples: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Return (source, target) samples for the 2D benchmarks of Table 1.

    The global torch/numpy RNGs are re-seeded with `seed`, so a data set is fully
    determined by (name, n_samples, seed).
    """
    import sklearn.datasets

    torch.manual_seed(seed)
    np.random.seed(seed)

    if name == "moons":
        source = torch.randn(n_samples, 2, device=device, dtype=dtype)
        moons, _ = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
        target = torch.tensor(moons, device=device, dtype=dtype) * 2 - 1
    elif name == "scurve":
        source = torch.randn(n_samples, 2, device=device, dtype=dtype)
        scurve, _ = sklearn.datasets.make_s_curve(n_samples=n_samples, noise=0.05, random_state=seed)
        scurve = np.vstack([scurve[:, 0], scurve[:, 2]]).T * 1.5
        target = torch.tensor(scurve, device=device, dtype=dtype)
    elif name == "8gaussians":
        source = torch.randn(n_samples, 2, device=device, dtype=dtype)
        std_dev = torch.sqrt(torch.tensor(0.1, device=device, dtype=dtype))
        target = _eight_gaussians(n_samples, std_dev=std_dev, device=device, dtype=dtype)
    elif name == "moons_8gaussians":
        moons, _ = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
        source = torch.tensor(moons * 3 - 1, device=device, dtype=dtype) * 2
        target = _eight_gaussians(n_samples, std_dev=np.sqrt(0.1), device=device, dtype=dtype) * 2
    else:
        raise ValueError(f"Unknown 2D data set '{name}'. Use one of: {', '.join(DATASETS_2D)}")

    return source, target
