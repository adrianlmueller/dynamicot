from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data import load_celeba_vectors, load_emnist_mnist_vectors, load_mnist_upsample_vectors
from .models import build_time_conditioned_image_model
from .objectives import backward_dual_loss
from ..primals.batching import sample_training_batch
from ..primals.particles import (
    apply_initialization_coupling,
    build_time_features,
    initialize_particle_pairs,
    run_primal_updates,
)
from ..utils.io import create_results_dir, save_checkpoint, save_experiment_summary, save_json
from ..utils.progress import ProgressBar
from ..utils.runtime import resolve_device, resolve_dtype, set_seed


DEFAULT_IMAGE_TRAINING_TIME_CHUNK_SIZE = 10


@dataclass
class ImageTrainingArtifacts:
    model: torch.nn.Module
    dual_loss_history: list[float]
    parameter_norm_history: list[float]
    checkpoint_paths: list[str]
    final_checkpoint_path: str
    ema_checkpoint_path: str | None
    metrics_path: str
    results_dir: str


def _initialize_ema_state(model: torch.nn.Module):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _update_ema_state(ema_state: dict[str, torch.Tensor], model: torch.nn.Module, decay: float):
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            tensor_cpu = tensor.detach().cpu()
            if torch.is_floating_point(ema_state[name]):
                ema_state[name].mul_(decay).add_(tensor_cpu, alpha=1.0 - decay)
            else:
                ema_state[name].copy_(tensor_cpu)


def _parameter_l2_norm(model: torch.nn.Module):
    total_norm = 0.0
    for parameter in model.parameters():
        total_norm += torch.norm(parameter, 2).item() ** 2
    return total_norm ** 0.5


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    dual_cfg = config["training"]["dual"]
    optimizer_type = str(dual_cfg.get("optimizer", "adam")).lower()
    optimizer_kwargs = {
        "lr": float(dual_cfg["lr"]),
        "betas": (float(dual_cfg["beta1"]), float(dual_cfg["beta2"])),
        "eps": float(dual_cfg["eps"]),
    }
    if optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            weight_decay=float(dual_cfg.get("weight_decay", 1.0e-4)),
            **optimizer_kwargs,
        )
    raise ValueError("dual.optimizer must be one of: adam, adamw")


def _build_lr_schedule(config: dict[str, Any]):
    """Piecewise-constant learning rate: [(0, lr), (step_1, lr_1), ...]."""
    dual_cfg = config["training"]["dual"]
    schedule = [(0, float(dual_cfg["lr"]))]
    schedule_cfg = dual_cfg.get("lr_schedule")
    if schedule_cfg is None:
        return schedule
    if str(schedule_cfg.get("type", "")).lower() != "piecewise_constant":
        raise ValueError("dual.lr_schedule.type must be piecewise_constant")
    for milestone in schedule_cfg.get("milestones", []):
        step = int(milestone["step"])
        if step <= schedule[-1][0]:
            raise ValueError("dual.lr_schedule milestone steps must be positive and strictly increasing")
        schedule.append((step, float(milestone["lr"])))
    return schedule


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, schedule: list[tuple[int, float]], step: int):
    learning_rate = schedule[0][1]
    for milestone_step, milestone_learning_rate in schedule[1:]:
        if step < milestone_step:
            break
        learning_rate = milestone_learning_rate
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _prepare_image_experiment(config: dict[str, Any], dataset_loader):
    seed = config.get("seed")
    if seed is not None:
        set_seed(int(seed))

    device = resolve_device(config.get("device", "cpu"))
    dtype = resolve_dtype(config.get("dtype", "float32"))

    source_data, target_data, metadata = dataset_loader(config["data"], device=device, dtype=dtype)

    model_cfg = config["model"]
    input_channels = int(model_cfg.get("input_channels", metadata.get("input_channels", 1)))
    model = build_time_conditioned_image_model(
        model_cfg,
        image_size=int(metadata["width"]),
        input_channels=input_channels,
    ).to(device)

    return model, source_data, target_data, metadata


def prepare_emnist_mnist_experiment(config: dict[str, Any]):
    return _prepare_image_experiment(config, load_emnist_mnist_vectors)


def prepare_mnist_upsample_experiment(config: dict[str, Any]):
    return _prepare_image_experiment(config, load_mnist_upsample_vectors)


def prepare_celeba_experiment(config: dict[str, Any]):
    return _prepare_image_experiment(config, load_celeba_vectors)


def _train_image_dual_model(config: dict[str, Any], prepare_fn):
    model, source_data, target_data, _ = prepare_fn(config)
    optimizer = _build_optimizer(model, config)
    lr_schedule = _build_lr_schedule(config)
    results_dir = create_results_dir(config)
    save_experiment_summary(results_dir, config)

    training_cfg = config["training"]
    primal_cfg = training_cfg["primal"]
    horizon = int(training_cfg["horizon"])
    iterations = int(training_cfg["iterations"])
    snapshot_interval = int(training_cfg["snapshot_interval"])
    batch_size = int(training_cfg["batching"]["batch_size"])
    time_chunk_size = int(training_cfg.get("time_chunk_size", DEFAULT_IMAGE_TRAINING_TIME_CHUNK_SIZE))
    initialization_type = str(training_cfg.get("initialization_type", "linear")).lower()
    clamp = primal_cfg.get("clamp")
    clamp = None if clamp is None else (float(clamp[0]), float(clamp[1]))
    ema_decay = training_cfg.get("ema_decay")
    ema_decay = None if ema_decay is None else float(ema_decay)
    if ema_decay is not None and not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be between 0 and 1")
    if source_data.shape[0] % batch_size != 0:
        raise ValueError("batch_size must divide n_samples")

    h_minus, h_plus, h_feat = build_time_features(
        horizon=horizon,
        batch_size=batch_size,
        device=source_data.device,
        dtype=source_data.dtype,
    )

    dual_loss_history: list[float] = []
    parameter_norm_history: list[float] = []
    checkpoint_paths: list[str] = []
    batch_state: dict[str, torch.Tensor] = {}
    ema_state = _initialize_ema_state(model) if ema_decay is not None else None

    progress = ProgressBar(iterations, desc="image training", unit="iter")
    for step in progress:
        _set_optimizer_lr(optimizer, lr_schedule, step)
        x_batch, y_batch, batch_state = sample_training_batch(
            source_data,
            target_data,
            step=step,
            batch_size=batch_size,
            batching_type=training_cfg["batching"]["type"],
            state=batch_state,
        )
        x_batch, y_batch = apply_initialization_coupling(
            x_batch,
            y_batch,
            initialization_type=initialization_type,
        )

        x_h_minus_batch, x_h_plus_batch = initialize_particle_pairs(x_batch, y_batch, horizon=horizon)
        x_h_minus_batch, x_h_plus_batch = run_primal_updates(
            model,
            x_h_minus_batch,
            x_h_plus_batch,
            horizon=horizon,
            h_minus=h_minus,
            h_plus=h_plus,
            step_size=float(primal_cfg["lr"]),
            noise_scale=float(primal_cfg["noise_scale"]),
            num_steps=int(primal_cfg["num_steps"]),
            time_chunk_size=time_chunk_size,
            clamp=clamp,
        )

        optimizer.zero_grad(set_to_none=True)
        dual_loss = backward_dual_loss(
            model,
            x_batch,
            y_batch,
            [state.detach() for state in x_h_minus_batch],
            [state.detach() for state in x_h_plus_batch],
            h_steps=horizon,
            h_feat=h_feat,
            time_chunk_size=time_chunk_size,
        )
        optimizer.step()
        if ema_state is not None:
            _update_ema_state(ema_state, model, ema_decay)

        dual_loss_history.append(dual_loss)
        parameter_norm_history.append(_parameter_l2_norm(model))
        progress.set_postfix(
            loss=f"{dual_loss_history[-1]:.4f}",
            norm=f"{parameter_norm_history[-1]:.2f}",
        )

        if (step + 1) % snapshot_interval == 0:
            checkpoint_paths.append(
                save_checkpoint(
                    results_dir,
                    filename=f"checkpoint_step_{step + 1:07d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                )
            )
            if ema_state is not None:
                save_checkpoint(
                    results_dir,
                    filename=f"checkpoint_step_{step + 1:07d}_ema.pt",
                    model=None,
                    model_state_dict=ema_state,
                    optimizer=None,
                    step=step + 1,
                )
            progress.write(f"Saved checkpoint at step {step + 1} -> {checkpoint_paths[-1]}")

    progress.close()

    metrics_path = save_json(
        results_dir,
        "metrics.json",
        {
            "dual_loss_history": dual_loss_history,
            "parameter_norm_history": parameter_norm_history,
            "snapshot_interval": snapshot_interval,
            "checkpoint_paths": checkpoint_paths,
            "ema_decay": ema_decay,
        },
    )
    final_checkpoint_path = save_checkpoint(
        results_dir,
        filename="checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        step=iterations,
    )
    ema_checkpoint_path = None
    if ema_state is not None:
        ema_checkpoint_path = save_checkpoint(
            results_dir,
            filename="checkpoint_final_ema.pt",
            model=None,
            model_state_dict=ema_state,
            optimizer=None,
            step=iterations,
        )

    return ImageTrainingArtifacts(
        model=model,
        dual_loss_history=dual_loss_history,
        parameter_norm_history=parameter_norm_history,
        checkpoint_paths=checkpoint_paths,
        final_checkpoint_path=final_checkpoint_path,
        ema_checkpoint_path=ema_checkpoint_path,
        metrics_path=metrics_path,
        results_dir=results_dir,
    )


def train_emnist_mnist_dual_model(config: dict[str, Any]):
    data_cfg = config.get("data", {})
    source_cfg = data_cfg.get("source") or {}
    if not bool(data_cfg.get("train", True)) or not bool(source_cfg.get("train", True)):
        raise ValueError("dual_emnist_mnist training must use the MNIST and EMNIST train splits")
    return _train_image_dual_model(config, prepare_emnist_mnist_experiment)


def train_mnist_upsample_dual_model(config: dict[str, Any]):
    if not bool(config.get("data", {}).get("train", True)):
        raise ValueError("dual_mnist_upsample training must use the MNIST train split: set data.train to true")
    return _train_image_dual_model(config, prepare_mnist_upsample_experiment)


def train_celeba_dual_model(config: dict[str, Any]):
    if str(config.get("data", {}).get("split", "train")).lower() != "train":
        raise ValueError("dual_celeba training must use the CelebA train split: set data.split to train")
    return _train_image_dual_model(config, prepare_celeba_experiment)
