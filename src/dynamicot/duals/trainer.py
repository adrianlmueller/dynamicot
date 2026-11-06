from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..data import build_2d_dataset
from .models import TimeEmbeddingMLP
from .objectives import compute_dual_loss
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


@dataclass
class TrainingArtifacts:
    model: TimeEmbeddingMLP
    dual_loss_history: list[float]
    parameter_norm_history: list[float]
    checkpoint_paths: list[str]
    final_checkpoint_path: str
    metrics_path: str
    results_dir: str


def _parameter_l2_norm(model: torch.nn.Module):
    total_norm = 0.0
    for parameter in model.parameters():
        total_norm += torch.norm(parameter, 2).item() ** 2
    return total_norm ** 0.5


def _build_optimizer(model: torch.nn.Module, config: dict[str, Any]):
    dual_cfg = config["training"]["dual"]
    return torch.optim.Adam(
        model.parameters(),
        lr=float(dual_cfg["lr"]),
        betas=(float(dual_cfg["beta1"]), float(dual_cfg["beta2"])),
        eps=float(dual_cfg["eps"]),
    )


def build_model(config: dict[str, Any]):
    model_cfg = config["model"]
    return TimeEmbeddingMLP(
        spatial_dim=2,
        hidden_dims=list(model_cfg["hidden_dims"]),
        output_dim=1,
        time_embedding_dim=int(model_cfg.get("time_embedding_dim", 32)),
    )


def prepare_experiment(config: dict[str, Any]):
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(config.get("device", "cpu"))
    dtype = resolve_dtype(config.get("dtype", "float32"))

    data_cfg = config["data"]
    source_data, target_data = build_2d_dataset(
        data_cfg["name"],
        n_samples=int(data_cfg["n_samples"]),
        seed=seed,
        device=device,
        dtype=dtype,
    )
    model = build_model(config).to(device)
    return model, source_data, target_data


def train_dual_model(config: dict[str, Any], *, results_dir: str | None = None):
    model, source_data, target_data = prepare_experiment(config)
    optimizer = _build_optimizer(model, config)
    results_dir = create_results_dir(config) if results_dir is None else results_dir
    save_experiment_summary(results_dir, config)

    training_cfg = config["training"]
    horizon = int(training_cfg["horizon"])
    iterations = int(training_cfg["iterations"])
    snapshot_interval = int(training_cfg["snapshot_interval"])
    batch_size = int(training_cfg["batching"]["batch_size"])
    initialization_type = str(training_cfg.get("initialization_type", "linear")).lower()

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

    progress = ProgressBar(iterations, desc="dual_2d training", unit="iter")
    for step in progress:
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
            step_size=float(training_cfg["primal"]["lr"]),
            noise_scale=float(training_cfg["primal"]["noise_scale"]),
            num_steps=int(training_cfg["primal"]["num_steps"]),
        )

        optimizer.zero_grad(set_to_none=True)
        total_dual_loss = compute_dual_loss(
            model,
            x_batch,
            y_batch,
            [state.detach() for state in x_h_minus_batch],
            [state.detach() for state in x_h_plus_batch],
            h_steps=horizon,
            h_feat=h_feat,
        )
        dual_loss_history.append(float(total_dual_loss.item()))
        total_dual_loss.backward()
        optimizer.step()

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

    progress.close()

    metrics_path = save_json(
        results_dir,
        "metrics.json",
        {
            "dual_loss_history": dual_loss_history,
            "parameter_norm_history": parameter_norm_history,
            "snapshot_interval": snapshot_interval,
            "checkpoint_paths": checkpoint_paths,
        },
    )
    final_checkpoint_path = save_checkpoint(
        results_dir,
        filename="checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        step=iterations,
    )

    return TrainingArtifacts(
        model=model,
        dual_loss_history=dual_loss_history,
        parameter_norm_history=parameter_norm_history,
        checkpoint_paths=checkpoint_paths,
        final_checkpoint_path=final_checkpoint_path,
        metrics_path=metrics_path,
        results_dir=results_dir,
    )
