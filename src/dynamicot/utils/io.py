from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import torch


def create_results_dir(config: dict[str, Any]):
    output_cfg = config.get("output", {})
    base_dir = Path(output_cfg.get("base_dir", "dual_results"))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def save_experiment_summary(results_dir: str, config: dict[str, Any]):
    output_path = Path(results_dir) / "experiment_config.json"
    output_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(output_path)


def load_experiment_summary(results_dir: str):
    input_path = Path(results_dir) / "experiment_config.json"
    return json.loads(input_path.read_text(encoding="utf-8"))


def save_json(results_dir: str, filename: str, payload: dict[str, Any]):
    output_path = Path(results_dir) / filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(output_path)


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch_artifact(results_dir: str, filename: str, payload: Any, subdir: str | None = None):
    output_dir = Path(results_dir) if subdir is None else Path(results_dir) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    torch.save(payload, output_path)
    return str(output_path)


def save_checkpoint(
    results_dir: str,
    *,
    filename: str,
    model: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    model_state_dict: dict[str, torch.Tensor] | None = None,
):
    if model_state_dict is None:
        if model is None:
            raise ValueError("Either model or model_state_dict must be provided")
        model_state_dict = model.state_dict()

    checkpoints_dir = Path(results_dir) / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_path = checkpoints_dir / filename
    payload = {
        "step": step,
        "model_state_dict": model_state_dict,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, output_path)
    return str(output_path)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu"):
    return torch.load(path, map_location=map_location, weights_only=True)


def load_torch_artifact(path: str, map_location: str | torch.device = "cpu"):
    return torch.load(path, map_location=map_location, weights_only=True)
