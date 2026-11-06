from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from ..duals.image_trainer import prepare_mnist_upsample_experiment
from ..duals.inference import generate_from_source, generate_from_source_few_steps
from ..utils import load_checkpoint, load_experiment_summary, save_torch_artifact


def _build_test_config(config: dict, num_samples: int):
    eval_config = copy.deepcopy(config)
    eval_config["data"]["train"] = False
    eval_config["data"]["download"] = True
    eval_config["data"]["n_samples"] = num_samples
    return eval_config


def evaluate_mnist_upsample_checkpoint(
    results_dir: str,
    *,
    checkpoint_filename: str = "checkpoint_final.pt",
    checkpoint_path: str | None = None,
    evaluation_filename: str = "evaluation.pt",
    num_samples: int = 30,
    output_dir: str | None = None,
):
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    config = load_experiment_summary(results_dir)
    eval_config = _build_test_config(config, num_samples)
    model, fresh_source, target_data, metadata = prepare_mnist_upsample_experiment(eval_config)
    resolved_checkpoint_path = (
        Path(checkpoint_path).expanduser()
        if checkpoint_path is not None
        else Path(results_dir) / "checkpoints" / checkpoint_filename
    )
    if checkpoint_path is not None:
        checkpoint_filename = resolved_checkpoint_path.name
    checkpoint = load_checkpoint(
        str(resolved_checkpoint_path),
        map_location=next(model.parameters()).device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    width = int(metadata["width"])
    original_width = int(metadata.get("original_width", width))
    horizon = int(config["training"]["horizon"])
    inference_noise_scale = float(config["training"]["primal"]["noise_scale"]) / np.sqrt(
        2.0 * float(config["training"]["primal"]["lr"])
    )
    device = next(model.parameters()).device

    fresh_source = fresh_source[:num_samples].to(device)
    target_data = target_data[:num_samples].to(device)
    target_original = metadata["target_original"][:num_samples].to(device)
    generated = generate_from_source(
        model,
        fresh_source,
        horizon,
        noise_scale=inference_noise_scale,
        clamp=(0.0, 1.0),
    )[-1]

    step_candidates = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    step_choices = [
        num_steps
        for num_steps in step_candidates
        if num_steps <= horizon + 1 and (horizon + 1) % num_steps == 0
    ]
    if (horizon + 1) not in step_choices:
        step_choices.append(horizon + 1)

    few_step_outputs = {}
    for num_steps in step_choices:
        outputs = []
        for sample_idx in range(num_samples):
            outputs.append(
                generate_from_source_few_steps(
                    model,
                    fresh_source[sample_idx : sample_idx + 1],
                    horizon,
                    num_steps=num_steps,
                    noise_scale=inference_noise_scale,
                    clamp=(0.0, 1.0),
                )[-1].detach().cpu()
            )
        few_step_outputs[str(num_steps)] = torch.cat(outputs, dim=0)

    artifact_results_dir = results_dir if output_dir is None else output_dir
    return save_torch_artifact(
        artifact_results_dir,
        evaluation_filename,
        {
            "fresh_source": fresh_source.detach().cpu(),
            "target": target_data.detach().cpu(),
            "target_original": target_original.detach().cpu(),
            "generated": generated.detach().cpu(),
            "few_step_outputs": few_step_outputs,
            "step_choices": step_choices,
            "width": width,
            "original_width": original_width,
            "d_small": int(metadata["d_small"]),
            "mnist_split": "test",
            "eval_data_train": False,
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_path": str(resolved_checkpoint_path),
            "training_results_dir": str(results_dir),
        },
        subdir="evaluation",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST upsample dual dynamic OT experiment.")
    parser.add_argument("--results-dir", required=True, help="Results directory created by training.")
    parser.add_argument("--checkpoint-filename", default="checkpoint_final.pt", help="Checkpoint filename under <results-dir>/checkpoints.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional explicit checkpoint path. If omitted, uses <results-dir>/checkpoints/<checkpoint-filename>.",
    )
    parser.add_argument("--evaluation-filename", default="evaluation.pt", help="Evaluation filename under <results-dir>/evaluation.")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of fresh test samples to generate and save.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for the evaluation artifact. Defaults to results-dir.",
    )
    args = parser.parse_args()

    output_path = evaluate_mnist_upsample_checkpoint(
        args.results_dir,
        checkpoint_filename=args.checkpoint_filename,
        checkpoint_path=args.checkpoint_path,
        evaluation_filename=args.evaluation_filename,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
    )
    print(f"Saved evaluation artifact: {output_path}")


if __name__ == "__main__":
    main()
