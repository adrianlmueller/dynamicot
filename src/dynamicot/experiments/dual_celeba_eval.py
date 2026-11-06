from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from ..duals.image_trainer import prepare_celeba_experiment
from ..duals.inference import generate_from_source, generate_from_source_few_steps
from ..utils import load_checkpoint, load_experiment_summary, save_torch_artifact


DEFAULT_FEW_STEP_CHOICES = (1, 3, 5, 10, 15, 30)


def build_celeba_evaluation_config(config: dict, *, num_samples: int, split: str):
    eval_config = copy.deepcopy(config)
    eval_config["data"]["split"] = split
    eval_config["data"]["n_samples"] = num_samples
    return eval_config


def load_celeba_evaluation_model(
    results_dir: str,
    *,
    num_samples: int,
    split: str,
    checkpoint_path: str | Path,
):
    """Build the model and draw `num_samples` fresh uniform sources and holdout CelebA
    images from `split` (valid or test), then load the checkpoint."""
    if split not in {"valid", "test"}:
        raise ValueError("split must be either valid or test for CelebA evaluation")
    config = load_experiment_summary(results_dir)
    eval_config = build_celeba_evaluation_config(config, num_samples=num_samples, split=split)
    model, fresh_source, target_data, metadata = prepare_celeba_experiment(eval_config)
    checkpoint = load_checkpoint(str(checkpoint_path), map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return config, model, fresh_source, target_data, metadata


def evaluate_celeba_checkpoint(
    results_dir: str,
    *,
    checkpoint_filename: str = "checkpoint_final.pt",
    evaluation_filename: str = "evaluation.pt",
    num_samples: int = 30,
    split: str = "valid",
    step_choices: tuple[int, ...] = DEFAULT_FEW_STEP_CHOICES,
):
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    checkpoint_path = Path(results_dir) / "checkpoints" / checkpoint_filename
    config, model, fresh_source, target_data, metadata = load_celeba_evaluation_model(
        results_dir,
        num_samples=num_samples,
        split=split,
        checkpoint_path=checkpoint_path,
    )
    horizon = int(config["training"]["horizon"])
    inference_noise_scale = float(config["training"]["primal"]["noise_scale"]) / np.sqrt(
        2.0 * float(config["training"]["primal"]["lr"])
    )

    generated = generate_from_source(
        model,
        fresh_source,
        horizon,
        noise_scale=inference_noise_scale,
        clamp=(0.0, 1.0),
    )[-1]

    step_choices = [num_steps for num_steps in step_choices if 1 <= num_steps <= horizon + 1]
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

    return save_torch_artifact(
        results_dir,
        evaluation_filename,
        {
            "fresh_source": fresh_source.detach().cpu(),
            "target": target_data.detach().cpu(),
            "generated": generated.detach().cpu(),
            "few_step_outputs": few_step_outputs,
            "step_choices": step_choices,
            "width": int(metadata["width"]),
            "input_channels": int(metadata["input_channels"]),
            "celeba_split": split,
            "checkpoint_filename": checkpoint_filename,
        },
        subdir="evaluation",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained CelebA dual dynamic OT experiment.")
    parser.add_argument("--results-dir", required=True, help="Results directory created by training.")
    parser.add_argument("--checkpoint-filename", default="checkpoint_final.pt", help="Checkpoint filename under <results-dir>/checkpoints.")
    parser.add_argument("--evaluation-filename", default="evaluation.pt", help="Evaluation filename under <results-dir>/evaluation.")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of fresh samples to generate and save.")
    parser.add_argument("--split", choices=("valid", "test"), default="valid", help="CelebA holdout split.")
    args = parser.parse_args()

    output_path = evaluate_celeba_checkpoint(
        args.results_dir,
        checkpoint_filename=args.checkpoint_filename,
        evaluation_filename=args.evaluation_filename,
        num_samples=args.num_samples,
        split=args.split,
    )
    print(f"Saved evaluation artifact: {output_path}")


if __name__ == "__main__":
    main()
