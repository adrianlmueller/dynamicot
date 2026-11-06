from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import load_emnist_letter_vectors
from ..data.mnist import _select_mnist_targets
from ..duals.inference import (
    generate_from_source,
    generate_from_source_few_steps,
    generate_from_target,
    generate_from_target_few_steps,
)
from ..duals.models import build_time_conditioned_image_model
from ..utils import load_checkpoint, load_experiment_summary, resolve_device, resolve_dtype, save_torch_artifact, set_seed


def _build_test_source_config(config: dict, num_samples: int):
    eval_config = copy.deepcopy(config)
    eval_config["data"]["n_samples"] = num_samples
    eval_config["data"]["download"] = True
    source_config = eval_config["data"].setdefault("source", {})
    source_config["train"] = False
    source_config["download"] = True
    return eval_config


def _build_test_target_config(config: dict, num_samples: int):
    target_config = copy.deepcopy(config["data"])
    target_config["n_samples"] = num_samples
    target_config["train"] = False
    target_config["download"] = True
    return target_config


def _load_mnist_test_targets(config: dict, *, num_samples: int, width: int, device: torch.device, dtype: torch.dtype):
    target_config = _build_test_target_config(config, num_samples)
    target_original, digits = _select_mnist_targets(target_config, device=device, dtype=dtype)
    target_images = target_original.view(num_samples, 1, 28, 28)
    target_data = F.adaptive_avg_pool2d(target_images, (width, width)).view(num_samples, width * width)
    return target_data, target_original.view(num_samples, 28 * 28), digits


def evaluate_emnist_mnist_checkpoint(
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
    seed = config.get("seed")
    if seed is not None:
        set_seed(int(seed))

    eval_config = _build_test_source_config(config, num_samples)
    device = resolve_device(config.get("device", "cpu"))
    dtype = resolve_dtype(config.get("dtype", "float32"))
    width = int(eval_config["data"].get("width", 28))
    model_cfg = config["model"]
    input_channels = int(model_cfg.get("input_channels", 1))

    fresh_source, source_metadata = load_emnist_letter_vectors(eval_config["data"], device=device, dtype=dtype)
    model = build_time_conditioned_image_model(
        model_cfg,
        image_size=width,
        input_channels=input_channels,
    ).to(device)
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

    original_width = 28
    horizon = int(config["training"]["horizon"])
    inference_noise_scale = float(config["training"]["primal"]["noise_scale"]) / np.sqrt(
        2.0 * float(config["training"]["primal"]["lr"])
    )

    fresh_source = fresh_source[:num_samples].to(device)
    source_original = source_metadata["source_original"][:num_samples].to(device)
    reverse_target, reverse_target_original, reverse_digits = _load_mnist_test_targets(
        config,
        num_samples=num_samples,
        width=width,
        device=device,
        dtype=dtype,
    )
    generated = generate_from_source(
        model,
        fresh_source,
        horizon,
        noise_scale=inference_noise_scale,
        clamp=(0.0, 1.0),
    )[-1]
    reverse_generated = generate_from_target(
        model,
        reverse_target,
        horizon,
        noise_scale=inference_noise_scale,
        clamp=(0.0, 1.0),
    )[-1]

    source_upsampled = torch.nn.functional.interpolate(
        fresh_source.view(-1, 1, width, width),
        size=(original_width, original_width),
        mode="bilinear",
        align_corners=False,
    ).view(-1, original_width * original_width)
    generated_upsampled = torch.nn.functional.interpolate(
        generated.view(-1, 1, width, width),
        size=(original_width, original_width),
        mode="bilinear",
        align_corners=False,
    ).view(-1, original_width * original_width)

    step_candidates = [1, 2, 4, 5, 10, 20, 25, 50, 100]
    step_choices = [
        num_steps
        for num_steps in step_candidates
        if num_steps <= horizon + 1 and (horizon + 1) % num_steps == 0
    ]
    if (horizon + 1) not in step_choices:
        step_choices.append(horizon + 1)

    few_step_outputs = {}
    reverse_few_step_outputs = {}
    for num_steps in step_choices:
        outputs = []
        reverse_outputs = []
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
            reverse_outputs.append(
                generate_from_target_few_steps(
                    model,
                    reverse_target[sample_idx : sample_idx + 1],
                    horizon,
                    num_steps=num_steps,
                    noise_scale=inference_noise_scale,
                    clamp=(0.0, 1.0),
                )[-1].detach().cpu()
            )
        few_step_outputs[str(num_steps)] = torch.cat(outputs, dim=0)
        reverse_few_step_outputs[str(num_steps)] = torch.cat(reverse_outputs, dim=0)

    artifact_results_dir = results_dir if output_dir is None else output_dir
    return save_torch_artifact(
        artifact_results_dir,
        evaluation_filename,
        {
            "fresh_source": fresh_source.detach().cpu(),
            "source_original": source_original.detach().cpu(),
            "generated": generated.detach().cpu(),
            "reverse_target": reverse_target.detach().cpu(),
            "reverse_target_original": reverse_target_original.detach().cpu(),
            "reverse_generated": reverse_generated.detach().cpu(),
            "source_upsampled": source_upsampled.detach().cpu(),
            "generated_upsampled": generated_upsampled.detach().cpu(),
            "few_step_outputs": few_step_outputs,
            "reverse_few_step_outputs": reverse_few_step_outputs,
            "step_choices": step_choices,
            "width": width,
            "original_width": original_width,
            "reverse_digits": reverse_digits,
            "reverse_mnist_split": "test",
            "reverse_eval_target_train": False,
            "source_letters": source_metadata["source_letters"],
            "source_letter_labels": source_metadata["source_letter_labels"],
            "emnist_split": source_metadata["emnist_split"],
            "eval_source_train": False,
            "source_fix_orientation": source_metadata["source_fix_orientation"],
            "source_sampled_with_replacement": source_metadata["source_sampled_with_replacement"],
            "checkpoint_filename": checkpoint_filename,
            "checkpoint_path": str(resolved_checkpoint_path),
            "training_results_dir": str(results_dir),
        },
        subdir="evaluation",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained EMNIST-source to MNIST dual dynamic OT experiment.")
    parser.add_argument("--results-dir", required=True, help="Results directory created by training.")
    parser.add_argument("--checkpoint-filename", default="checkpoint_final.pt", help="Checkpoint filename under <results-dir>/checkpoints.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional explicit checkpoint path. If omitted, uses <results-dir>/checkpoints/<checkpoint-filename>.",
    )
    parser.add_argument("--evaluation-filename", default="evaluation.pt", help="Evaluation filename under <results-dir>/evaluation.")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of fresh EMNIST test samples to generate and save.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for the evaluation artifact. Defaults to results-dir.",
    )
    args = parser.parse_args()

    output_path = evaluate_emnist_mnist_checkpoint(
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
