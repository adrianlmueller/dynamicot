from __future__ import annotations

import argparse

import numpy as np
import torch

from ..data import build_image_source_data
from ..utils import load_checkpoint, load_experiment_summary, save_torch_artifact
from .dual_mnist_guidance import generate_from_source_guided, prepare_mnist_guidance_experiment


def _upsample_flat_images(batch: torch.Tensor, *, width: int, original_width: int):
    if width == original_width:
        return batch
    return torch.nn.functional.interpolate(
        batch.view(-1, 1, width, width),
        size=(original_width, original_width),
        mode="bilinear",
        align_corners=False,
    ).view(-1, original_width * original_width)


def _evaluation_defaults(
    config: dict,
    *,
    num_samples_per_class: int | None,
    guidance_scale: float | None,
):
    evaluation_cfg = config.get("evaluation", {})
    resolved_num_samples = (
        int(num_samples_per_class)
        if num_samples_per_class is not None
        else int(evaluation_cfg.get("num_samples_per_class", 5))
    )
    resolved_guidance_scale = (
        float(guidance_scale)
        if guidance_scale is not None
        else float(evaluation_cfg.get("guidance_scale", 2.0))
    )
    if resolved_num_samples < 1:
        raise ValueError("num_samples_per_class must be at least 1")
    if resolved_guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    return resolved_num_samples, resolved_guidance_scale


def evaluate_mnist_guidance_checkpoint(
    results_dir: str,
    *,
    checkpoint_filename: str = "checkpoint_final.pt",
    evaluation_filename: str = "evaluation.pt",
    num_samples_per_class: int | None = None,
    guidance_scale: float | None = None,
):
    config = load_experiment_summary(results_dir)
    num_samples_per_class, guidance_scale = _evaluation_defaults(
        config,
        num_samples_per_class=num_samples_per_class,
        guidance_scale=guidance_scale,
    )
    model, source_data, target_data, metadata = prepare_mnist_guidance_experiment(config)
    device = next(model.parameters()).device
    dtype = source_data.dtype
    checkpoint = load_checkpoint(
        f"{results_dir}/checkpoints/{checkpoint_filename}",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    width = int(metadata["width"])
    original_width = int(metadata.get("original_width", width))
    input_channels = int(config["model"].get("input_channels", 1))
    horizon = int(config["training"]["horizon"])
    unconditional_label = float(metadata["unconditional_label"])
    inference_noise_scale = float(config["training"]["primal"]["noise_scale"]) / np.sqrt(
        2.0 * float(config["training"]["primal"]["lr"])
    )

    fresh_source_template = build_image_source_data(
        config["data"].get("source"),
        n_samples=num_samples_per_class,
        width=width,
        input_channels=input_channels,
        device=device,
        dtype=dtype,
    )

    target_digit_labels = metadata["target_digit_labels"]
    target_labels = metadata["target_labels"]
    class_samples = {}
    for digit in metadata["digits"]:
        digit_indices = (target_digit_labels == int(digit)).nonzero(as_tuple=True)[0]
        if digit_indices.numel() < num_samples_per_class:
            raise ValueError(
                f"Requested {num_samples_per_class} samples for MNIST digit {digit}, "
                f"but only {digit_indices.numel()} are available"
            )
        selected = digit_indices[:num_samples_per_class]
        x_star = source_data[selected].to(device)
        y_star = target_data[selected].to(device)
        target_original = metadata["target_original"][selected].to(device)
        labels = target_labels[selected].to(device)
        x_star_fresh = fresh_source_template.detach().clone()

        generated_y = generate_from_source_guided(
            model,
            x_star,
            horizon,
            input_labels=labels,
            unconditional_label=unconditional_label,
            noise_scale=inference_noise_scale,
            clamp=(0.0, 1.0),
        )[-1]
        generated_y_fresh = generate_from_source_guided(
            model,
            x_star_fresh,
            horizon,
            input_labels=labels,
            unconditional_label=unconditional_label,
            noise_scale=inference_noise_scale,
            clamp=(0.0, 1.0),
        )[-1]
        guided_generated_y = generate_from_source_guided(
            model,
            x_star,
            horizon,
            input_labels=labels,
            unconditional_label=unconditional_label,
            guidance_scale=guidance_scale,
            noise_scale=inference_noise_scale,
            clamp=(0.0, 1.0),
        )[-1]
        guided_generated_y_fresh = generate_from_source_guided(
            model,
            x_star_fresh,
            horizon,
            input_labels=labels,
            unconditional_label=unconditional_label,
            guidance_scale=guidance_scale,
            noise_scale=inference_noise_scale,
            clamp=(0.0, 1.0),
        )[-1]

        y_star_upsampled = _upsample_flat_images(y_star, width=width, original_width=original_width)
        generated_y_upsampled = _upsample_flat_images(generated_y, width=width, original_width=original_width)
        generated_y_fresh_upsampled = _upsample_flat_images(
            generated_y_fresh,
            width=width,
            original_width=original_width,
        )
        guided_generated_y_upsampled = _upsample_flat_images(
            guided_generated_y,
            width=width,
            original_width=original_width,
        )
        guided_generated_y_fresh_upsampled = _upsample_flat_images(
            guided_generated_y_fresh,
            width=width,
            original_width=original_width,
        )

        class_samples[str(digit)] = {
            "label": labels[:1].detach().cpu(),
            "x_star": x_star.detach().cpu(),
            "y_star": y_star.detach().cpu(),
            "target_original": target_original.detach().cpu(),
            "x_star_fresh": x_star_fresh.detach().cpu(),
            "generated_y": generated_y.detach().cpu(),
            "generated_y_fresh": generated_y_fresh.detach().cpu(),
            "guided_generated_y": guided_generated_y.detach().cpu(),
            "guided_generated_y_fresh": guided_generated_y_fresh.detach().cpu(),
            "y_star_upsampled": y_star_upsampled.detach().cpu(),
            "generated_y_upsampled": generated_y_upsampled.detach().cpu(),
            "generated_y_fresh_upsampled": generated_y_fresh_upsampled.detach().cpu(),
            "guided_generated_y_upsampled": guided_generated_y_upsampled.detach().cpu(),
            "guided_generated_y_fresh_upsampled": guided_generated_y_fresh_upsampled.detach().cpu(),
        }

    return save_torch_artifact(
        results_dir,
        evaluation_filename,
        {
            "class_samples": class_samples,
            "digits": list(metadata["digits"]),
            "label_values": dict(metadata["label_values"]),
            "unconditional_label": unconditional_label,
            "guidance_scale": guidance_scale,
            "num_samples_per_class": num_samples_per_class,
            "width": width,
            "original_width": original_width,
            "checkpoint_filename": checkpoint_filename,
        },
        subdir="evaluation",
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained conditional MNIST guidance dual dynamic OT experiment.")
    parser.add_argument("--results-dir", required=True, help="Results directory created by training.")
    parser.add_argument("--checkpoint-filename", default="checkpoint_final.pt", help="Checkpoint filename under <results-dir>/checkpoints.")
    parser.add_argument("--evaluation-filename", default="evaluation.pt", help="Evaluation filename under <results-dir>/evaluation.")
    parser.add_argument("--num-samples-per-class", type=int, default=None, help="Number of samples per MNIST digit to generate and save.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Classifier-free guidance scale for guided samples.")
    args = parser.parse_args()

    output_path = evaluate_mnist_guidance_checkpoint(
        args.results_dir,
        checkpoint_filename=args.checkpoint_filename,
        evaluation_filename=args.evaluation_filename,
        num_samples_per_class=args.num_samples_per_class,
        guidance_scale=args.guidance_scale,
    )
    print(f"Saved evaluation artifact: {output_path}")


if __name__ == "__main__":
    main()
