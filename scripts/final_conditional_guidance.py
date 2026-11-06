from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final_conditional_guidance.yaml"


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load experiment configs. Install it with `pip install pyyaml`."
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} must define a mapping at the top level")
    return config


def _resolve_config_path(path: str):
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    return config_path.resolve()


def _resolve_checkpoint_path(config: dict[str, Any]):
    checkpoint_value = config.get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError("Config must define checkpoint_path")

    checkpoint_path = Path(str(checkpoint_value)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path.resolve()


def _infer_training_results_dir(checkpoint_path: Path):
    if checkpoint_path.parent.name == "checkpoints":
        results_dir = checkpoint_path.parent.parent
        if (results_dir / "experiment_config.json").is_file():
            return results_dir

    raise ValueError(
        "Could not infer the training results directory. Expected checkpoint_path to point into "
        "<results_dir>/checkpoints/ with <results_dir>/experiment_config.json present."
    )


def _make_output_dir(config: dict[str, Any]):
    output_cfg = config.get("output", {})
    base_dir = Path(output_cfg.get("base_dir", "final_conditional_guidance")).expanduser()
    if not base_dir.is_absolute():
        base_dir = REPO_ROOT / base_dir

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _scale_label(scale: float):
    return f"{scale:g}".replace("-", "m").replace(".", "p")


def _plot_digit_grid(
    *,
    generated_by_digit: dict[int, Any],
    digits: list[int],
    width: int,
    num_samples_per_label: int,
    output_path: Path,
    dpi: int,
    wspace: float,
    hspace: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_size = 0.55
    ncols = len(digits)
    nrows = num_samples_per_label
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            image_size * (ncols + max(ncols - 1, 0) * wspace),
            image_size * (nrows + max(nrows - 1, 0) * hspace),
        ),
        squeeze=False,
    )

    for col_idx, digit in enumerate(digits):
        samples = generated_by_digit[int(digit)].detach().cpu()
        for row_idx in range(num_samples_per_label):
            axes[row_idx, col_idx].imshow(
                samples[row_idx].numpy().reshape(width, width),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row_idx, col_idx].axis("off")

    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
        wspace=wspace,
        hspace=hspace,
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _evaluate_conditional_guidance(
    *,
    checkpoint_path: Path,
    training_results_dir: Path,
    config: dict[str, Any],
    output_dir: Path,
):
    import numpy as np
    import torch

    from dynamicot.data import build_image_source_data
    from dynamicot.experiments.dual_mnist_guidance import (
        _label_values_for_digits,
        _resolve_digits,
        _unconditional_label_from_config,
        build_time_label_conditioned_image_model,
        generate_from_source_guided,
    )
    from dynamicot.utils import load_checkpoint, load_experiment_summary, resolve_device, resolve_dtype, set_seed

    training_config = load_experiment_summary(str(training_results_dir))
    data_cfg = training_config.get("data", {})
    if not bool(data_cfg.get("train", True)):
        raise ValueError(
            "The checkpoint's experiment_config.json is not marked as MNIST train split. "
            "Refusing to evaluate because checkpoint provenance is not the expected conditional setup."
        )
    if "conditioning" not in training_config.get("training", {}):
        raise ValueError("The checkpoint config does not contain training.conditioning; is this a conditional checkpoint?")

    seed = config.get("seed", training_config.get("seed"))
    if seed is not None:
        set_seed(int(seed))

    device = resolve_device(training_config.get("device", "cpu"))
    dtype = resolve_dtype(training_config.get("dtype", "float32"))
    width = int(data_cfg.get("width", 28))
    digits = _resolve_digits(data_cfg)
    label_values = _label_values_for_digits(digits)
    unconditional_label = _unconditional_label_from_config(training_config)
    model_cfg = training_config["model"]
    input_channels = int(model_cfg.get("input_channels", 1))

    model = build_time_label_conditioned_image_model(
        model_cfg,
        image_size=width,
        input_channels=input_channels,
    ).to(device)
    checkpoint = load_checkpoint(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    num_samples_per_label = int(config.get("num_samples_per_label", config.get("num_samples", 5)))
    if num_samples_per_label < 1:
        raise ValueError("num_samples_per_label must be at least 1")

    guidance_scales = [float(scale) for scale in config.get("guidance_scales", [])]
    if any(scale < 0.0 for scale in guidance_scales):
        raise ValueError("guidance_scales must be non-negative")

    fresh_source = build_image_source_data(
        data_cfg.get("source"),
        n_samples=num_samples_per_label,
        width=width,
        input_channels=input_channels,
        device=device,
        dtype=dtype,
    )

    horizon = int(training_config["training"]["horizon"])
    inference_noise_scale = float(training_config["training"]["primal"]["noise_scale"]) / np.sqrt(
        2.0 * float(training_config["training"]["primal"]["lr"])
    )

    generated_by_digit: dict[int, torch.Tensor] = {}
    guided_by_scale: dict[str, dict[int, torch.Tensor]] = {}
    for digit in digits:
        labels = torch.full(
            (num_samples_per_label, 1),
            float(label_values[int(digit)]),
            device=device,
            dtype=dtype,
        )
        generated_by_digit[int(digit)] = generate_from_source_guided(
            model,
            fresh_source,
            horizon,
            input_labels=labels,
            unconditional_label=unconditional_label,
            noise_scale=inference_noise_scale,
            clamp=(0.0, 1.0),
        )[-1].detach().cpu()

        for guidance_scale in guidance_scales:
            scale_label = _scale_label(guidance_scale)
            guided_by_scale.setdefault(scale_label, {})[int(digit)] = generate_from_source_guided(
                model,
                fresh_source,
                horizon,
                input_labels=labels,
                unconditional_label=unconditional_label,
                guidance_scale=guidance_scale,
                noise_scale=inference_noise_scale,
                clamp=(0.0, 1.0),
            )[-1].detach().cpu()

    evaluation = {
        "fresh_source": fresh_source.detach().cpu(),
        "generated_by_digit": generated_by_digit,
        "guided_by_scale": guided_by_scale,
        "digits": digits,
        "label_values": label_values,
        "unconditional_label": unconditional_label,
        "guidance_scales": guidance_scales,
        "num_samples_per_label": num_samples_per_label,
        "width": width,
        "source_config": data_cfg.get("source"),
        "source_is_fresh": True,
        "checkpoint_path": str(checkpoint_path),
        "training_results_dir": str(training_results_dir),
    }

    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    evaluation_filename = str(config.get("evaluation_filename", "evaluation.pt"))
    evaluation_path = evaluation_dir / evaluation_filename
    torch.save(evaluation, evaluation_path)
    return evaluation, evaluation_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot fresh-source conditional and guided MNIST generations from a conditional guidance checkpoint."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the final conditional guidance YAML config.",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = _load_yaml_config(config_path)

    checkpoint_path = _resolve_checkpoint_path(config)
    training_results_dir = _infer_training_results_dir(checkpoint_path)
    output_cfg = config.get("output", {})
    output_dir = _make_output_dir(config)
    shutil.copy2(config_path, output_dir / config_path.name)

    evaluation, evaluation_path = _evaluate_conditional_guidance(
        checkpoint_path=checkpoint_path,
        training_results_dir=training_results_dir,
        config=config,
        output_dir=output_dir,
    )

    digits = [int(digit) for digit in evaluation["digits"]]
    width = int(evaluation["width"])
    num_samples_per_label = int(evaluation["num_samples_per_label"])
    dpi = int(output_cfg.get("dpi", 200))
    image_gap_fraction = float(output_cfg.get("image_gap_fraction", 0.1))
    wspace = float(output_cfg.get("wspace", image_gap_fraction))
    hspace = float(output_cfg.get("hspace", image_gap_fraction))

    conditional_plot_filename = str(output_cfg.get("conditional_plot_filename", "conditional_generation.png"))
    guidance_plot_pattern = str(output_cfg.get("guidance_plot_filename_pattern", "guided_generation_scale_{scale_label}.png"))

    conditional_plot_path = output_dir / conditional_plot_filename
    _plot_digit_grid(
        generated_by_digit=evaluation["generated_by_digit"],
        digits=digits,
        width=width,
        num_samples_per_label=num_samples_per_label,
        output_path=conditional_plot_path,
        dpi=dpi,
        wspace=wspace,
        hspace=hspace,
    )

    guidance_plot_paths = []
    for guidance_scale in evaluation["guidance_scales"]:
        scale_label = _scale_label(float(guidance_scale))
        plot_path = output_dir / guidance_plot_pattern.format(
            scale=guidance_scale,
            scale_label=scale_label,
            num_samples_per_label=num_samples_per_label,
        )
        _plot_digit_grid(
            generated_by_digit=evaluation["guided_by_scale"][scale_label],
            digits=digits,
            width=width,
            num_samples_per_label=num_samples_per_label,
            output_path=plot_path,
            dpi=dpi,
            wspace=wspace,
            hspace=hspace,
        )
        guidance_plot_paths.append(plot_path)

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "training_results_dir": str(training_results_dir),
        "source_is_fresh": True,
        "source_config": evaluation["source_config"],
        "digits": digits,
        "label_values": evaluation["label_values"],
        "guidance_scales": evaluation["guidance_scales"],
        "num_samples_per_label": num_samples_per_label,
        "evaluation_artifact": str(evaluation_path.relative_to(output_dir)),
        "conditional_plot": conditional_plot_path.name,
        "guidance_plots": [path.name for path in guidance_plot_paths],
        "config_copy": config_path.name,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved evaluation artifact: {evaluation_path}")
    print(f"Saved conditional plot: {conditional_plot_path}")
    for plot_path in guidance_plot_paths:
        print(f"Saved guided plot: {plot_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
