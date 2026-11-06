from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final_upsample_plots.yaml"


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
    base_dir = Path(output_cfg.get("base_dir", "final_upsample")).expanduser()
    if not base_dir.is_absolute():
        base_dir = REPO_ROOT / base_dir

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _plot_image(ax, image_tensor, width: int):
    image = image_tensor.detach().cpu().numpy().reshape(width, width)
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")


def _assert_heldout_evaluation(evaluation: dict[str, Any]):
    split = evaluation.get("mnist_split")
    eval_data_train = evaluation.get("eval_data_train")
    if split != "test" or eval_data_train is not False:
        raise ValueError(
            "Evaluation artifact is not marked as held-out MNIST test data. "
            f"mnist_split={split!r}, eval_data_train={eval_data_train!r}"
        )


def _plot_source_above_generated(
    *,
    evaluation_path: Path,
    output_path: Path,
    num_samples: int,
    dpi: int,
    wspace: float,
    hspace: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dynamicot.utils import load_torch_artifact

    evaluation = load_torch_artifact(str(evaluation_path))
    _assert_heldout_evaluation(evaluation)

    fresh_source = evaluation["fresh_source"]
    generated = evaluation["generated"]
    width = int(evaluation["width"])
    available_samples = min(int(fresh_source.shape[0]), int(generated.shape[0]))

    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if num_samples > available_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but evaluation artifact only contains {available_samples}"
        )

    image_size = 0.55
    fig, axes = plt.subplots(
        2,
        num_samples,
        figsize=(
            image_size * (num_samples + max(num_samples - 1, 0) * wspace),
            image_size * (2.0 + hspace),
        ),
        squeeze=False,
    )
    for idx in range(num_samples):
        _plot_image(axes[0, idx], fresh_source[idx], width)
        _plot_image(axes[1, idx], generated[idx], width)

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


def _save_square_image_grid(
    *,
    images,
    width: int,
    num_samples: int,
    output_path: Path,
    dpi: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    ncols = int(math.ceil(math.sqrt(num_samples)))
    nrows = int(math.ceil(num_samples / ncols))
    image_batch = images[:num_samples].detach().cpu().reshape(num_samples, width, width)
    canvas = torch.zeros((nrows * width, ncols * width), dtype=image_batch.dtype)

    for idx in range(num_samples):
        row, col = divmod(idx, ncols)
        canvas[row * width : (row + 1) * width, col * width : (col + 1) * width] = image_batch[idx]

    fig = plt.figure(figsize=(0.55 * ncols, 0.55 * nrows), frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(canvas.numpy(), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.axis("off")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _validate_few_step_outputs(evaluation: dict[str, Any], steps: list[int]):
    few_step_outputs = evaluation["few_step_outputs"]
    missing_steps = [step for step in steps if str(step) not in few_step_outputs]
    if missing_steps:
        available_steps = sorted(int(step) for step in few_step_outputs)
        raise ValueError(f"Missing few-step outputs for {missing_steps}; available steps are {available_steps}")


def _plot_few_step_source_to_generated(
    *,
    evaluation_path: Path,
    output_path: Path,
    num_samples: int,
    steps: list[int],
    dpi: int,
    wspace: float,
    hspace: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dynamicot.utils import load_torch_artifact

    evaluation = load_torch_artifact(str(evaluation_path))
    _assert_heldout_evaluation(evaluation)
    _validate_few_step_outputs(evaluation, steps)

    fresh_source = evaluation["fresh_source"]
    few_step_outputs = evaluation["few_step_outputs"]
    width = int(evaluation["width"])
    available_samples = int(fresh_source.shape[0])
    if num_samples > available_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but evaluation artifact only contains {available_samples}"
        )

    ncols = 1 + len(steps)
    image_size = 0.55
    fig, axes = plt.subplots(
        num_samples,
        ncols,
        figsize=(
            image_size * (ncols + max(ncols - 1, 0) * wspace),
            image_size * (num_samples + max(num_samples - 1, 0) * hspace),
        ),
        squeeze=False,
    )
    for sample_idx in range(num_samples):
        _plot_image(axes[sample_idx, 0], fresh_source[sample_idx], width)
        for step_idx, step in enumerate(steps, start=1):
            _plot_image(axes[sample_idx, step_idx], few_step_outputs[str(step)][sample_idx], width)

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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a dual_mnist_upsample checkpoint on held-out MNIST test data and "
            "plot each fresh source vertically above its generated sample."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the final upsample plotting YAML config.",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = _load_yaml_config(config_path)

    from dynamicot.experiments.dual_mnist_upsample_eval import evaluate_mnist_upsample_checkpoint
    from dynamicot.utils import load_experiment_summary, load_torch_artifact

    checkpoint_path = _resolve_checkpoint_path(config)
    training_results_dir = _infer_training_results_dir(checkpoint_path)
    training_config = load_experiment_summary(str(training_results_dir))
    if not bool(training_config.get("data", {}).get("train", True)):
        raise ValueError(
            "The checkpoint's experiment_config.json is not marked as MNIST train split. "
            "Refusing to evaluate because held-out status cannot be guaranteed."
        )

    num_samples = int(config.get("num_samples", 50))
    evaluation_filename = str(config.get("evaluation_filename", "evaluation.pt"))
    output_cfg = config.get("output", {})
    plot_filename = str(output_cfg.get("plot_filename", f"source_above_generated_n{num_samples}.png"))
    source_grid_filename = str(output_cfg.get("source_grid_filename", f"fresh_source_grid_n{num_samples}.png"))
    generated_grid_filename = str(output_cfg.get("generated_grid_filename", f"generated_grid_n{num_samples}.png"))
    few_step_plot_filename = str(output_cfg.get("few_step_plot_filename", f"few_step_source_to_generated_n{num_samples}.png"))
    few_step_grid_pattern = str(output_cfg.get("few_step_grid_filename_pattern", "few_step_generated_grid_{step}_steps.png"))
    dpi = int(output_cfg.get("dpi", 200))
    image_gap_fraction = float(output_cfg.get("image_gap_fraction", 0.1))
    wspace = float(output_cfg.get("wspace", image_gap_fraction))
    hspace = float(output_cfg.get("hspace", image_gap_fraction))
    few_step_steps = [int(step) for step in config.get("few_step_steps", [1, 2, 4, 5, 10, 25, 50, 100])]

    output_dir = _make_output_dir(config)
    shutil.copy2(config_path, output_dir / config_path.name)

    evaluation_path = Path(
        evaluate_mnist_upsample_checkpoint(
            str(training_results_dir),
            checkpoint_filename=checkpoint_path.name,
            checkpoint_path=str(checkpoint_path),
            evaluation_filename=evaluation_filename,
            num_samples=num_samples,
            output_dir=str(output_dir),
        )
    )
    evaluation = load_torch_artifact(str(evaluation_path))
    _assert_heldout_evaluation(evaluation)

    plot_path = output_dir / plot_filename
    _plot_source_above_generated(
        evaluation_path=evaluation_path,
        output_path=plot_path,
        num_samples=num_samples,
        dpi=dpi,
        wspace=wspace,
        hspace=hspace,
    )

    width = int(evaluation["width"])
    source_grid_path = output_dir / source_grid_filename
    generated_grid_path = output_dir / generated_grid_filename
    _save_square_image_grid(
        images=evaluation["fresh_source"],
        width=width,
        num_samples=num_samples,
        output_path=source_grid_path,
        dpi=dpi,
    )
    _save_square_image_grid(
        images=evaluation["generated"],
        width=width,
        num_samples=num_samples,
        output_path=generated_grid_path,
        dpi=dpi,
    )

    few_step_plot_path = output_dir / few_step_plot_filename
    _plot_few_step_source_to_generated(
        evaluation_path=evaluation_path,
        output_path=few_step_plot_path,
        num_samples=num_samples,
        steps=few_step_steps,
        dpi=dpi,
        wspace=wspace,
        hspace=hspace,
    )

    _validate_few_step_outputs(evaluation, few_step_steps)
    few_step_grid_paths = []
    for step in few_step_steps:
        grid_path = output_dir / few_step_grid_pattern.format(step=step, num_samples=num_samples)
        _save_square_image_grid(
            images=evaluation["few_step_outputs"][str(step)],
            width=width,
            num_samples=num_samples,
            output_path=grid_path,
            dpi=dpi,
        )
        few_step_grid_paths.append(grid_path)

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "training_results_dir": str(training_results_dir),
        "training_mnist_split": "train",
        "evaluation_mnist_split": "test",
        "evaluation_data_train": False,
        "num_samples": num_samples,
        "evaluation_artifact": str(evaluation_path.relative_to(output_dir)),
        "plot": plot_path.name,
        "source_grid_plot": source_grid_path.name,
        "generated_grid_plot": generated_grid_path.name,
        "few_step_steps": few_step_steps,
        "few_step_plot": few_step_plot_path.name,
        "few_step_grid_plots": [path.name for path in few_step_grid_paths],
        "config_copy": config_path.name,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved held-out evaluation artifact: {evaluation_path}")
    print(f"Saved source/generated plot: {plot_path}")
    print(f"Saved source grid plot: {source_grid_path}")
    print(f"Saved generated grid plot: {generated_grid_path}")
    print(f"Saved few-step source/generated plot: {few_step_plot_path}")
    for grid_path in few_step_grid_paths:
        print(f"Saved few-step generated grid plot: {grid_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
