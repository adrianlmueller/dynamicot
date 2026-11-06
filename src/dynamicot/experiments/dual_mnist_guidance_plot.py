from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils import load_experiment_summary, load_json, load_torch_artifact


def _plot_image(ax, image_tensor, width: int, title: str):
    ax.imshow(image_tensor.numpy().reshape(width, width), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")


def _filename_suffix(
    *,
    evaluation_filename: str,
    checkpoint_filename: str,
    filename_suffix: str | None,
):
    if filename_suffix is not None:
        return f"_{filename_suffix}" if filename_suffix else ""
    if checkpoint_filename != "checkpoint_final.pt":
        return f"_{Path(checkpoint_filename).stem}"
    if evaluation_filename != "evaluation.pt":
        return f"_{Path(evaluation_filename).stem}"
    return ""


def _plot_metrics(config: dict, metrics: dict, output_path: Path):
    architecture = str(config["model"].get("architecture", "cnn")).upper()
    plot_paths: list[str] = []

    plt.figure(figsize=(10, 6))
    plt.plot(metrics["dual_loss_history"], label="Total Dual Loss (Dual Function Value)", color="blue")
    plt.title("Total Dual Loss over Training Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Total Dual Loss")
    plt.legend()
    plt.grid()
    loss_plot_path = output_path / "dual_loss_over_iterations.png"
    plt.savefig(loss_plot_path)
    plt.close()
    plot_paths.append(str(loss_plot_path))

    plt.figure(figsize=(12, 8))
    plt.plot(metrics["parameter_norm_history"], label=f"Unified {architecture} Parameters L2 Norm")
    plt.title(f"Evolution of Conditional {architecture} Parameter L2 Norms During Training")
    plt.xlabel("Training Iteration")
    plt.ylabel("L2 Norm of Parameters")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    norm_plot_path = output_path / "model_parameter_norms.png"
    plt.savefig(norm_plot_path)
    plt.close()
    plot_paths.append(str(norm_plot_path))

    return plot_paths


def _validate_plot_samples(evaluation: dict, plot_samples: int | None):
    available = int(evaluation["num_samples_per_class"])
    resolved = available if plot_samples is None else int(plot_samples)
    if resolved < 1:
        raise ValueError("num_samples must be at least 1")
    if resolved > available:
        raise ValueError(
            f"Requested {resolved} samples to plot, but the evaluation only contains {available} per class. "
            "Re-run dual_mnist_guidance_eval with a larger --num-samples-per-class first."
        )
    return resolved


def _plot_conditional_grid(
    evaluation: dict,
    *,
    output_path: Path,
    suffix: str,
    plot_samples: int,
    guided: bool,
):
    width = int(evaluation["width"])
    digits = list(evaluation["digits"])
    class_samples = evaluation["class_samples"]
    num_rows = 4 * len(digits)
    fig, axes = plt.subplots(
        num_rows,
        plot_samples,
        figsize=(3.0 * plot_samples, 2.7 * num_rows),
        squeeze=False,
    )
    generated_key = "guided_generated_y" if guided else "generated_y"
    generated_fresh_key = "guided_generated_y_fresh" if guided else "generated_y_fresh"
    mode = f"guided, scale={evaluation['guidance_scale']:.3g}" if guided else "unguided"

    for class_idx, digit in enumerate(digits):
        samples = class_samples[str(digit)]
        row = 4 * class_idx
        for sample_idx in range(plot_samples):
            _plot_image(axes[row, sample_idx], samples["x_star"][sample_idx], width, f"Digit {digit} X {sample_idx}")
            _plot_image(axes[row + 1, sample_idx], samples["y_star"][sample_idx], width, f"Digit {digit} target {sample_idx}")
            _plot_image(
                axes[row + 2, sample_idx],
                samples[generated_key][sample_idx],
                width,
                f"Digit {digit} generated {sample_idx}",
            )
            _plot_image(
                axes[row + 3, sample_idx],
                samples[generated_fresh_key][sample_idx],
                width,
                f"Digit {digit} fresh {sample_idx}",
            )

    plt.suptitle(f"Conditional MNIST generated samples by digit ({mode})")
    plt.tight_layout()
    infix = "_guidance" if guided else ""
    plot_path = output_path / f"conditional_generated_data_samples{infix}{suffix}.png"
    plt.savefig(plot_path)
    plt.close()
    return str(plot_path)


def _plot_conditional_upsampled_grid(
    evaluation: dict,
    *,
    output_path: Path,
    suffix: str,
    plot_samples: int,
    guided: bool,
):
    original_width = int(evaluation["original_width"])
    digits = list(evaluation["digits"])
    class_samples = evaluation["class_samples"]
    num_rows = 4 * len(digits)
    fig, axes = plt.subplots(
        num_rows,
        plot_samples,
        figsize=(3.0 * plot_samples, 2.7 * num_rows),
        squeeze=False,
    )
    generated_key = "guided_generated_y_upsampled" if guided else "generated_y_upsampled"
    generated_fresh_key = "guided_generated_y_fresh_upsampled" if guided else "generated_y_fresh_upsampled"
    mode = f"guided, scale={evaluation['guidance_scale']:.3g}" if guided else "unguided"

    for class_idx, digit in enumerate(digits):
        samples = class_samples[str(digit)]
        row = 4 * class_idx
        for sample_idx in range(plot_samples):
            _plot_image(
                axes[row, sample_idx],
                samples["target_original"][sample_idx],
                original_width,
                f"Digit {digit} target {sample_idx} original",
            )
            _plot_image(
                axes[row + 1, sample_idx],
                samples["y_star_upsampled"][sample_idx],
                original_width,
                f"Digit {digit} target {sample_idx} upsampled",
            )
            _plot_image(
                axes[row + 2, sample_idx],
                samples[generated_key][sample_idx],
                original_width,
                f"Digit {digit} generated {sample_idx}",
            )
            _plot_image(
                axes[row + 3, sample_idx],
                samples[generated_fresh_key][sample_idx],
                original_width,
                f"Digit {digit} fresh {sample_idx}",
            )

    plt.suptitle(f"Conditional MNIST generated samples by digit, upsampled to original size ({mode})")
    plt.tight_layout()
    infix = "_guidance" if guided else ""
    plot_path = output_path / f"conditional_generated_data_samples_upsampled{infix}{suffix}.png"
    plt.savefig(plot_path)
    plt.close()
    return str(plot_path)


def plot_mnist_guidance_evaluation(
    results_dir: str,
    *,
    evaluation_filename: str = "evaluation.pt",
    evaluation_dir: str | None = None,
    output_dir: str | None = None,
    filename_suffix: str | None = None,
    skip_metrics: bool = False,
    num_samples: int | None = None,
):
    results_path = Path(results_dir)
    evaluation_path = Path(evaluation_dir) if evaluation_dir is not None else results_path / "evaluation"
    output_path = Path(output_dir) if output_dir is not None else results_path
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_experiment_summary(str(results_path))
    evaluation = load_torch_artifact(str(evaluation_path / evaluation_filename))
    checkpoint_filename = evaluation.get("checkpoint_filename", "checkpoint_final.pt")
    suffix = _filename_suffix(
        evaluation_filename=evaluation_filename,
        checkpoint_filename=checkpoint_filename,
        filename_suffix=filename_suffix,
    )

    plot_paths: list[str] = []
    if not skip_metrics:
        metrics = load_json(str(results_path / "metrics.json"))
        plot_paths.extend(_plot_metrics(config, metrics, output_path))

    plot_samples = _validate_plot_samples(evaluation, num_samples)
    plot_paths.append(
        _plot_conditional_grid(
            evaluation,
            output_path=output_path,
            suffix=suffix,
            plot_samples=plot_samples,
            guided=False,
        )
    )
    plot_paths.append(
        _plot_conditional_grid(
            evaluation,
            output_path=output_path,
            suffix=suffix,
            plot_samples=plot_samples,
            guided=True,
        )
    )
    plot_paths.append(
        _plot_conditional_upsampled_grid(
            evaluation,
            output_path=output_path,
            suffix=suffix,
            plot_samples=plot_samples,
            guided=False,
        )
    )
    plot_paths.append(
        _plot_conditional_upsampled_grid(
            evaluation,
            output_path=output_path,
            suffix=suffix,
            plot_samples=plot_samples,
            guided=True,
        )
    )

    return tuple(plot_paths)


def main():
    parser = argparse.ArgumentParser(description="Plot artifacts from a trained conditional MNIST guidance experiment.")
    parser.add_argument("--results-dir", required=True, help="Results directory created by training.")
    parser.add_argument("--evaluation-filename", default="evaluation.pt", help="Evaluation filename under <results-dir>/evaluation.")
    parser.add_argument(
        "--evaluation-dir",
        default=None,
        help="Directory containing the evaluation artifact. Defaults to <results-dir>/evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where plots are written. Defaults to <results-dir>.",
    )
    parser.add_argument(
        "--filename-suffix",
        default=None,
        help="Suffix appended to generated plot filenames. Defaults to the checkpoint/evaluation label.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only plot the selected evaluation artifact, without re-saving training metric plots.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of saved samples per class to plot. Defaults to all samples in the selected evaluation file.",
    )
    args = parser.parse_args()

    for plot_path in plot_mnist_guidance_evaluation(
        args.results_dir,
        evaluation_filename=args.evaluation_filename,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        filename_suffix=args.filename_suffix,
        skip_metrics=args.skip_metrics,
        num_samples=args.num_samples,
    ):
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
