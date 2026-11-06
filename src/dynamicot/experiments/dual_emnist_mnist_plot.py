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


def plot_emnist_mnist_evaluation(
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
    if filename_suffix is not None:
        suffix = f"_{filename_suffix}" if filename_suffix else ""
    elif checkpoint_filename != "checkpoint_final.pt":
        suffix = f"_{Path(checkpoint_filename).stem}"
    elif evaluation_filename != "evaluation.pt":
        suffix = f"_{Path(evaluation_filename).stem}"
    else:
        suffix = ""

    plot_paths: list[str] = []
    if not skip_metrics:
        metrics = load_json(str(results_path / "metrics.json"))
        architecture = str(config["model"].get("architecture", "cnn")).upper()

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
        plt.title(f"Evolution of {architecture} Parameter L2 Norms During Training")
        plt.xlabel("Training Iteration")
        plt.ylabel("L2 Norm of Parameters")
        plt.legend(loc="best", fontsize="small")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        norm_plot_path = output_path / "model_parameter_norms.png"
        plt.savefig(norm_plot_path)
        plt.close()
        plot_paths.append(str(norm_plot_path))

    width = int(evaluation["width"])
    original_width = int(evaluation["original_width"])
    fresh_source = evaluation["fresh_source"]
    source_original = evaluation["source_original"]
    generated = evaluation["generated"]
    source_upsampled = evaluation["source_upsampled"]
    generated_upsampled = evaluation["generated_upsampled"]
    few_step_outputs = evaluation["few_step_outputs"]
    step_choices = evaluation["step_choices"]

    available_samples = fresh_source.shape[0]
    plot_samples = available_samples if num_samples is None else num_samples
    if plot_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if plot_samples > available_samples:
        raise ValueError(
            f"Requested {plot_samples} samples to plot, but {evaluation_filename} only contains {available_samples}. "
            "Re-run dual_emnist_mnist_eval with a larger --num-samples first."
        )

    fig, axes = plt.subplots(2, plot_samples, figsize=(3.0 * plot_samples, 6), squeeze=False)
    for idx in range(plot_samples):
        _plot_image(axes[0, idx], fresh_source[idx], width, f"Fresh EMNIST Source {idx}")
        _plot_image(axes[1, idx], generated[idx], width, f"Generated {idx}")
    plt.suptitle(f"Fresh EMNIST test source samples and generated outputs (n={plot_samples})")
    plt.tight_layout()
    final_plot_path = output_path / f"final_generated_data_samples{suffix}.png"
    plt.savefig(final_plot_path)
    plt.close()
    plot_paths.append(str(final_plot_path))

    fig, axes = plt.subplots(3, plot_samples, figsize=(3.0 * plot_samples, 9), squeeze=False)
    for idx in range(plot_samples):
        _plot_image(axes[0, idx], source_original[idx], original_width, f"Fresh EMNIST Source {idx} (Original)")
        _plot_image(axes[1, idx], source_upsampled[idx], original_width, f"Fresh EMNIST Source {idx} (Upsampled)")
        _plot_image(axes[2, idx], generated_upsampled[idx], original_width, f"Generated {idx}")
    plt.suptitle(f"Fresh EMNIST test source samples and generated outputs, upsampled to original size")
    plt.tight_layout()
    upsampled_plot_path = output_path / f"final_generated_data_samples_upsampled{suffix}.png"
    plt.savefig(upsampled_plot_path)
    plt.close()
    plot_paths.append(str(upsampled_plot_path))

    fig, axes = plt.subplots(
        plot_samples,
        len(step_choices),
        figsize=(3.0 * len(step_choices), 3.0 * plot_samples),
        squeeze=False,
    )
    for sample_idx in range(plot_samples):
        for choice_idx, num_steps in enumerate(step_choices):
            _plot_image(
                axes[sample_idx, choice_idx],
                few_step_outputs[str(num_steps)][sample_idx],
                width,
                f"Sample {sample_idx}, Steps {num_steps}",
            )
    plt.suptitle("Generated EMNIST-source test samples with different numbers of inference steps")
    plt.tight_layout()
    few_steps_plot_path = output_path / f"generated_data_few_steps{suffix}.png"
    plt.savefig(few_steps_plot_path)
    plt.close()
    plot_paths.append(str(few_steps_plot_path))

    return tuple(plot_paths)


def main():
    parser = argparse.ArgumentParser(description="Plot artifacts from a trained EMNIST-source to MNIST dual dynamic OT experiment.")
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
        help="Number of saved evaluation samples to plot. Defaults to all samples in the selected evaluation file.",
    )
    args = parser.parse_args()

    for plot_path in plot_emnist_mnist_evaluation(
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
