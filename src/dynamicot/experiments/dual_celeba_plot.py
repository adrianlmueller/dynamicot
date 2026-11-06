from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils import load_json, load_torch_artifact


def plot_rgb_image(ax, image_tensor, width: int, title: str | None = None):
    ax.imshow(image_tensor.detach().cpu().view(3, width, width).permute(1, 2, 0).clamp(0.0, 1.0).numpy())
    if title is not None:
        ax.set_title(title)
    ax.axis("off")


def plot_celeba_evaluation(
    results_dir: str,
    *,
    evaluation_filename: str = "evaluation.pt",
    filename_suffix: str = "",
    skip_metrics: bool = False,
):
    results_path = Path(results_dir)
    evaluation = load_torch_artifact(str(results_path / "evaluation" / evaluation_filename))
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    plot_paths: list[str] = []

    if not skip_metrics:
        metrics = load_json(str(results_path / "metrics.json"))
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(metrics["dual_loss_history"], color="blue")
        axes[0].set_title("Total Dual Loss over Training Iterations")
        axes[1].plot(metrics["parameter_norm_history"], color="tab:orange")
        axes[1].set_title("Parameter L2 Norm over Training Iterations")
        for ax in axes:
            ax.set_xlabel("Training Iteration")
            ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        metrics_plot_path = results_path / "metrics_plot.png"
        plt.savefig(metrics_plot_path)
        plt.close(fig)
        plot_paths.append(str(metrics_plot_path))

    width = int(evaluation["width"])
    split = str(evaluation["celeba_split"]).upper()
    target = evaluation["target"]
    generated = evaluation["generated"]
    few_step_outputs = evaluation["few_step_outputs"]
    step_choices = evaluation["step_choices"]
    num_samples = generated.shape[0]

    fig, axes = plt.subplots(2, num_samples, figsize=(3.0 * num_samples, 6), squeeze=False)
    for idx in range(num_samples):
        plot_rgb_image(axes[0, idx], target[idx], width, f"{split} Target {idx}")
        plot_rgb_image(axes[1, idx], generated[idx], width, f"Generated {idx}")
    plt.suptitle(f"CelebA {split} holdout samples and generated samples from fresh sources")
    plt.tight_layout()
    samples_plot_path = results_path / f"final_generated_data_samples{suffix}.png"
    plt.savefig(samples_plot_path)
    plt.close(fig)
    plot_paths.append(str(samples_plot_path))

    fig, axes = plt.subplots(
        num_samples,
        len(step_choices),
        figsize=(3.0 * len(step_choices), 3.0 * num_samples),
        squeeze=False,
    )
    for sample_idx in range(num_samples):
        for choice_idx, num_steps in enumerate(step_choices):
            plot_rgb_image(
                axes[sample_idx, choice_idx],
                few_step_outputs[str(num_steps)][sample_idx],
                width,
                f"Sample {sample_idx}, Steps {num_steps}",
            )
    plt.suptitle("Generated CelebA samples with different numbers of inference steps")
    plt.tight_layout()
    few_steps_plot_path = results_path / f"generated_data_few_steps{suffix}.png"
    plt.savefig(few_steps_plot_path)
    plt.close(fig)
    plot_paths.append(str(few_steps_plot_path))

    return tuple(plot_paths)


def main():
    parser = argparse.ArgumentParser(description="Plot artifacts from a trained CelebA dual dynamic OT experiment.")
    parser.add_argument("--results-dir", required=True, help="Results directory created by training.")
    parser.add_argument("--evaluation-filename", default="evaluation.pt", help="Evaluation filename under <results-dir>/evaluation.")
    parser.add_argument("--filename-suffix", default="", help="Suffix appended to the plot filenames.")
    parser.add_argument("--skip-metrics", action="store_true", help="Do not re-plot the training metrics.")
    args = parser.parse_args()

    for plot_path in plot_celeba_evaluation(
        args.results_dir,
        evaluation_filename=args.evaluation_filename,
        filename_suffix=args.filename_suffix,
        skip_metrics=args.skip_metrics,
    ):
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
