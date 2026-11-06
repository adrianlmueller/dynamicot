from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..utils import load_torch_artifact


def _plot_image(ax, image_tensor, width: int, title: str):
    ax.imshow(image_tensor.numpy().reshape(width, width), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")


def plot_mnist_upsample_evaluation(
    results_dir: str,
    *,
    evaluation_filename: str = "evaluation.pt",
    evaluation_dir: str | None = None,
    output_dir: str | None = None,
    filename_suffix: str | None = None,
    num_samples: int | None = None,
):
    results_path = Path(results_dir)
    evaluation_path = Path(evaluation_dir) if evaluation_dir is not None else results_path / "evaluation"
    output_path = Path(output_dir) if output_dir is not None else results_path
    output_path.mkdir(parents=True, exist_ok=True)

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

    width = int(evaluation["width"])
    fresh_source = evaluation["fresh_source"]
    target = evaluation["target"]
    generated = evaluation["generated"]
    few_step_outputs = evaluation["few_step_outputs"]
    step_choices = evaluation["step_choices"]

    available_samples = fresh_source.shape[0]
    plot_samples = available_samples if num_samples is None else num_samples
    if plot_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if plot_samples > available_samples:
        raise ValueError(
            f"Requested {plot_samples} samples to plot, but {evaluation_filename} only contains {available_samples}. "
            "Re-run dual_mnist_upsample_eval with a larger --num-samples first."
        )

    fig, axes = plt.subplots(3, plot_samples, figsize=(3.0 * plot_samples, 9), squeeze=False)
    for idx in range(plot_samples):
        _plot_image(axes[0, idx], fresh_source[idx], width, f"Fresh Source {idx}")
        _plot_image(axes[1, idx], target[idx], width, f"Target {idx}")
        _plot_image(axes[2, idx], generated[idx], width, f"Generated {idx}")
    plt.suptitle(f"Fresh source, paired target, and generated test samples (n={plot_samples})")
    plt.tight_layout()
    final_plot_path = output_path / f"final_generated_data_samples{suffix}.png"
    plt.savefig(final_plot_path)
    plt.close()

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
    plt.suptitle("Generated test samples with different numbers of inference steps")
    plt.tight_layout()
    few_steps_plot_path = output_path / f"generated_data_few_steps{suffix}.png"
    plt.savefig(few_steps_plot_path)
    plt.close()

    return str(final_plot_path), str(few_steps_plot_path)


def main():
    parser = argparse.ArgumentParser(description="Plot artifacts from a trained MNIST upsample dual dynamic OT experiment.")
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
        "--num-samples",
        type=int,
        default=None,
        help="Number of saved evaluation samples to plot. Defaults to all samples in the selected evaluation file.",
    )
    args = parser.parse_args()

    final_plot_path, few_steps_plot_path = plot_mnist_upsample_evaluation(
        args.results_dir,
        evaluation_filename=args.evaluation_filename,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        filename_suffix=args.filename_suffix,
        num_samples=args.num_samples,
    )
    print(f"Saved plot: {final_plot_path}")
    print(f"Saved plot: {few_steps_plot_path}")


if __name__ == "__main__":
    main()
