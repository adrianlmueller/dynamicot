from __future__ import annotations

import argparse

from ..duals.image_trainer import train_emnist_mnist_dual_model
from ..utils import load_yaml_config
from .dual_emnist_mnist_eval import evaluate_emnist_mnist_checkpoint
from .dual_emnist_mnist_plot import plot_emnist_mnist_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train the EMNIST-source to MNIST dual dynamic OT experiment.")
    parser.add_argument(
        "--config",
        default="configs/dual_emnist_mnist.yaml",
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=30,
        help="Number of fresh EMNIST test samples to evaluate and plot after training.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    artifacts = train_emnist_mnist_dual_model(config)
    print(f"Training completed. Results dir: {artifacts.results_dir}")
    print(f"Stored {len(artifacts.checkpoint_paths)} checkpoint files.")
    print(f"Final checkpoint: {artifacts.final_checkpoint_path}")
    if artifacts.ema_checkpoint_path is not None:
        print(f"Final EMA checkpoint: {artifacts.ema_checkpoint_path}")
    print(f"Metrics file: {artifacts.metrics_path}")

    evaluation_path = evaluate_emnist_mnist_checkpoint(
        artifacts.results_dir,
        num_samples=args.num_eval_samples,
    )
    print(f"Saved evaluation artifact: {evaluation_path}")
    for plot_path in plot_emnist_mnist_evaluation(artifacts.results_dir):
        print(f"Saved plot: {plot_path}")
    if artifacts.ema_checkpoint_path is not None:
        ema_evaluation_path = evaluate_emnist_mnist_checkpoint(
            artifacts.results_dir,
            checkpoint_filename="checkpoint_final_ema.pt",
            evaluation_filename="evaluation_ema.pt",
            num_samples=args.num_eval_samples,
        )
        print(f"Saved EMA evaluation artifact: {ema_evaluation_path}")
        for plot_path in plot_emnist_mnist_evaluation(
            artifacts.results_dir,
            evaluation_filename="evaluation_ema.pt",
            filename_suffix="ema",
            skip_metrics=True,
        ):
            print(f"Saved EMA plot: {plot_path}")


if __name__ == "__main__":
    main()
