from __future__ import annotations

import argparse

from ..duals.image_trainer import train_mnist_upsample_dual_model
from ..utils import load_yaml_config
from .dual_mnist_upsample_eval import evaluate_mnist_upsample_checkpoint
from .dual_mnist_upsample_plot import plot_mnist_upsample_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train the MNIST upsample dual dynamic OT experiment.")
    parser.add_argument(
        "--config",
        default="configs/dual_mnist_upsample_6.yaml",
        help="Path to a YAML experiment config.",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=30,
        help="Number of fresh MNIST test samples to evaluate and plot after training.",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)

    artifacts = train_mnist_upsample_dual_model(config)
    print(f"Training completed. Results dir: {artifacts.results_dir}")
    print(f"Stored {len(artifacts.checkpoint_paths)} checkpoint files.")
    print(f"Final checkpoint: {artifacts.final_checkpoint_path}")
    if artifacts.ema_checkpoint_path is not None:
        print(f"Final EMA checkpoint: {artifacts.ema_checkpoint_path}")
    print(f"Metrics file: {artifacts.metrics_path}")

    evaluation_path = evaluate_mnist_upsample_checkpoint(
        artifacts.results_dir,
        num_samples=args.num_eval_samples,
    )
    print(f"Saved evaluation artifact: {evaluation_path}")
    for plot_path in plot_mnist_upsample_evaluation(artifacts.results_dir):
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
