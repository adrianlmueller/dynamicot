from __future__ import annotations

import argparse
import datetime as dt
import math
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final_celeba_plots.yaml"


def _save_square_grid(samples, *, width: int, output_path: Path, dpi: int):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dynamicot.experiments.dual_celeba_plot import plot_rgb_image

    side = math.isqrt(samples.shape[0])
    fig, axes = plt.subplots(side, side, figsize=(3.0 * side, 3.0 * side), squeeze=False)
    for index, ax in enumerate(axes.flat):
        plot_rgb_image(ax, samples[index], width)
    fig.tight_layout()
    fig.subplots_adjust(wspace=fig.subplotpars.hspace)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Square grids of generated, holdout and few-step CelebA samples."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to the final CelebA plotting YAML config.")
    args = parser.parse_args()

    import numpy as np
    import torch

    from dynamicot.duals.inference import generate_from_source, generate_from_source_few_steps
    from dynamicot.experiments.dual_celeba_eval import load_celeba_evaluation_model
    from dynamicot.utils import load_yaml_config

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml_config(config_path)
    checkpoint_path = Path(str(config["checkpoint_path"])).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    training_results_dir = checkpoint_path.parent.parent

    num_samples = int(config["num_samples"])
    if math.isqrt(num_samples) ** 2 != num_samples:
        raise ValueError("num_samples must be a perfect square")
    few_step_steps = [int(step) for step in config["few_step_steps"]]
    output_cfg = config.get("output", {})
    dpi = int(output_cfg.get("dpi", 100))

    training_config, model, fresh_source, target_data, metadata = load_celeba_evaluation_model(
        str(training_results_dir),
        num_samples=num_samples,
        split=str(config.get("split", "valid")),
        checkpoint_path=checkpoint_path,
    )
    horizon = int(training_config["training"]["horizon"])
    primal_cfg = training_config["training"]["primal"]
    inference_noise_scale = float(primal_cfg["noise_scale"]) / np.sqrt(2.0 * float(primal_cfg["lr"]))
    width = int(metadata["width"])

    base_dir = Path(output_cfg.get("base_dir", "results/final_celeba"))
    if not base_dir.is_absolute():
        base_dir = REPO_ROOT / base_dir
    output_dir = base_dir / f"results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / config_path.name)

    generated = generate_from_source(
        model, fresh_source, horizon, noise_scale=inference_noise_scale, clamp=(0.0, 1.0)
    )[-1].detach().cpu()
    _save_square_grid(generated, width=width, output_path=output_dir / "generated_samples.png", dpi=dpi)
    _save_square_grid(target_data.cpu(), width=width, output_path=output_dir / "dataset_samples.png", dpi=dpi)

    for num_steps in few_step_steps:
        samples = torch.cat(
            [
                generate_from_source_few_steps(
                    model,
                    fresh_source[idx : idx + 1],
                    horizon,
                    num_steps=num_steps,
                    noise_scale=inference_noise_scale,
                    clamp=(0.0, 1.0),
                )[-1].detach().cpu()
                for idx in range(num_samples)
            ],
            dim=0,
        )
        _save_square_grid(samples, width=width, output_path=output_dir / f"few_step_{num_steps}.png", dpi=dpi)

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
