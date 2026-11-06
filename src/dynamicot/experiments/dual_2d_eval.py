from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..data import build_2d_dataset
from ..duals.inference import generate_from_source_few_steps
from ..duals.objectives import cost_function
from ..duals.trainer import build_model
from ..utils import load_checkpoint, load_experiment_summary, resolve_device, resolve_dtype, save_json


EVALUATION_SEED_OFFSET = 1000


def squared_w2(x: torch.Tensor, y: torch.Tensor):
    """Squared Wasserstein-2 distance between two equally sized point clouds (exact
    assignment via the Hungarian algorithm), without the 1/2 factor."""
    from scipy.optimize import linear_sum_assignment

    cost = torch.cdist(x.detach().cpu().float(), y.detach().cpu().float()).square()
    row_ind, col_ind = linear_sum_assignment(cost.numpy())
    return float(cost[row_ind, col_ind].sum().item() / x.shape[0])


def evaluate_2d_checkpoint(results_dir: str, *, checkpoint_filename: str = "checkpoint_final.pt"):
    """Generate from fresh source samples with H_test steps and report the squared W2
    distance to fresh target samples and the path energy (Table 1/2)."""
    config = load_experiment_summary(results_dir)
    eval_cfg = config["evaluation"]
    device = resolve_device(config.get("device", "cpu"))
    dtype = resolve_dtype(config.get("dtype", "float32"))
    horizon = int(config["training"]["horizon"])

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(str(Path(results_dir) / "checkpoints" / checkpoint_filename), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    source, target = build_2d_dataset(
        config["data"]["name"],
        n_samples=int(eval_cfg["n_samples"]),
        seed=int(config["seed"]) + EVALUATION_SEED_OFFSET,
        device=device,
        dtype=dtype,
    )

    results = []
    for h_test in eval_cfg["h_test"]:
        h_test = int(h_test)
        path = generate_from_source_few_steps(model, source, horizon, num_steps=h_test)
        path_energy = h_test * sum(
            (2 * cost_function(path[idx], path[idx + 1])).mean().item() for idx in range(h_test)
        )
        results.append(
            {
                "h_test": h_test,
                "w2_generated_target": squared_w2(path[h_test], target),
                "path_energy": path_energy,
            }
        )

    save_json(results_dir, "evaluation.json", {"checkpoint_filename": checkpoint_filename, "results": results})
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D dual dynamic OT run.")
    parser.add_argument("--results-dir", required=True, help="Run directory created by dual_2d.")
    parser.add_argument("--checkpoint-filename", default="checkpoint_final.pt")
    args = parser.parse_args()

    for row in evaluate_2d_checkpoint(args.results_dir, checkpoint_filename=args.checkpoint_filename):
        print(row)


if __name__ == "__main__":
    main()
