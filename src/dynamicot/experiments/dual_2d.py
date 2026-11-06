from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from ..duals.trainer import train_dual_model
from ..utils import create_results_dir, load_yaml_config, save_experiment_summary
from .dual_2d_eval import evaluate_2d_checkpoint


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict]):
    """Mean and standard deviation over repetitions for every (dataset, coupling, H_test)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["initialization_type"], row["h_test"]), []).append(row)
    summary = []
    for (dataset, initialization_type, h_test), group in groups.items():
        w2 = np.array([row["w2_generated_target"] for row in group])
        energy = np.array([row["path_energy"] for row in group])
        summary.append(
            {
                "dataset": dataset,
                "initialization_type": initialization_type,
                "h_test": h_test,
                "num_repetitions": len(group),
                "w2_mean": w2.mean(),
                "w2_std": w2.std(),
                "path_energy_mean": energy.mean(),
                "path_energy_std": energy.std(),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the 2D experiments (Table 1/2).")
    parser.add_argument("--config", default="configs/dual_2d_table.yaml", help="Path to a YAML config.")
    parser.add_argument("--device", help="Override config device (cpu, cuda, cuda:0, mps, gpu, auto).")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.device:
        config["device"] = args.device

    table_dir = Path(create_results_dir(config))
    save_experiment_summary(str(table_dir), config)
    table_cfg = config["table"]
    base_seed = int(config["seed"])

    rows = []
    for dataset in table_cfg["datasets"]:
        for initialization_type in table_cfg["initialization_types"]:
            for repetition in range(int(table_cfg["num_repetitions"])):
                run_config = copy.deepcopy(config)
                del run_config["table"]
                run_config["seed"] = base_seed + repetition
                run_config["data"]["name"] = dataset
                run_config["training"]["initialization_type"] = initialization_type

                run_dir = table_dir / f"{dataset}_{initialization_type}_rep{repetition}"
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n=== {dataset}, {initialization_type}, repetition {repetition} -> {run_dir}")
                train_dual_model(run_config, results_dir=str(run_dir))
                for result in evaluate_2d_checkpoint(str(run_dir)):
                    rows.append(
                        {
                            "dataset": dataset,
                            "initialization_type": initialization_type,
                            "repetition": repetition,
                            **result,
                        }
                    )

    summary = _summarize(rows)
    _write_csv(table_dir / "results_raw.csv", rows)
    _write_csv(table_dir / "results_summary.csv", summary)

    print("\ndataset, coupling, H_test: W2 (mean +- std) | path energy (mean +- std)")
    for row in summary:
        print(
            f"{row['dataset']}, {row['initialization_type']}, {row['h_test']}: "
            f"{row['w2_mean']:.3f} +- {row['w2_std']:.3f} | "
            f"{row['path_energy_mean']:.3f} +- {row['path_energy_std']:.3f}"
        )
    print(f"\nSaved results to {table_dir}")


if __name__ == "__main__":
    main()
