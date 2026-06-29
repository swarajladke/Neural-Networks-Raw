"""
Raw AGNIS — src/agnis/evaluation/phase1_logging.py

Handles saving run metrics, task accuracy matrices, forgetting data, active fraction tracking,
and representation overlaps to JSON and CSV formats under structured results folders.
"""

import os
import json
import csv
import yaml
from typing import Dict, Any, List, Optional


def format_threshold(value: float) -> str:
    return f"{value:.4g}".replace(".", "p")


def save_phase1_run_results(
    results_dir: str,
    condition: str,
    model_name: str,
    seed: int,
    metrics_dict: Dict[str, Any],
    accuracy_matrix: List[List[Optional[float]]],
    forgetting_list: List[float],
    prediction_errors: List[float],
    active_fractions: List[float],
    memory_usages: List[int],
    overlap_matrix: List[List[float]],
    config_data: Dict[str, Any],
) -> str:
    """
    Save all experiment outputs to nested directory structure.

    Parameters
    ----------
    results_dir : str
        Base results directory (e.g. results/phase1/)
    condition : str
        Condition name (orthogonal, overlapping, clustered, capacity_stress)
    model_name : str
        Model name
    seed : int
        Random seed
    metrics_dict : dict
        Aggregated summary metrics
    accuracy_matrix : list of list
        Checkpoint-by-checkpoint task accuracies
    forgetting_list : list of float
        Per-task forgetting scores
    prediction_errors : list of float
        Per-step training prediction errors
    active_fractions : list of float
        Per-step active unit fraction
    memory_usages : list of int
        Per-step memory sizes
    overlap_matrix : list of list of float
        Pairwise representation cosine similarities
    config_data : dict
        Full configuration parameters used

    Returns
    -------
    str
        Path to the saved run folder.
    """
    # Create seed-specific directory path
    sensitivity_mode = metrics_dict.get("sensitivity_mode", False)
    write_thresh = metrics_dict.get("write_error_threshold", 0.2)
    novelty_thresh = metrics_dict.get("write_novelty_threshold", 0.15)

    if sensitivity_mode:
        w_str = format_threshold(write_thresh)
        n_str = format_threshold(novelty_thresh)
        run_dir = os.path.join(
            results_dir,
            condition,
            model_name,
            f"write_{w_str}_novelty_{n_str}",
            f"seed_{seed}"
        )
    else:
        run_dir = os.path.join(results_dir, condition, model_name, f"seed_{seed}")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    # 1. Save metrics.json
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # 2. Save config_used.yaml
    config_path = os.path.join(run_dir, "config_used.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    # 3. Save accuracy_matrix.csv
    acc_path = os.path.join(run_dir, "accuracy_matrix.csv")
    with open(acc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Task_{i}" for i in range(len(accuracy_matrix[0]))])
        for row in accuracy_matrix:
            writer.writerow([v if v is not None else "" for v in row])

    # 4. Save forgetting.csv
    forgetting_path = os.path.join(run_dir, "forgetting.csv")
    with open(forgetting_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Forgetting"])
        for i, val in enumerate(forgetting_list):
            writer.writerow([f"Task_{i}", val])

    # 5. Save prediction_error.csv
    error_path = os.path.join(run_dir, "prediction_error.csv")
    with open(error_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "PredictionError"])
        for i, val in enumerate(prediction_errors):
            writer.writerow([i, val])

    # 6. Save active_fraction.csv
    active_path = os.path.join(run_dir, "active_fraction.csv")
    with open(active_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "ActiveFraction"])
        for i, val in enumerate(active_fractions):
            writer.writerow([i, val])

    # 7. Save memory_usage.csv
    memory_path = os.path.join(run_dir, "memory_usage.csv")
    with open(memory_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "MemorySize"])
        for i, val in enumerate(memory_usages):
            writer.writerow([i, val])

    # 8. Save representation_overlap.csv
    overlap_path = os.path.join(run_dir, "representation_overlap.csv")
    with open(overlap_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Task_{i}" for i in range(len(overlap_matrix))])
        for row in overlap_matrix:
            writer.writerow(row)

    return run_dir
