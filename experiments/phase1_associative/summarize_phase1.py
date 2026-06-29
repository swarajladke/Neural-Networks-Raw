"""
Raw AGNIS — experiments/phase1_associative/summarize_phase1.py

Aggregates all JSON results from Phase 1 sweeps, computes seed-level averages and
standard deviations, and writes results to a summary CSV and a readable markdown report.
"""

import os
import json
import csv
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


def load_all_metrics(results_dir: str) -> List[Dict[str, Any]]:
    """Recursively find and load all metrics.json files."""
    all_runs = []
    
    for root, dirs, files in os.walk(results_dir):
        if "metrics.json" in files:
            path = os.path.join(root, "metrics.json")
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                all_runs.append(data)
            except Exception as e:
                print(f"[Summarizer] Error reading {path}: {e}")
                
    return all_runs


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Group runs by condition and model."""
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        cond = run.get("condition")
        model = run.get("model")
        if cond and model:
            grouped[cond][model].append(run)
    return grouped


def main():
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "phase1"
    )
    if not os.path.exists(results_dir):
        print(f"[Summarizer] Results directory does not exist: {results_dir}")
        return

    runs = load_all_metrics(results_dir)
    if not runs:
        print("[Summarizer] No metrics.json files found.")
        return

    grouped = aggregate_runs(runs)

    # Output paths
    csv_path = os.path.join(results_dir, "summary.csv")
    md_path = os.path.join(results_dir, "summary.md")

    # CSV headers
    headers = [
        "condition",
        "model",
        "num_seeds",
        "final_accuracy_mean", "final_accuracy_std",
        "forgetting_mean", "forgetting_std",
        "bwt_mean", "bwt_std",
        "fwt_mean", "fwt_std",
        "prediction_error_mean", "prediction_error_std",
        "active_fraction_mean", "active_fraction_std",
        "memory_size_mean", "memory_size_std",
        "memory_hit_rate_mean", "memory_hit_rate_std",
        "overlap_mean", "overlap_std",
        "interference_mean", "interference_std",
        "runtime_mean", "runtime_std",
    ]

    summary_rows = []
    
    # Markdown tables group by condition
    md_tables = {}

    for cond in sorted(grouped.keys()):
        cond_rows = []
        
        # Prepare MD header
        md_table = [
            f"\n### Condition: {cond}\n",
            "| Model | Seeds | Accuracy (mean±std) | Forgetting (mean±std) | BWT (mean±std) | Overlap (mean±std) | Hit Rate | Runtime (s) |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for model in sorted(grouped[cond].keys()):
            model_runs = grouped[cond][model]
            n_seeds = len(model_runs)

            # Extract lists of values
            accs = [r.get("final_average_accuracy", 0.0) for r in model_runs]
            forgettings = [r.get("average_forgetting", 0.0) for r in model_runs]
            bwts = [r.get("backward_transfer", 0.0) for r in model_runs]
            fwts = [r.get("forward_transfer", 0.0) for r in model_runs]
            errors = [r.get("mean_prediction_error", 0.0) for r in model_runs]
            actives = [r.get("mean_active_fraction", 0.0) for r in model_runs]
            mem_sizes = [r.get("final_memory_size", 0) for r in model_runs]
            hit_rates = [r.get("memory_hit_rate", 0.0) for r in model_runs]
            overlaps = [r.get("representation_overlap_mean", 0.0) for r in model_runs]
            interferences = [r.get("interference_score", 0.0) for r in model_runs]
            runtimes = [r.get("runtime_seconds", 0.0) for r in model_runs]

            # Compute stats
            row_data = {
                "condition": cond,
                "model": model,
                "num_seeds": n_seeds,
                "final_accuracy_mean": np.mean(accs), "final_accuracy_std": np.std(accs),
                "forgetting_mean": np.mean(forgettings), "forgetting_std": np.std(forgettings),
                "bwt_mean": np.mean(bwts), "bwt_std": np.std(bwts),
                "fwt_mean": np.mean(fwts), "fwt_std": np.std(fwts),
                "prediction_error_mean": np.mean(errors), "prediction_error_std": np.std(errors),
                "active_fraction_mean": np.mean(actives), "active_fraction_std": np.std(actives),
                "memory_size_mean": np.mean(mem_sizes), "memory_size_std": np.std(mem_sizes),
                "memory_hit_rate_mean": np.mean(hit_rates), "memory_hit_rate_std": np.std(hit_rates),
                "overlap_mean": np.mean(overlaps), "overlap_std": np.std(overlaps),
                "interference_mean": np.mean(interferences), "interference_std": np.std(interferences),
                "runtime_mean": np.mean(runtimes), "runtime_std": np.std(runtimes),
            }
            summary_rows.append(row_data)

            # Append MD row
            md_table.append(
                f"| `{model}` | {n_seeds} | {np.mean(accs):.3f}±{np.std(accs):.3f} | {np.mean(forgettings):.3f}±{np.std(forgettings):.3f} | {np.mean(bwts):.3f}±{np.std(bwts):.3f} | {np.mean(overlaps):.3f}±{np.std(overlaps):.3f} | {np.mean(hit_rates):.1%} | {np.mean(runtimes):.1f}s |"
            )

        md_tables[cond] = "\n".join(md_table)

    # Write summary CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[Summarizer] CSV summary saved to {csv_path}")

    # Write summary MD
    with open(md_path, "w") as f:
        f.write("# Phase 1 Continual Learning Benchmark Summary\n")
        f.write("Auto-generated summary report of all completed sweeps.\n")
        for cond, table in md_tables.items():
            f.write(table + "\n")
    print(f"[Summarizer] Markdown report saved to {md_path}")


if __name__ == "__main__":
    main()
