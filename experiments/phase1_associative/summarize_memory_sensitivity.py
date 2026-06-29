"""
Raw AGNIS — experiments/phase1_associative/summarize_memory_sensitivity.py

Aggregates all JSON results from the memory sensitivity sweeps, compiles statistics,
and generates the gate-sensitivity comparison tables markdown report.
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


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[float, List[Dict[str, Any]]]]]:
    """Group runs by condition, model, and write_error_threshold."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in runs:
        cond = run.get("condition")
        model = run.get("model")
        w_thresh = run.get("write_error_threshold")
        if cond and model and w_thresh is not None:
            grouped[cond][model][w_thresh].append(run)
    return grouped


def main():
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "phase1_memory_sensitivity"
    )
    if not os.path.exists(results_dir):
        print(f"[Summarizer] Results directory does not exist: {results_dir}")
        return

    runs = load_all_metrics(results_dir)
    if not runs:
        print("[Summarizer] No metrics.json files found.")
        return

    grouped = aggregate_runs(runs)

    csv_path = os.path.join(results_dir, "sensitivity_summary.csv")
    md_path = os.path.join(results_dir, "sensitivity_summary.md")

    # CSV headers
    headers = [
        "condition",
        "model",
        "write_threshold",
        "num_seeds",
        "memory_writes_mean", "memory_writes_std",
        "hit_rate_mean", "hit_rate_std",
        "avg_acc_mean", "avg_acc_std",
        "forgetting_mean", "forgetting_std",
        "replay_benefit_mean", "replay_benefit_std",
    ]

    csv_rows = []
    md_content = [
        "# Raw AGNIS v0.2b — Memory & Replay Sensitivity Sweep Summary\n",
        "Aggregated results showing the effects of different surprise thresholds on memory writes, hit rates, accuracy, and replay benefits.\n"
    ]

    for cond in sorted(grouped.keys()):
        for model in sorted(grouped[cond].keys()):
            md_content.append(f"\n### Condition: {cond} | Model: {model}\n")
            md_content.append(
                "| Threshold | Memory writes (mean±std) | Hit rate (mean±std) | Avg acc (mean±std) | Forgetting (mean±std) | Replay benefit (mean±std) |"
            )
            md_content.append("|---|---|---|---|---|---|")

            thresholds = sorted(grouped[cond][model].keys(), reverse=True)
            for w_thresh in thresholds:
                model_runs = grouped[cond][model][w_thresh]
                n_seeds = len(model_runs)

                writes = [r.get("final_memory_size", 0) for r in model_runs]
                hit_rates = [r.get("memory_hit_rate", 0.0) for r in model_runs]
                accs = [r.get("final_average_accuracy", 0.0) for r in model_runs]
                forgettings = [r.get("average_forgetting", 0.0) for r in model_runs]
                benefits = [r.get("replay_benefit", 0.0) for r in model_runs]

                # Stats
                w_mean, w_std = np.mean(writes), np.std(writes)
                h_mean, h_std = np.mean(hit_rates), np.std(hit_rates)
                a_mean, a_std = np.mean(accs), np.std(accs)
                f_mean, f_std = np.mean(forgettings), np.std(forgettings)
                b_mean, b_std = np.mean(benefits), np.std(benefits)

                csv_rows.append({
                    "condition": cond,
                    "model": model,
                    "write_threshold": w_thresh,
                    "num_seeds": n_seeds,
                    "memory_writes_mean": w_mean, "memory_writes_std": w_std,
                    "hit_rate_mean": h_mean, "hit_rate_std": h_std,
                    "avg_acc_mean": a_mean, "avg_acc_std": a_std,
                    "forgetting_mean": f_mean, "forgetting_std": f_std,
                    "replay_benefit_mean": b_mean, "replay_benefit_std": b_std,
                })

                # Append MD row
                md_content.append(
                    f"| {w_thresh:.2f} | {w_mean:.1f}±{w_std:.1f} | {h_mean:.1%}±{h_std:.1%} | {a_mean:.3f}±{a_std:.3f} | {f_mean:.3f}±{f_std:.3f} | {b_mean:+.3f}±{b_std:.3f} |"
                )

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"[Summarizer] CSV sensitivity summary saved to {csv_path}")

    # Write MD
    with open(md_path, "w") as f:
        f.write("\n".join(md_content) + "\n")
    print(f"[Summarizer] Markdown sensitivity report saved to {md_path}")


if __name__ == "__main__":
    main()
