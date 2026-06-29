"""
Raw AGNIS — experiments/phase2_sequences/summarize_phase2.py

Aggregates Phase 2 experimental outputs across seeds and models to produce a clean markdown summary report.
"""

import os
import json
import csv
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any


def load_all_metrics(results_dir: str) -> List[Dict[str, Any]]:
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


def main():
    results_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "results", "phase2"
    )
    if not os.path.exists(results_dir):
        print(f"[Summarizer] Results directory does not exist: {results_dir}")
        return

    runs = load_all_metrics(results_dir)
    if not runs:
        print("[Summarizer] No metrics.json files found.")
        return

    # Group by condition -> model
    grouped = defaultdict(lambda: defaultdict(list))
    for run in runs:
        cond = run.get("condition")
        model = run.get("model")
        if cond and model:
            grouped[cond][model].append(run)

    csv_path = os.path.join(results_dir, "summary.csv")
    md_path = os.path.join(results_dir, "summary.md")

    csv_headers = [
        "condition",
        "model",
        "seeds",
        "avg_acc_mean", "avg_acc_std",
        "forgetting_mean", "forgetting_std",
        "consistency_mean", "consistency_std",
        "memory_size_mean", "memory_size_std",
        "hit_rate_mean", "hit_rate_std",
        "runtime_mean",
    ]

    csv_rows = []
    md_content = [
        "# Phase 2 Continual Sequence Prediction Sweep Summary\n",
        "Auto-generated summary report of all completed sequence sweeps.\n"
    ]

    for cond in sorted(grouped.keys()):
        md_content.append(f"### Condition: {cond}\n")
        md_content.append(
            "| Model | Seeds | Accuracy (mean±std) | Forgetting (mean±std) | Consistency (mean±std) | Memory Size (mean±std) | Hit Rate | Runtime (s) |"
        )
        md_content.append("|---|---|---|---|---|---|---|---|")

        for model in sorted(grouped[cond].keys()):
            model_runs = grouped[cond][model]
            n_seeds = len(model_runs)

            accs = [r.get("final_average_accuracy", 0.0) for r in model_runs]
            forgettings = [r.get("average_forgetting", 0.0) for r in model_runs]
            consistencies = [r.get("final_temporal_consistency", 0.0) for r in model_runs]
            mem_sizes = [r.get("final_memory_size", 0) for r in model_runs]
            hit_rates = [r.get("memory_hit_rate", 0.0) for r in model_runs]
            runtimes = [r.get("runtime_seconds", 0.0) for r in model_runs]

            acc_mean, acc_std = np.mean(accs), np.std(accs)
            f_mean, f_std = np.mean(forgettings), np.std(forgettings)
            c_mean, c_std = np.mean(consistencies), np.std(consistencies)
            m_mean, m_std = np.mean(mem_sizes), np.std(mem_sizes)
            h_mean, h_std = np.mean(hit_rates), np.std(hit_rates)
            rt_mean = np.mean(runtimes)

            csv_rows.append({
                "condition": cond,
                "model": model,
                "seeds": n_seeds,
                "avg_acc_mean": acc_mean, "avg_acc_std": acc_std,
                "forgetting_mean": f_mean, "forgetting_std": f_std,
                "consistency_mean": c_mean, "consistency_std": c_std,
                "memory_size_mean": m_mean, "memory_size_std": m_std,
                "hit_rate_mean": h_mean, "hit_rate_std": h_std,
                "runtime_mean": rt_mean,
            })

            # Append MD row
            md_content.append(
                f"| `{model}` | {n_seeds} | {acc_mean:.3f}±{acc_std:.3f} | {f_mean:.3f}±{f_std:.3f} | {c_mean:.1%}±{c_std:.1%} | {m_mean:.1f}±{m_std:.1f} | {h_mean:.1%} | {rt_mean:.1f}s |"
            )
        md_content.append("")  # Blank line after each condition

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"[Summarizer] CSV summary saved to {csv_path}")

    # Write MD
    with open(md_path, "w") as f:
        f.write("\n".join(md_content) + "\n")
    print(f"[Summarizer] Markdown summary saved to {md_path}")


if __name__ == "__main__":
    main()
