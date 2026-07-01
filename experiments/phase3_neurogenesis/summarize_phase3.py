"""
Raw AGNIS — experiments/phase3_neurogenesis/summarize_phase3.py

Aggregates Phase 3 neurogenesis seeds results, calculates stats, and creates summary reports.
"""

import os
import sys
import json
import csv
import numpy as np
from collections import defaultdict

# Add directory paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def summarize_phase3_results(results_dir: str):
    """Aggregate all seed files under results/phase3/ and print summaries."""
    print(f"[Summarizer] Scanning results folder: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"[Summarizer] Error: directory {results_dir} does not exist.")
        return

    # Structure: condition -> model -> list of metrics dictionaries
    raw_data = defaultdict(lambda: defaultdict(list))
    
    # Traverse folder
    for root, dirs, files in os.walk(results_dir):
        if "metrics.json" in files:
            path = os.path.join(root, "metrics.json")
            try:
                with open(path, "r") as f:
                    metrics = json.load(f)
                cond = metrics["condition"]
                model = metrics["model"]
                raw_data[cond][model].append(metrics)
            except Exception as e:
                print(f"[Summarizer] Error reading {path}: {e}")

    if not raw_data:
        print("[Summarizer] No valid metrics.json files found.")
        return

    # Output files
    md_path = os.path.join(results_dir, "summary.md")
    csv_path = os.path.join(results_dir, "summary.csv")

    csv_rows = [
        [
            "Condition",
            "Model",
            "Seeds",
            "Accuracy_Mean",
            "Accuracy_Std",
            "Forgetting_Mean",
            "Forgetting_Std",
            "Final_Dim_Mean",
            "Final_Dim_Std",
            "Births_Mean",
            "Prunes_Mean",
            "Replay_Benefit_Mean",
            "Consistency_Mean",
            "Runtime_Mean",
        ]
    ]

    md_report = "# Phase 3 Autonomous Neurogenesis Sweep Summary\n\n"
    md_report += "Auto-generated summary report of all completed neurogenesis sweeps.\n\n"

    for cond in sorted(raw_data.keys()):
        md_report += f"### Condition: {cond}\n\n"
        md_report += (
            "| Model | Seeds | Accuracy (mean±std) | Forgetting (mean±std) | "
            "Final Dim (mean±std) | Births | Prunes | Replay Benefit | Consistency | Runtime |\n"
        )
        md_report += "|---|---|---|---|---|---|---|---|---|---|\n"

        for model in sorted(raw_data[cond].keys()):
            runs = raw_data[cond][model]
            seeds_count = len(runs)
            
            accs = [r["final_average_accuracy"] for r in runs]
            forgs = [r["average_forgetting"] for r in runs]
            dims = [r["final_latent_dim"] for r in runs]
            births = [r["birth_events"] for r in runs]
            prunes = [r["prune_events"] for r in runs]
            benefits = [r.get("replay_benefit", 0.0) for r in runs]
            consistencies = [r.get("final_temporal_consistency", 0.0) for r in runs]
            runtimes = [r["runtime_seconds"] for r in runs]

            acc_mean, acc_std = np.mean(accs), np.std(accs)
            forg_mean, forg_std = np.mean(forgs), np.std(forgs)
            dim_mean, dim_std = np.mean(dims), np.std(dims)
            birth_mean = np.mean(births)
            prune_mean = np.mean(prunes)
            benefit_mean = np.mean(benefits)
            cons_mean = np.mean(consistencies)
            runtime_mean = np.mean(runtimes)

            md_report += (
                f"| `{model}` | {seeds_count} | {acc_mean:.3f}±{acc_std:.3f} | {forg_mean:.3f}±{forg_std:.3f} | "
                f"{dim_mean:.1f}±{dim_std:.1f} | {birth_mean:.1f} | {prune_mean:.1f} | "
                f"{benefit_mean*100:+.1f}% | {cons_mean*100:.1f}% | {runtime_mean:.1f}s |\n"
            )

            csv_rows.append(
                [
                    cond,
                    model,
                    seeds_count,
                    f"{acc_mean:.4f}",
                    f"{acc_std:.4f}",
                    f"{forg_mean:.4f}",
                    f"{forg_std:.4f}",
                    f"{dim_mean:.2f}",
                    f"{dim_std:.2f}",
                    f"{birth_mean:.2f}",
                    f"{prune_mean:.2f}",
                    f"{benefit_mean:.4f}",
                    f"{cons_mean:.4f}",
                    f"{runtime_mean:.2f}",
                ]
            )
        md_report += "\n"

    # Save summary report files
    with open(md_path, "w") as f:
        f.write(md_report)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"[Summarizer] Markdown summary saved to {md_path}")
    print(f"[Summarizer] CSV summary saved to {csv_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results", "phase3")
    summarize_phase3_results(results_dir)
