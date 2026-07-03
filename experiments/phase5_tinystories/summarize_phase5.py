"""
Raw AGNIS — experiments/phase5_tinystories/summarize_phase5.py

Aggregates Phase 5 TinyStories seeds results, calculates stats, and creates summary reports.
"""

import os
import sys
import json
import csv
import numpy as np
from collections import defaultdict

# Add directory paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def summarize_phase5_results(results_dir: str):
    """Aggregate all seed files under results/phase5/ and print summaries."""
    print(f"[Summarizer] Scanning results folder: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"[Summarizer] Error: directory {results_dir} does not exist.")
        return

    # Structure: model -> list of metrics dictionaries
    raw_data = defaultdict(list)
    
    # Traverse folder
    for root, dirs, files in os.walk(results_dir):
        if "metrics.json" in files:
            path = os.path.join(root, "metrics.json")
            try:
                with open(path, "r") as f:
                    metrics = json.load(f)
                model = metrics["model"]
                raw_data[model].append(metrics)
            except Exception as e:
                print(f"[Summarizer] Error reading {path}: {e}")

    if not raw_data:
        print("[Summarizer] No valid metrics.json files found.")
        return

    # Output files
    md_path = os.path.join(results_dir, "summary.md")
    csv_path = os.path.join(results_dir, "summary.csv")

    csv_headers = [
        "Model",
        "Seeds",
        "Cont_Accuracy_Mean",
        "Cont_Accuracy_Std",
        "Cont_BPC_Mean",
        "Cont_BPC_Std",
        "Cont_Acc_Forgetting_Mean",
        "Cont_Acc_Forgetting_Std",
        "Cont_BPC_Forgetting_Mean",
        "Cont_BPC_Forgetting_Std",
        "Repetition_Rate_Mean",
        "Keyword_Retention_Mean",
        "Name_Consistency_Mean",
        "Sentence_Completion_Mean",
        "Distinct_2_Mean",
        "Distinct_3_Mean",
        "Final_Dim_Mean",
        "Final_Dim_Std",
        "Births_Mean",
        "Prunes_Mean",
        "Runtime_Mean"
    ]

    csv_rows = [csv_headers]

    md_report = "# Phase 5 TinyStories Mini Sweep Summary\n\n"
    md_report += "Auto-generated summary report of all completed story continuation sweeps.\n\n"
    
    md_report += (
        "| Model | Seeds | Cont Acc (mean±std) | Cont BPC (mean±std) | Acc Forg | BPC Forg | "
        "Rep Rate | Key Ret | Name Cons | Sent Comp | Dist-2 | Dist-3 | Final Dim (mean±std) | Births | Prunes | Runtime |\n"
    )
    md_report += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

    for model in sorted(raw_data.keys()):
        runs = raw_data[model]
        seeds_count = len(runs)
        
        accs = [r["final_average_continuation_accuracy"] for r in runs]
        bpcs = [r["final_average_continuation_bpc"] for r in runs]
        acc_forgs = [r["continuation_accuracy_forgetting"] for r in runs]
        bpc_forgs = [r["continuation_bpc_forgetting"] for r in runs]
        
        rep_rates = [r.get("average_repetition_rate", 0.0) for r in runs]
        key_ret = [r.get("average_keyword_retention", 0.0) for r in runs]
        name_cons = [r.get("average_name_consistency", 0.0) for r in runs]
        sent_comp = [r.get("average_sentence_completion_rate", 0.0) for r in runs]
        dist2 = [r.get("average_distinct_2", 0.0) for r in runs]
        dist3 = [r.get("average_distinct_3", 0.0) for r in runs]
        
        dims = [r["final_d_z"] for r in runs]
        births = [r["total_units_born"] for r in runs]
        prunes = [r["total_units_pruned"] for r in runs]
        runtimes = [r["runtime_seconds"] for r in runs]

        acc_mean, acc_std = np.mean(accs), np.std(accs)
        bpc_mean, bpc_std = np.mean(bpcs), np.std(bpcs)
        acc_forg_mean, acc_forg_std = np.mean(acc_forgs), np.std(acc_forgs)
        bpc_forg_mean, bpc_forg_std = np.mean(bpc_forgs), np.std(bpc_forgs)
        
        rep_mean = np.mean(rep_rates)
        key_mean = np.mean(key_ret)
        name_mean = np.mean(name_cons)
        sent_mean = np.mean(sent_comp)
        dist2_mean = np.mean(dist2)
        dist3_mean = np.mean(dist3)
        
        dim_mean, dim_std = np.mean(dims), np.std(dims)
        birth_mean = np.mean(births)
        prune_mean = np.mean(prunes)
        runtime_mean = np.mean(runtimes)

        # Append to CSV
        csv_rows.append([
            model,
            seeds_count,
            acc_mean, acc_std,
            bpc_mean, bpc_std,
            acc_forg_mean, acc_forg_std,
            bpc_forg_mean, bpc_forg_std,
            rep_mean,
            key_mean,
            name_mean,
            sent_mean,
            dist2_mean,
            dist3_mean,
            dim_mean, dim_std,
            birth_mean,
            prune_mean,
            runtime_mean
        ])

        # Append to MD
        md_report += (
            f"| `{model}` | {seeds_count} | "
            f"{acc_mean:.3f}±{acc_std:.3f} | {bpc_mean:.3f}±{bpc_std:.3f} | "
            f"{acc_forg_mean:.3f}±{acc_forg_std:.3f} | {bpc_forg_mean:.3f}±{bpc_forg_std:.3f} | "
            f"{rep_mean:.1%} | {key_mean:.1%} | {name_mean:.1%} | {sent_mean:.1%} | "
            f"{dist2_mean:.1%} | {dist3_mean:.1%} | "
            f"{dim_mean:.1f}±{dim_std:.1f} | {birth_mean:.1f} | {prune_mean:.1f} | {runtime_mean:.1f}s |\n"
        )

    # Write Markdown
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"[Summarizer] Markdown summary saved to {md_path}")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"[Summarizer] CSV summary saved to {csv_path}")


if __name__ == "__main__":
    results_folder = "results/phase5"
    if len(sys.argv) > 1:
        results_folder = sys.argv[1]
    summarize_phase5_results(results_folder)
