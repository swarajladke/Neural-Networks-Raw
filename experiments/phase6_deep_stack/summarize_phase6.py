"""
Raw AGNIS — experiments/phase6_deep_stack/summarize_phase6.py

Aggregates Phase 6 Deep AGNIS sweep results, calculates stats, and creates summary reports.
"""

import os
import sys
import json
import csv
import numpy as np
from collections import defaultdict

# Add directory paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agnis.text.char_metrics import compute_retained_accuracy, compute_peak_accuracy, compute_forward_transfer


def summarize_phase6_results(results_dir: str):
    """Aggregate all seed files under results/phase6/ and print summaries."""
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
        "Peak_Accuracy_Mean",
        "Peak_Accuracy_Std",
        "Retained_Accuracy_Mean",
        "Retained_Accuracy_Std",
        "Forward_Transfer_Mean",
        "Forward_Transfer_Std",
        "Cont_Acc_Forgetting_Mean",
        "Cont_Acc_Forgetting_Std",
        "Cont_BPC_Forgetting_Mean",
        "Cont_BPC_Forgetting_Std",
        "Probe_Acc_Layer0",
        "Probe_Acc_Layer1",
        "Probe_Acc_Layer2",
        "Repetition_Rate_Mean",
        "Distinct_2_Mean",
        "Distinct_3_Mean",
        "Final_Dim_Mean",
        "Births_Mean",
        "Prunes_Mean",
        "Runtime_Mean"
    ]

    csv_rows = [csv_headers]

    md_report = "# Phase 6 Deep Hierarchical AGNIS Sweep Summary\n\n"
    md_report += "Auto-generated summary report of all completed deep hierarchical sweep configurations.\n\n"
    
    md_report += (
        "| Model | Seeds | Cont Acc | Peak Acc | Retained Acc | FWT | Acc Forg | BPC Forg | "
        "Probe L0 | Probe L1 | Probe L2 | Rep Rate | Dist-2 | Dist-3 | Final Dims | Births | Prunes | Runtime |\n"
    )
    md_report += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"

    for model in sorted(raw_data.keys()):
        runs = raw_data[model]
        seeds_count = len(runs)
        
        accs = [r["average_accuracy_after"] for r in runs]
        bpcs = [r["average_bpc_after"] for r in runs]
        
        # Calculate mean forgetting
        acc_forgs = [sum(r["forgetting"]) / len(r["forgetting"]) if r["forgetting"] else 0.0 for r in runs]
        bpc_forgs = [sum(r["bpc_forgetting"]) / len(r["bpc_forgetting"]) if r["bpc_forgetting"] else 0.0 for r in runs]
        
        # Calculate peak, retained, and FWT from matrices
        peak_list = []
        retained_list = []
        fwt_list = []

        for r in runs:
            matrix = r.get("accuracy_matrix_after", [])
            random_acc = r.get("random_accuracy", 1.0 / 50.0)
            if matrix:
                peaks = compute_peak_accuracy(matrix)
                retained = compute_retained_accuracy(matrix)
                fwt = compute_forward_transfer(matrix, random_accuracy=random_acc)
                
                peak_list.append(np.mean(peaks))
                retained_list.append(np.mean(retained))
                if fwt:
                    fwt_list.append(np.mean(fwt))
                else:
                    fwt_list.append(0.0)
            else:
                peak_list.append(r.get("mean_peak_accuracy", r["average_accuracy_after"]))
                retained_list.append(r.get("mean_retained_accuracy", r["average_accuracy_after"]))
                fwt_list.append(r.get("mean_forward_transfer", 0.0))

        peak_mean, peak_std = np.mean(peak_list), np.std(peak_list)
        retained_mean, retained_std = np.mean(retained_list), np.std(retained_list)
        fwt_mean, fwt_std = np.mean(fwt_list), np.std(fwt_list)

        rep_rates = [r.get("repetition_rate_mean", 0.0) for r in runs]
        dist2 = [r.get("distinct_2_mean", 0.0) for r in runs]
        dist3 = [r.get("distinct_3_mean", 0.0) for r in runs]
        
        births = [r.get("births", 0) for r in runs]
        prunes = [r.get("prunes", 0) for r in runs]
        runtimes = [r["runtime_seconds"] for r in runs]

        # Final dims representation
        dims_list = [r.get("final_dims", []) for r in runs]
        dims_str = "/" .join(str(d) for d in dims_list[0]) if dims_list else ""

        # Probe accuracy per layer
        probe_l0, probe_l1, probe_l2 = [], [], []
        for r in runs:
            if "probe_accuracies_after" in r and r["probe_accuracies_after"]:
                final_eval_probes = r["probe_accuracies_after"][-1]
                if final_eval_probes:
                    mean_probes = np.mean(final_eval_probes, axis=0)
                    if isinstance(mean_probes, np.ndarray) and mean_probes.ndim > 0:
                        if len(mean_probes) > 0:
                            probe_l0.append(mean_probes[0])
                        if len(mean_probes) > 1:
                            probe_l1.append(mean_probes[1])
                        if len(mean_probes) > 2:
                            probe_l2.append(mean_probes[2])

        p0_mean = np.mean(probe_l0) if probe_l0 else 0.0
        p1_mean = np.mean(probe_l1) if probe_l1 else 0.0
        p2_mean = np.mean(probe_l2) if probe_l2 else 0.0

        acc_mean, acc_std = np.mean(accs), np.std(accs)
        bpc_mean, bpc_std = np.mean(bpcs), np.std(bpcs)
        acc_forg_mean, acc_forg_std = np.mean(acc_forgs), np.std(acc_forgs)
        bpc_forg_mean, bpc_forg_std = np.mean(bpc_forgs), np.std(bpc_forgs)
        
        rep_mean = np.mean(rep_rates)
        dist2_mean = np.mean(dist2)
        dist3_mean = np.mean(dist3)
        
        birth_mean = np.mean(births)
        prune_mean = np.mean(prunes)
        runtime_mean = np.mean(runtimes)

        # Append to CSV
        csv_rows.append([
            model,
            seeds_count,
            acc_mean, acc_std,
            bpc_mean, bpc_std,
            peak_mean, peak_std,
            retained_mean, retained_std,
            fwt_mean, fwt_std,
            acc_forg_mean, acc_forg_std,
            bpc_forg_mean, bpc_forg_std,
            p0_mean,
            p1_mean,
            p2_mean,
            rep_mean,
            dist2_mean,
            dist3_mean,
            dims_str,
            birth_mean,
            prune_mean,
            runtime_mean
        ])

        # Append to MD
        p0_str = f"{p0_mean:.3f}" if probe_l0 else "-"
        p1_str = f"{p1_mean:.3f}" if probe_l1 else "-"
        p2_str = f"{p2_mean:.3f}" if probe_l2 else "-"

        md_report += (
            f"| `{model}` | {seeds_count} | "
            f"{acc_mean:.3f}±{acc_std:.3f} | {peak_mean:.3f}±{peak_std:.3f} | "
            f"{retained_mean:.3f}±{retained_std:.3f} | {fwt_mean:.3f}±{fwt_std:.3f} | "
            f"{acc_forg_mean:.3f}±{acc_forg_std:.3f} | {bpc_forg_mean:.3f}±{bpc_forg_std:.3f} | "
            f"{p0_str} | {p1_str} | {p2_str} | "
            f"{rep_mean:.1%} | {dist2_mean:.1%} | {dist3_mean:.1%} | "
            f"`{dims_str}` | {birth_mean:.1f} | {prune_mean:.1f} | {runtime_mean:.1f}s |\n"
        )

    # Perform significance testing relative to deep_agnis_3L_neurogenesis if it exists
    target_model = "deep_agnis_3L_neurogenesis"
    if target_model in raw_data:
        target_runs = raw_data[target_model]
        target_accs = [r["average_accuracy_after"] for r in target_runs]
        
        md_report += "\n## Statistical Significance Tests (relative to `deep_agnis_3L_neurogenesis` Accuracy)\n\n"
        md_report += "| Comparison Model | p-value (t-test) | Significant (p < 0.05)? |\n"
        md_report += "|---|---|---|\n"
        
        from scipy import stats
        for model in sorted(raw_data.keys()):
            if model == target_model:
                continue
            model_runs = raw_data[model]
            model_accs = [r["average_accuracy_after"] for r in model_runs]
            
            if len(model_accs) > 1 and len(target_accs) > 1:
                t_stat, p_val = stats.ttest_ind(target_accs, model_accs, equal_var=False)
                sig = "Yes" if p_val < 0.05 else "No"
                md_report += f"| `{model}` | {p_val:.4f} | {sig} |\n"

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
    results_folder = "results/phase6"
    if len(sys.argv) > 1:
        results_folder = sys.argv[1]
    summarize_phase6_results(results_folder)
