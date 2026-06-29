"""
Raw AGNIS — experiments/phase1_associative/plot_phase1.py

Plotting script for Phase 1 benchmark harness. Generates heatmaps and line plots
for accuracy, prediction error, active fraction, memory growth, and representation overlaps.
"""

import os
from typing import List, Dict, Any, Optional

try:
    import matplotlib
    matplotlib.use('Agg') # non-interactive backend for server/Kaggle runs
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def generate_all_plots(
    run_dir: str,
    accuracy_matrix: List[List[Optional[float]]],
    forgetting_list: List[float],
    prediction_errors: List[float],
    active_fractions: List[float],
    memory_usages: List[int],
    overlap_matrix: List[List[float]],
) -> bool:
    """
    Generate and save all required benchmark plots.

    Parameters
    ----------
    run_dir : str
        Directory to save plots (into run_dir/plots/)
    ...

    Returns
    -------
    bool
        True if plots were generated, False if skipped due to missing matplotlib.
    """
    if not _HAS_MATPLOTLIB:
        print("[Plotting] matplotlib or numpy is not installed. Skipping plot generation.")
        return False

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Convert lists to numpy arrays
    acc_arr = np.array([[v if v is not None else np.nan for v in row] for row in accuracy_matrix])
    overlap_arr = np.array(overlap_matrix)

    num_tasks = acc_arr.shape[1]

    # 1. Accuracy Matrix Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(acc_arr, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(num_tasks))
    ax.set_yticks(np.arange(acc_arr.shape[0]))
    ax.set_xticklabels([f"T_{i}" for i in range(num_tasks)])
    ax.set_yticklabels([f"Ckpt_{i}" for i in range(acc_arr.shape[0])])
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Evaluation Checkpoints")
    ax.set_title("Task Accuracy Matrix Heatmap")
    fig.colorbar(im, ax=ax, label="Accuracy")
    
    # Loop over data dimensions and create text annotations
    for i in range(acc_arr.shape[0]):
        for j in range(num_tasks):
            val = acc_arr[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val > 0.4 else "white")
                
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_matrix_heatmap.png"), dpi=150)
    plt.close()

    # 2. Forgetting Curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"Task_{i}" for i in range(len(forgetting_list))], forgetting_list, color="#e74c3c", edgecolor="black")
    ax.set_ylabel("Forgetting (Peak - Final)")
    ax.set_xlabel("Tasks")
    ax.set_title("Forgetting Per Task")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "forgetting_curve.png"), dpi=150)
    plt.close()

    # 3. Prediction Error Curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(prediction_errors, color="#3498db", label="Error")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Prediction Error (MSE)")
    ax.set_title("Training Prediction Error Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "prediction_error_curve.png"), dpi=150)
    plt.close()

    # 4. Active Fraction Curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(active_fractions, color="#2ecc71", label="Active Fraction")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Active Fraction (kWTA active / total)")
    ax.set_title("Latent Sparsity Active Fraction Curve")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "active_fraction_curve.png"), dpi=150)
    plt.close()

    # 5. Memory Growth Curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(memory_usages, color="#9b59b6", label="Memory Size")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Memory Entries Count")
    ax.set_title("Episodic Memory Size Growth Curve")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "memory_growth_curve.png"), dpi=150)
    plt.close()

    # 6. Representation Overlap Heatmap
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(overlap_arr, cmap="plasma", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(num_tasks))
    ax.set_yticks(np.arange(num_tasks))
    ax.set_xticklabels([f"T_{i}" for i in range(num_tasks)])
    ax.set_yticklabels([f"T_{i}" for i in range(num_tasks)])
    ax.set_xlabel("Tasks")
    ax.set_ylabel("Tasks")
    ax.set_title("Mean Task Latent Overlap Heatmap")
    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            val = overlap_arr[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) < 0.5 else "black")
            
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "representation_overlap_heatmap.png"), dpi=150)
    plt.close()

    return True
