"""
Raw AGNIS — experiments/phase2_sequences/plot_phase2.py

Generates visualization plots for Phase 2 sequence benchmark runs.
"""

import os
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless runs
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_accuracy_matrix(
    run_dir: str,
    accuracy_matrix: List[List[Optional[float]]],
    condition: str,
    model_name: str,
    seed: int,
):
    """Plot task accuracy matrix as a heatmap."""
    matrix = np.array(
        [[v if v is not None else np.nan for v in row] for row in accuracy_matrix]
    )
    n_tasks = matrix.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)

    # Labels
    ax.set_xticks(np.arange(n_tasks))
    ax.set_yticks(np.arange(n_tasks))
    ax.set_xticklabels([f"T{i}" for i in range(n_tasks)])
    ax.set_yticklabels([f"Ckpt {i}" for i in range(n_tasks)])
    ax.set_xlabel("Evaluated Task")
    ax.set_ylabel("Checkpoint Boundary")
    ax.set_title(f"Accuracy Matrix ({condition} | {model_name} | Seed {seed})")

    # Annotate text
    for i in range(n_tasks):
        for j in range(n_tasks):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "black" if val > 0.5 else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    weight="bold",
                )

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()

    out_path = os.path.join(run_dir, "plots", "accuracy_matrix_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_plots(
    run_dir: str,
    accuracy_matrix: List[List[Optional[float]]],
    prediction_errors: List[float],
    recurrent_drives: List[float],
    condition: str,
    model_name: str,
    seed: int,
):
    """Generate all baseline plots for a completed benchmark run."""
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    
    # 1. Accuracy matrix heatmap
    try:
        plot_accuracy_matrix(run_dir, accuracy_matrix, condition, model_name, seed)
    except Exception as e:
        print(f"[Plotting] Failed accuracy heatmap: {e}")

    # 2. Prediction error curve
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(prediction_errors, color="#d9534f", alpha=0.85, label="Prediction Error")
        ax.set_xlabel("Transition Training Step")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"Prediction Error Decay ({condition} | {model_name})")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "plots", "prediction_error_curve.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[Plotting] Failed error curve: {e}")

    # 3. Recurrent drive norm curve
    try:
        if recurrent_drives and any(v > 0 for v in recurrent_drives):
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(recurrent_drives, color="#0275d8", alpha=0.85, label="Recurrent Drive Norm")
            ax.set_xlabel("Transition Step")
            ax.set_ylabel("L2 Norm")
            ax.set_title(f"Recurrent State Drive Norm ({condition} | {model_name})")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "plots", "recurrent_drive_curve.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"[Plotting] Failed recurrent drive curve: {e}")
