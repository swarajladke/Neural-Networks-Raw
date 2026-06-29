"""
Raw AGNIS — src/agnis/utils/visualization.py

Plotting utilities for forgetting curves, error plots, sparsity, and memory.

Requirements: matplotlib (optional — gracefully skips if not installed).
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

from typing import List, Dict, Optional


def plot_forgetting_curves(
    accuracy_matrix: List[List[Optional[float]]],
    task_names: Optional[List[str]] = None,
    title: str = "Task Accuracy Over Time",
    save_path: Optional[str] = None,
):
    """
    Plot accuracy of each task over training checkpoints.

    Parameters
    ----------
    accuracy_matrix : list of list of float
        [checkpoint][task_id] = accuracy or None
    task_names : list of str, optional
    title : str
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    if not _HAS_MATPLOTLIB:
        print("[Visualization] matplotlib not installed. Skipping plot.")
        return

    n_checkpoints = len(accuracy_matrix)
    n_tasks = len(accuracy_matrix[0]) if accuracy_matrix else 0

    if task_names is None:
        task_names = [f"Task {i}" for i in range(n_tasks)]

    fig, ax = plt.subplots(figsize=(8, 5))

    for task_id in range(n_tasks):
        xs, ys = [], []
        for ckpt in range(n_checkpoints):
            acc = accuracy_matrix[ckpt][task_id]
            if acc is not None:
                xs.append(ckpt)
                ys.append(acc)
        if xs:
            ax.plot(xs, ys, marker="o", label=task_names[task_id])

    ax.set_xlabel("Training Checkpoint (task boundary)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Visualization] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_error_curve(
    error_curve: List[float],
    title: str = "Prediction Error Over Time",
    task_boundaries: Optional[List[int]] = None,
    save_path: Optional[str] = None,
):
    """
    Plot prediction error over training steps.

    Parameters
    ----------
    error_curve : list of float
    title : str
    task_boundaries : list of int, optional
        Step indices where new tasks began (shown as vertical lines).
    save_path : str, optional
    """
    if not _HAS_MATPLOTLIB:
        print("[Visualization] matplotlib not installed. Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(error_curve, linewidth=0.8, alpha=0.9, color="#3498db")

    if task_boundaries:
        for step in task_boundaries:
            ax.axvline(x=step, color="#e74c3c", linestyle="--", alpha=0.6, linewidth=1)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Prediction Error (MSE)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Visualization] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_forgetting_comparison(
    model_forgettings: Dict[str, List[float]],
    title: str = "Average Forgetting by Model",
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing average forgetting across multiple models.

    Parameters
    ----------
    model_forgettings : dict
        {"model_name": [per_task_forgetting]} mapping.
    """
    if not _HAS_MATPLOTLIB:
        print("[Visualization] matplotlib not installed. Skipping plot.")
        return

    names = list(model_forgettings.keys())
    avg_forgettings = [
        sum(v) / len(v) if v else 0.0 for v in model_forgettings.values()
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 5))
    colors = ["#e74c3c", "#e67e22", "#f39c12", "#2ecc71", "#3498db", "#9b59b6"]
    bars = ax.bar(names, avg_forgettings, color=colors[:len(names)], edgecolor="white")

    for bar, val in zip(bars, avg_forgettings):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_ylabel("Average Forgetting (↓ better)")
    ax.set_title(title)
    ax.set_ylim(0, max(avg_forgettings) * 1.2 + 0.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Visualization] Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
