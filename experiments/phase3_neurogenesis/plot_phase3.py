"""
Raw AGNIS — experiments/phase3_neurogenesis/plot_phase3.py

Generates charts of latent capacity growth timelines and error decay profiles.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Use non-interactive backend for server/no-display environments
plt.switch_backend("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Phase 3 Neurogenesis Metrics")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to seed results directory")
    return parser.parse_args()


def plot_run_metrics(run_dir: str):
    """Plot capacity timeline and prediction error curve."""
    print(f"[Plotting] Reading run data from: {run_dir}")
    
    cap_path = os.path.join(run_dir, "capacity_timeline.csv")
    err_path = os.path.join(run_dir, "prediction_error.csv")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Plot Capacity Timeline
    if os.path.exists(cap_path):
        try:
            df_cap = pd.read_csv(cap_path)
            plt.figure(figsize=(10, 4))
            plt.plot(df_cap["Step"], df_cap["LatentDim"], color="#1f77b4", linewidth=2.5, label="Latent Dimension")
            plt.title("Autonomous Latent Space Capacity Growth Timeline", fontsize=12, fontweight="bold", pad=15)
            plt.xlabel("Training Step", fontsize=10)
            plt.ylabel("Latent Dimension Size", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="upper left")
            plt.tight_layout()
            
            out_path = os.path.join(plots_dir, "capacity_timeline.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[Plotting] Saved capacity timeline to {out_path}")
        except Exception as e:
            print(f"[Plotting] Error plotting capacity timeline: {e}")

    # 2. Plot Prediction Error Curve
    if os.path.exists(err_path):
        try:
            df_err = pd.read_csv(err_path)
            plt.figure(figsize=(10, 4))
            
            # Smooth error curve with rolling average
            rolling_window = min(20, len(df_err))
            smoothed = df_err["PredictionError"].rolling(window=rolling_window, min_periods=1).mean()
            
            plt.plot(df_err["Step"], df_err["PredictionError"], color="#ff7f0e", alpha=0.25, label="Raw MSE")
            plt.plot(df_err["Step"], smoothed, color="#d62728", linewidth=2.0, label=f"Smoothed MSE (EMA-{rolling_window})")
            
            plt.title("Prediction Error Decay Profile", fontsize=12, fontweight="bold", pad=15)
            plt.xlabel("Training Step", fontsize=10)
            plt.ylabel("Mean Squared Error (MSE)", fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="upper right")
            plt.tight_layout()
            
            out_path = os.path.join(plots_dir, "error_decay.png")
            plt.savefig(out_path, dpi=150)
            plt.close()
            print(f"[Plotting] Saved error decay curve to {out_path}")
        except Exception as e:
            print(f"[Plotting] Error plotting error decay curve: {e}")


if __name__ == "__main__":
    args = parse_args()
    plot_run_metrics(args.run_dir)
