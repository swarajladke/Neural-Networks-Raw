"""
Raw AGNIS — experiments/phase1_associative/run_memory_sensitivity.py

Sweep runner script to perform memory and replay sensitivity stress tests.
"""

import sys
import os
import argparse
import subprocess
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Raw AGNIS Memory Sensitivity Sweep Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kaggle_phase1_memory_sensitivity.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["clustered", "capacity_stress", "overlapping"],
        help="Conditions to sweep",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["agnis_kwta", "agnis_memory", "agnis_replay", "agnis_full_fixed"],
        help="Models to sweep",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Seeds to sweep",
    )
    parser.add_argument(
        "--write-thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.1, 0.05, 0.03, 0.01],
        help="Write error thresholds to sweep",
    )
    parser.add_argument(
        "--novelty-thresholds",
        type=float,
        nargs="+",
        default=[0.15, 0.10, 0.05],
        help="Novelty thresholds to sweep",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=None,
        help="Latent dims to sweep (overrides config d_z)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=None,
        help="Override number of tasks (e.g. 5 or 10)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print execution plan without training",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    benchmark_script = os.path.join(
        os.path.dirname(__file__), "run_benchmark.py"
    )
    
    # Calculate total runs
    latent_dims = args.latent_dims if args.latent_dims else [None]
    total_runs = (
        len(args.conditions)
        * len(args.models)
        * len(args.seeds)
        * len(args.write_thresholds)
        * len(args.novelty_thresholds)
        * len(latent_dims)
    )
    
    print(f"\n================ Memory Sensitivity Sweep Plan ================")
    print(f"Config: {args.config}")
    print(f"Conditions: {args.conditions}")
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Write Thresholds: {args.write_thresholds}")
    print(f"Novelty Thresholds: {args.novelty_thresholds}")
    print(f"Latent Dims Override: {args.latent_dims}")
    print(f"N Tasks Override: {args.n_tasks}")
    print(f"Total planned runs: {total_runs}")
    print(f"===============================================================\n")

    run_count = 0
    for condition in args.conditions:
        for model in args.models:
            for seed in args.seeds:
                for w_thresh in args.write_thresholds:
                    for n_thresh in args.novelty_thresholds:
                        for l_dim in latent_dims:
                            run_count += 1
                            cmd = [
                                sys.executable,
                                benchmark_script,
                                "--condition", condition,
                                "--model", model,
                                "--seed", str(seed),
                                "--config", args.config,
                                "--write-threshold", str(w_thresh),
                                "--novelty-threshold", str(n_thresh),
                            ]
                            if l_dim is not None:
                                cmd += ["--latent-dim", str(l_dim)]
                            if args.n_tasks is not None:
                                cmd += ["--n-tasks", str(args.n_tasks)]
                            if args.dry_run:
                                cmd.append("--dry-run")
                                
                            print(f"[{run_count}/{total_runs}] Running: {' '.join(cmd)}")
                            
                            if not args.dry_run:
                                try:
                                    subprocess.run(cmd, check=True)
                                except subprocess.CalledProcessError as e:
                                    print(f"[Sweep] Error on run {run_count}: {e}")
                                    print("[Sweep] Continuing sweep...")

    print(f"\n[Sweep] Completed memory sensitivity sweep. Total attempted runs: {run_count}\n")


if __name__ == "__main__":
    main()
