"""
Raw AGNIS — experiments/phase3_neurogenesis/run_neurogenesis_sweep.py

Phase 3 Sweep Coordinator for multi-model, multi-seed neurogenesis benchmark configurations.
"""

import os
import sys
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add directory paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))


def run_single_benchmark(args_tuple) -> dict:
    """Run a single benchmark seed config as a subprocess."""
    condition, model, seed, config_path, smoke = args_tuple
    
    script_path = os.path.join(os.path.dirname(__file__), "run_neurogenesis_benchmark.py")
    
    cmd = [
        "python",
        script_path,
        "--condition", condition,
        "--model", model,
        "--seed", str(seed),
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if smoke:
        cmd.append("--smoke")
        
    print(f"[Sweep] Starting: model={model}, condition={condition}, seed={seed}...")
    start_time = time.time()
    
    # Run process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    elapsed = time.time() - start_time
    exit_code = process.returncode
    
    return {
        "model": model,
        "condition": condition,
        "seed": seed,
        "elapsed": elapsed,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr
    }


def main():
    parser = argparse.ArgumentParser(description="Raw AGNIS Phase 3 Neurogenesis Sweep Coordinator")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent processes")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to sweep over")
    parser.add_argument("--smoke", action="store_true", help="Run a single fast validation sweep configuration")
    parser.add_argument("--config", type=str, default=None, help="Custom configuration file path")
    parser.add_argument(
        "--conditions",
        type=str,
        default="doublet,capacity_stress_sequence",
        help="Comma-separated task conditions to run",
    )
    args = parser.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",")]
    
    # Define models to run
    models = [
        "seq_agnis_full_fixed",
        "seq_agnis_neurogenesis",
        "seq_agnis_neurogenesis_no_maturity",
        "seq_agnis_neurogenesis_no_pruning",
    ]

    seeds = list(range(args.seeds))

    if args.smoke:
        print("[Sweep] Smoke test mode active. Limiting sweep scope.")
        seeds = [0]
        models = ["seq_agnis_neurogenesis"]
        conditions = ["doublet"]

    # Build queue
    tasks_queue = []
    for cond in conditions:
        for model in models:
            for seed in seeds:
                tasks_queue.append((cond, model, seed, args.config, args.smoke))

    total_runs = len(tasks_queue)
    print(f"[Sweep] Queue built with {total_runs} benchmark configurations.")
    print(f"[Sweep] Concurrent workers: {args.workers}")
    print(f"[Sweep] Sweep conditions: {conditions}")
    print(f"[Sweep] Sweep models: {models}")
    print(f"[Sweep] Seeds: {seeds}\n")

    start_sweep_time = time.time()
    success_count = 0
    failure_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_benchmark, task): task for task in tasks_queue}
        
        for future in as_completed(futures):
            res = future.result()
            task_details = f"model={res['model']}, condition={res['condition']}, seed={res['seed']}"
            if res["exit_code"] == 0:
                print(f"[Sweep] SUCCESS: {task_details} ({res['elapsed']:.1f}s)")
                success_count += 1
            else:
                print(f"[Sweep] FAILED: {task_details} ({res['elapsed']:.1f}s)")
                print(f"--- Stdout ---\n{res['stdout']}\n")
                print(f"--- Stderr ---\n{res['stderr']}\n")
                failure_count += 1

    total_duration = time.time() - start_sweep_time
    print(f"\n=== Sweep Finished ===")
    print(f"Total Runs: {total_runs}")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Total sweep execution time: {total_duration:.1f} seconds")
    print(f"======================")

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
