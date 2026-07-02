"""
Raw AGNIS — experiments/phase4_char_language/run_char_sweep.py

Phase 4 Sweep Coordinator for multi-model, multi-seed character language configurations.
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
    model, seed, config_path, smoke, domain_order = args_tuple
    
    script_path = os.path.join(os.path.dirname(__file__), "run_char_benchmark.py")
    
    cmd = [
        "python",
        script_path,
        "--model", model,
        "--seed", str(seed),
    ]
    if config_path:
        cmd.extend(["--config", config_path])
    if smoke:
        cmd.append("--smoke")
    if domain_order:
        cmd.extend(["--domain-order", domain_order])
        
    print(f"[Sweep] Starting: model={model}, seed={seed}...")
    start_time = time.time()
    
    # Run process
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    elapsed = time.time() - start_time
    exit_code = process.returncode
    
    return {
        "model": model,
        "seed": seed,
        "elapsed": elapsed,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr
    }


def main():
    parser = argparse.ArgumentParser(description="Raw AGNIS Phase 4 Character Language Sweep Coordinator")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent processes")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to sweep over")
    parser.add_argument("--smoke", action="store_true", help="Run a single fast validation sweep configuration")
    parser.add_argument("--config", type=str, default="configs/kaggle_phase4.yaml", help="Custom configuration file path")
    parser.add_argument("--domain-order", type=str, default="prose,code,arithmetic,dialogue", help="Task execution order")
    parser.add_argument(
        "--models",
        type=str,
        default="bigram_baseline,trigram_baseline,rnn_baseline,seq_agnis_fixed,seq_agnis_neurogenesis,seq_agnis_neuro_no_maturity,seq_agnis_neuro_no_pruning,seq_agnis_no_replay",
        help="Comma-separated model names to evaluate",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    seeds = list(range(args.seeds))

    if args.smoke:
        print("[Sweep] Smoke test mode active. Limiting sweep scope.")
        seeds = [0]
        models = ["seq_agnis_neurogenesis"]

    # Build queue
    tasks_queue = []
    for model in models:
        for seed in seeds:
            tasks_queue.append((model, seed, args.config, args.smoke, args.domain_order))

    total_runs = len(tasks_queue)
    print(f"[Sweep] Queue built with {total_runs} benchmark configurations.")
    print(f"[Sweep] Concurrent workers: {args.workers}")
    print(f"[Sweep] Models: {models}")
    print(f"[Sweep] Seeds: {seeds}")

    # Run tasks in parallel
    completed_runs = 0
    failed_runs = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_benchmark, task): task for task in tasks_queue}
        
        for future in as_completed(futures):
            res = future.result()
            completed_runs += 1
            
            model = res["model"]
            seed = res["seed"]
            elapsed = res["elapsed"]
            exit_code = res["exit_code"]
            
            if exit_code == 0:
                print(f"[Sweep] Completed [{completed_runs}/{total_runs}]: model={model}, seed={seed} ({elapsed:.1f}s)")
            else:
                failed_runs += 1
                print(f"[Sweep] FAILED [{completed_runs}/{total_runs}]: model={model}, seed={seed} ({elapsed:.1f}s) - Exit Code {exit_code}")
                print(f"[Sweep] Stderr details:\n{res['stderr']}")

    total_time = time.time() - start_time
    print(f"\n[Sweep] Finished sweep in {total_time:.1f}s.")
    print(f"[Sweep] Successful runs: {completed_runs - failed_runs}/{total_runs}")
    if failed_runs > 0:
        print(f"[Sweep] WARNING: {failed_runs} runs failed.")


if __name__ == "__main__":
    main()
