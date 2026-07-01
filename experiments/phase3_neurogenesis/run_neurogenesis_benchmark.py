"""
Raw AGNIS — experiments/phase3_neurogenesis/run_neurogenesis_benchmark.py

Phase 3 Autonomous Neurogenesis Benchmark Runner.
"""

import sys
import os
import argparse
import time
import random
import torch
import yaml
import json
import csv
from typing import List, Tuple, Dict, Any, Optional

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from agnis.utils.config import load_config, default_config, AGNISConfig
from agnis.sequence.sequence_tasks import (
    generate_periodic_tasks,
    generate_doublet_tasks,
    generate_copy_tasks,
    generate_palindrome_tasks,
    SequenceTask,
)
from agnis.sequence.sequence_wrapper import SeqAgnisModel
from agnis.sequence.temporal_metrics import (
    evaluate_model_on_sequence_task,
    compute_temporal_consistency,
)
from agnis.neurogenesis.growth_controller import GrowthController
from agnis.evaluation.phase1_logging import format_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Raw AGNIS Phase 3 Neurogenesis Benchmark Runner")
    parser.add_argument(
        "--condition",
        type=str,
        default="doublet",
        choices=["periodic", "doublet", "copy", "palindrome", "capacity_stress_sequence"],
        help="Benchmark task condition",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="seq_agnis_neurogenesis",
        choices=[
            "seq_agnis_full_fixed",
            "seq_agnis_neurogenesis",
            "seq_agnis_neurogenesis_no_maturity",
            "seq_agnis_neurogenesis_no_pruning",
        ],
        help="Model variant under comparison",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny local validation run")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit without running")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def initialize_neurogenesis_model(model_name: str, config: AGNISConfig, d_in: int, d_out: int) -> SeqAgnisModel:
    """Initialize Seq AGNIS with correct neurogenesis/maturity gating parameters."""
    start_dim = config.model.d_z
    max_dim = 128
    
    # Configure generic fixed model
    config.model.use_recurrent = True
    config.model.use_sparsity = True
    config.memory.use_memory = True
    config.memory.use_replay = True
    
    if model_name == "seq_agnis_full_fixed":
        # Effectively disabled growth by capping max_dim at start_dim
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=start_dim, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=True, use_replay=True,
            maturity_enabled=True, max_latent_dim=start_dim
        )
    elif model_name == "seq_agnis_neurogenesis":
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=start_dim, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=True, use_replay=True,
            maturity_enabled=True, max_latent_dim=max_dim
        )
    elif model_name == "seq_agnis_neurogenesis_no_maturity":
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=start_dim, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=True, use_replay=True,
            maturity_enabled=False, max_latent_dim=max_dim
        )
    elif model_name == "seq_agnis_neurogenesis_no_pruning":
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=start_dim, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=True, use_replay=True,
            maturity_enabled=True, max_latent_dim=max_dim
        )
        
    raise ValueError(f"Unknown model: {model_name}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.config:
        config = load_config(args.config)
    else:
        config = default_config()

    # Apply smoke overrides
    if args.smoke:
        print("[NeurogenesisBenchmark] Smoke test active. Overriding settings.")
        config.training.n_tasks = 2
        config.training.sequences_per_task = 2
        config.training.sequence_length = 6
        config.training.epochs_per_task = 1
        config.model.d_z = 8
        config.memory.capacity = 10

    n_tasks = config.training.n_tasks
    seqs_per_task = config.training.sequences_per_task
    vocab_size_per_task = 4
    total_vocab_size = n_tasks * vocab_size_per_task
    
    copy_length = 3
    half_length = 3

    # Generate sequence tasks
    if args.condition in ["doublet", "capacity_stress_sequence"]:
        tasks = generate_doublet_tasks(n_tasks, seqs_per_task, config.training.sequence_length, vocab_size_per_task)
    elif args.condition == "periodic":
        tasks = generate_periodic_tasks(n_tasks, seqs_per_task, config.training.sequence_length, vocab_size_per_task)
    elif args.condition == "copy":
        tasks = generate_copy_tasks(n_tasks, seqs_per_task, copy_length, vocab_size_per_task)
        config.training.sequence_length = 2 * copy_length + 1
    elif args.condition == "palindrome":
        tasks = generate_palindrome_tasks(n_tasks, seqs_per_task, half_length, vocab_size_per_task)
        config.training.sequence_length = 2 * half_length
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    # Set up capacity stress conditions (e.g. 5 tasks instead of 3 to saturate bottleneck)
    if args.condition == "capacity_stress_sequence" and not args.smoke:
        print("[NeurogenesisBenchmark] Capacity Stress sequence sweep enabled.")
        config.training.n_tasks = 5
        n_tasks = 5
        total_vocab_size = 5 * vocab_size_per_task
        # Re-generate tasks for 5 tasks
        tasks = generate_doublet_tasks(5, seqs_per_task, config.training.sequence_length, vocab_size_per_task)

    if args.dry_run:
        print("\n=== Neurogenesis Sweep Dry Run Plan ===")
        print(f"Condition: {args.condition}")
        print(f"Model: {args.model}")
        print(f"Seed: {args.seed}")
        print(f"Num Tasks: {n_tasks}")
        print(f"Vocab size: {total_vocab_size}")
        print(f"Sequence length: {config.training.sequence_length}")
        print(f"========================================\n")
        sys.exit(0)

    start_time = time.time()

    # Initialize model
    model = initialize_neurogenesis_model(args.model, config, d_in=total_vocab_size, d_out=total_vocab_size)

    # Growth controller setup
    gc = GrowthController(threshold=0.35, consecutive_n=5, lambda_cost=0.01)

    # Diagnostic logs
    prediction_errors = []
    recurrent_drive_norms = []
    active_fractions = []
    capacity_timeline = []
    birth_events = 0
    prune_events = 0

    accuracy_before_sleep = []
    accuracy_after_sleep = []
    consistency_before_sleep = []
    consistency_after_sleep = []

    step_counter = 0

    # Training
    for t_idx, task in enumerate(tasks):
        print(f"[NeurogenesisBenchmark] Training Task {t_idx}: {task.name}...")
        model.start_task(t_idx)

        for epoch in range(config.training.epochs_per_task):
            seqs = list(task.sequences)
            random.shuffle(seqs)

            for seq in seqs:
                model.reset_sequence_state()
                for t in range(len(seq) - 1):
                    step_counter += 1
                    x = torch.zeros(total_vocab_size)
                    x[seq[t]] = 1.0
                    y = torch.zeros(total_vocab_size)
                    y[seq[t+1]] = 1.0

                    train_metrics = model.train_transition(x, y)
                    error = train_metrics.get("error", 0.0)
                    prediction_errors.append(error)

                    stats = model.get_stats()
                    sparsity = stats.get("sparsity_level", 0.0)
                    active_fractions.append(1.0 - sparsity)
                    recurrent_drive_norms.append(stats.get("recurrent_drive_norm_mean", 0.0))

                    # Growth check
                    if args.model != "seq_agnis_full_fixed":
                        current_capacity = model.base_model.cell.d_z
                        novelty = 1.0 - model.base_model.last_retrieval_similarity
                        
                        import math
                        joint_dim = total_vocab_size * 2
                        error_l2 = math.sqrt(error * joint_dim)

                        trigger = gc.update(
                            error=error_l2,
                            novelty=novelty,
                            uncertainty=0.0,
                            interference=0.0,
                            coverage=0.0,
                            cost=float(current_capacity),
                        )
                        
                        if trigger and current_capacity < model.base_model.cell.max_latent_dim:
                            # Trigger birth of 2 units
                            joint_input = torch.cat([x, y])
                            model.base_model.cell.grow_units(
                                k=2,
                                current_input=joint_input,
                                residual_error=model.base_model.cell._last_error
                            )
                            birth_events += 1

                    capacity_timeline.append((step_counter, model.base_model.cell.d_z))

        # Evaluate before sleep
        row_before = []
        c_before = []
        for eval_idx in range(n_tasks):
            if eval_idx > t_idx:
                row_before.append(None)
                c_before.append(None)
            else:
                eval_task = tasks[eval_idx]
                acc = evaluate_model_on_sequence_task(model, eval_task.sequences, total_vocab_size)
                cons = compute_temporal_consistency(model, args.condition, eval_task, total_vocab_size)
                row_before.append(acc)
                c_before.append(cons)
        accuracy_before_sleep.append(row_before)
        consistency_before_sleep.append(c_before)
        print(f"  Next-symbol accuracies before sleep: {row_before}")

        # Sleep/replay
        if hasattr(model, "sleep"):
            print(f"[NeurogenesisBenchmark] Sleep consolidation after Task {t_idx}...")
            model.sleep()

        # Pruning boundary (only if enabled)
        if args.model in ["seq_agnis_neurogenesis", "seq_agnis_neurogenesis_no_maturity"] and not args.smoke:
            initial_cap = model.base_model.cell.d_z
            # Conservative pruning: min_age = 50, usage_threshold = 0.01, importance = 0.01, maturity = 0.5
            model.base_model.cell.prune_units(min_age=50, usage_threshold=0.01, importance_threshold=0.01, maturity_threshold=0.5)
            final_cap = model.base_model.cell.d_z
            if final_cap < initial_cap:
                prune_events += (initial_cap - final_cap)

        # Evaluate after sleep
        row_after = []
        c_after = []
        for eval_idx in range(n_tasks):
            if eval_idx > t_idx:
                row_after.append(None)
                c_after.append(None)
            else:
                eval_task = tasks[eval_idx]
                acc = evaluate_model_on_sequence_task(model, eval_task.sequences, total_vocab_size)
                cons = compute_temporal_consistency(model, args.condition, eval_task, total_vocab_size)
                row_after.append(acc)
                c_after.append(cons)
        accuracy_after_sleep.append(row_after)
        consistency_after_sleep.append(c_after)
        print(f"  Next-symbol accuracies after sleep:  {row_after}")

    accuracy_matrix = accuracy_after_sleep
    runtime = time.time() - start_time

    # Compute CL Metrics
    from agnis.evaluation.continual_metrics import compute_phase1_metrics
    random_accuracy = 1.0 / total_vocab_size
    cl_metrics_before = compute_phase1_metrics(accuracy_before_sleep, random_accuracy=random_accuracy)
    cl_metrics = compute_phase1_metrics(accuracy_matrix, random_accuracy=random_accuracy)

    final_stats = model.get_stats()

    # Results schema
    results_schema = {
        "phase": "phase3_neurogenesis",
        "condition": args.condition,
        "model": args.model,
        "seed": args.seed,
        "run_id": f"phase3_{args.condition}_{args.model}_seed_{args.seed}",
        "num_tasks": n_tasks,
        "sequence_length": config.training.sequence_length,
        "sequences_per_task": seqs_per_task,
        
        # Growth and capacity metrics
        "initial_latent_dim": config.model.d_z,
        "final_latent_dim": model.base_model.cell.d_z,
        "birth_events": birth_events,
        "prune_events": prune_events,
        "max_latent_dim_reached": max([cap for _, cap in capacity_timeline]) if capacity_timeline else config.model.d_z,
        
        # CL performance
        "final_average_accuracy": cl_metrics.get("final_average_accuracy", 0.0),
        "average_forgetting": cl_metrics.get("average_forgetting", 0.0),
        "forgetting_per_task": cl_metrics.get("forgetting_per_task", {}),
        "backward_transfer": cl_metrics.get("backward_transfer", 0.0),
        "forward_transfer": cl_metrics.get("forward_transfer", 0.0),
        "accuracy_before_sleep": accuracy_before_sleep,
        "accuracy_after_sleep": accuracy_after_sleep,
        "replay_benefit": cl_metrics.get("final_average_accuracy", 0.0) - cl_metrics_before.get("final_average_accuracy", 0.0),
        
        # Temporal metrics
        "consistency_before_sleep": consistency_before_sleep,
        "consistency_after_sleep": consistency_after_sleep,
        "final_temporal_consistency": consistency_after_sleep[-1][-1] if consistency_after_sleep else 0.0,
        
        # Diagnostic averages
        "mean_prediction_error": sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0,
        "mean_active_fraction": sum(active_fractions) / len(active_fractions) if active_fractions else 0.0,
        "recurrent_drive_norm_mean": sum(recurrent_drive_norms) / len(recurrent_drive_norms) if recurrent_drive_norms else 0.0,
        
        # Memory stats
        "final_memory_size": final_stats.get("memory_size", 0),
        "memory_hit_rate": final_stats.get("memory_hit_rate", 0.0),
        "runtime_seconds": runtime,
    }

    # Save outputs
    run_dir = os.path.join(config.results_dir, "phase3", args.condition, args.model, f"seed_{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(results_schema, f, indent=2)

    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

    # Save capacity timeline
    cap_path = os.path.join(run_dir, "capacity_timeline.csv")
    with open(cap_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "LatentDim"])
        for step, dim in capacity_timeline:
            writer.writerow([step, dim])

    # Save accuracy matrix to CSV
    acc_path = os.path.join(run_dir, "accuracy_matrix.csv")
    with open(acc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"Task_{i}" for i in range(n_tasks)])
        for row in accuracy_matrix:
            writer.writerow([v if v is not None else "" for v in row])

    # Save prediction errors CSV
    err_path = os.path.join(run_dir, "prediction_error.csv")
    with open(err_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "PredictionError"])
        for i, val in enumerate(prediction_errors):
            writer.writerow([i, val])

    print(f"[NeurogenesisBenchmark] Run finished successfully. Results saved in {run_dir}")


if __name__ == "__main__":
    main()
