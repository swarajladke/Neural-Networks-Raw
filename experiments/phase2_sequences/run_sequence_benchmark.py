"""
Raw AGNIS — experiments/phase2_sequences/run_sequence_benchmark.py

Phase 2 Continual Sequence Prediction Benchmark Runner.
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
from agnis.sequence.sequence_wrapper import (
    SeqAgnisModel,
    SimpleRNNBaseline,
    MLPWindowBaseline,
)
from agnis.sequence.temporal_metrics import (
    evaluate_model_on_sequence_task,
    compute_temporal_consistency,
)
from agnis.evaluation.phase1_logging import format_threshold


def parse_args():
    parser = argparse.ArgumentParser(description="Raw AGNIS Phase 2 Sequence Benchmark Runner")
    parser.add_argument(
        "--condition",
        type=str,
        default="periodic",
        choices=["periodic", "doublet", "copy", "palindrome"],
        help="Benchmark sequence task condition",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="seq_agnis_full_fixed",
        choices=[
            "mlp_context_window",
            "simple_rnn",
            "seq_agnis_no_recurrent",
            "seq_agnis_recurrent",
            "seq_agnis_recurrent_kwta",
            "seq_agnis_recurrent_memory",
            "seq_agnis_recurrent_replay",
            "seq_agnis_full_fixed",
        ],
        help="Model baseline or ablation variant",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny local validation run")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit without running")
    parser.add_argument("--write-threshold", type=float, default=None, help="Overwrite write error threshold")
    parser.add_argument("--novelty-threshold", type=float, default=None, help="Overwrite write novelty threshold")
    parser.add_argument("--sensitivity-mode", action="store_true", help="Partition results directory by thresholds")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def initialize_seq_model(model_name: str, config: AGNISConfig, d_in: int, d_out: int) -> Any:
    """Factory to initialize Seq AGNIS and baseline wrappers."""
    if model_name == "mlp_context_window":
        return MLPWindowBaseline(d_in=d_in, d_out=d_out, context_window=2, d_hidden=64, lr=0.01)
    elif model_name == "simple_rnn":
        return SimpleRNNBaseline(d_in=d_in, d_out=d_out, d_hidden=32, lr=0.01)
    
    # Seq AGNIS configurations
    elif model_name == "seq_agnis_no_recurrent":
        config.model.use_recurrent = False
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=config.model.d_z, config=config,
            R_update_enabled=False, R_drive_enabled=False, use_recurrent=False,
            use_memory=False, use_replay=False
        )
    elif model_name == "seq_agnis_recurrent":
        config.model.use_recurrent = True
        config.model.use_sparsity = False
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=config.model.d_z, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=False, use_replay=False
        )
    elif model_name == "seq_agnis_recurrent_kwta":
        config.model.use_recurrent = True
        config.model.use_sparsity = True
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=config.model.d_z, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=False, use_replay=False
        )
    elif model_name == "seq_agnis_recurrent_memory":
        config.model.use_recurrent = True
        config.model.use_sparsity = True
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=config.model.d_z, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=True, use_replay=False
        )
    elif model_name in ["seq_agnis_recurrent_replay", "seq_agnis_full_fixed"]:
        config.model.use_recurrent = True
        config.model.use_sparsity = True
        return SeqAgnisModel(
            d_in=d_in, d_out=d_out, d_z=config.model.d_z, config=config,
            R_update_enabled=True, R_drive_enabled=True, use_recurrent=True,
            use_memory=True, use_replay=True
        )
    
    raise ValueError(f"Unknown sequence model baseline: {model_name}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = default_config()

    # Apply overrides
    if args.smoke:
        print("[SeqBenchmark] Smoke test active. Overriding settings.")
        config.training.n_tasks = 2
        config.training.sequences_per_task = 2
        config.training.sequence_length = 6
        config.training.epochs_per_task = 1
        config.model.d_z = 8
        config.memory.capacity = 10

    if args.write_threshold is not None:
        config.memory.write_error_threshold = args.write_threshold
    if args.novelty_threshold is not None:
        config.memory.write_novelty_threshold = args.novelty_threshold

    # Determine vocabulary and sequence lengths per task condition
    n_tasks = config.training.n_tasks
    seqs_per_task = config.training.sequences_per_task
    vocab_size_per_task = 4
    total_vocab_size = n_tasks * vocab_size_per_task
    
    copy_length = 3
    half_length = 3
    
    # 2. Generate Tasks
    if args.condition == "periodic":
        tasks = generate_periodic_tasks(n_tasks, seqs_per_task, config.training.sequence_length, vocab_size_per_task)
    elif args.condition == "doublet":
        tasks = generate_doublet_tasks(n_tasks, seqs_per_task, config.training.sequence_length, vocab_size_per_task)
    elif args.condition == "copy":
        tasks = generate_copy_tasks(n_tasks, seqs_per_task, copy_length, vocab_size_per_task)
        config.training.sequence_length = 2 * copy_length + 1
    elif args.condition == "palindrome":
        tasks = generate_palindrome_tasks(n_tasks, seqs_per_task, half_length, vocab_size_per_task)
        config.training.sequence_length = 2 * half_length
    else:
        raise ValueError(f"Unknown condition: {args.condition}")

    if args.dry_run:
        print("\n=== Seq Benchmark Dry Run Plan ===")
        print(f"Condition: {args.condition}")
        print(f"Model: {args.model}")
        print(f"Seed: {args.seed}")
        print(f"Num Tasks: {n_tasks}")
        print(f"Vocab size: {total_vocab_size}")
        print(f"Sequence length: {config.training.sequence_length}")
        print(f"===================================\n")
        sys.exit(0)

    start_time = time.time()

    # 3. Initialize wrapper
    model = initialize_seq_model(args.model, config, d_in=total_vocab_size, d_out=total_vocab_size)

    # Logging structures
    prediction_errors = []
    recurrent_drive_norms = []
    active_fractions = []
    memory_usages = []
    
    accuracy_before_sleep = []
    accuracy_after_sleep = []
    consistency_before_sleep = []
    consistency_after_sleep = []

    # 4. Learning loop
    for t_idx, task in enumerate(tasks):
        print(f"[SeqBenchmark] Training Task {t_idx}: {task.name}...")
        model.start_task(t_idx)

        for epoch in range(config.training.epochs_per_task):
            # Shuffle sequences
            seqs = list(task.sequences)
            random.shuffle(seqs)

            for seq in seqs:
                model.reset_sequence_state()
                for t in range(len(seq) - 1):
                    x = torch.zeros(total_vocab_size)
                    x[seq[t]] = 1.0
                    y = torch.zeros(total_vocab_size)
                    y[seq[t+1]] = 1.0

                    train_metrics = model.train_transition(x, y)
                    prediction_errors.append(train_metrics.get("error", 0.0))

                    stats = model.get_stats()
                    sparsity = stats.get("sparsity_level", 0.0)
                    active_fractions.append(1.0 - sparsity)
                    memory_usages.append(stats.get("memory_size", 0))
                    recurrent_drive_norms.append(stats.get("recurrent_drive_norm_mean", 0.0))

        # Evaluation before sleep
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

        # Sleep/consolidation
        if hasattr(model, "sleep"):
            print(f"[SeqBenchmark] Sleep consolidation after Task {t_idx}...")
            model.sleep()

        # Evaluation after sleep
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

    # 5. Compute CL Metrics
    from agnis.evaluation.continual_metrics import compute_phase1_metrics
    random_accuracy = 1.0 / total_vocab_size
    cl_metrics_before = compute_phase1_metrics(accuracy_before_sleep, random_accuracy=random_accuracy)
    cl_metrics = compute_phase1_metrics(accuracy_matrix, random_accuracy=random_accuracy)

    final_stats = model.get_stats()

    # Build results schema
    results_schema = {
        "phase": "phase2_sequences",
        "condition": args.condition,
        "model": args.model,
        "seed": args.seed,
        "run_id": f"phase2_{args.condition}_{args.model}_write_{format_threshold(config.memory.write_error_threshold)}_novelty_{format_threshold(config.memory.write_novelty_threshold)}_seed_{args.seed}",
        "num_tasks": n_tasks,
        "sequence_length": config.training.sequence_length,
        "sequences_per_task": seqs_per_task,
        "latent_dim": config.model.d_z,
        "uses_recurrent_state": config.model.use_recurrent,
        "uses_kwta": config.model.use_sparsity,
        "uses_memory": getattr(model, "use_memory", False),
        "uses_replay": getattr(model, "use_replay", False),
        
        # Performance metrics
        "final_average_accuracy": cl_metrics.get("final_average_accuracy", 0.0),
        "average_forgetting": cl_metrics.get("average_forgetting", 0.0),
        "forgetting_per_task": cl_metrics.get("forgetting_per_task", {}),
        "backward_transfer": cl_metrics.get("backward_transfer", 0.0),
        "forward_transfer": cl_metrics.get("forward_transfer", 0.0),
        "accuracy_before_sleep": accuracy_before_sleep,
        "accuracy_after_sleep": accuracy_after_sleep,
        "forgetting_before_sleep": cl_metrics_before.get("average_forgetting", 0.0),
        "forgetting_after_sleep": cl_metrics.get("average_forgetting", 0.0),
        "replay_benefit": cl_metrics.get("final_average_accuracy", 0.0) - cl_metrics_before.get("final_average_accuracy", 0.0),
        
        # Temporal consistency metrics
        "consistency_before_sleep": consistency_before_sleep,
        "consistency_after_sleep": consistency_after_sleep,
        "final_temporal_consistency": consistency_after_sleep[-1][-1] if consistency_after_sleep else 0.0,
        
        # Diagnostics
        "mean_prediction_error": sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0,
        "final_prediction_error": prediction_errors[-1] if prediction_errors else 0.0,
        "mean_active_fraction": sum(active_fractions) / len(active_fractions) if active_fractions else 0.0,
        
        # Recurrent matrix metrics
        "rho": final_stats.get("rho", 0.0),
        "eta_R": final_stats.get("eta_R", 0.0),
        "R_norm_final": final_stats.get("R_norm_final", 0.0),
        "R_update_norm_mean": final_stats.get("R_update_norm_mean", 0.0),
        "recurrent_drive_norm_mean": sum(recurrent_drive_norms) / len(recurrent_drive_norms) if recurrent_drive_norms else 0.0,
        
        # Memory metrics
        "final_memory_size": final_stats.get("memory_size", 0),
        "memory_hit_rate": final_stats.get("memory_hit_rate", 0.0),
        "memory_writes_per_task": final_stats.get("memory_writes_per_task", []),
        "memory_retrievals_per_task": final_stats.get("memory_retrievals_per_task", []),
        "memory_hits_per_task": final_stats.get("memory_hits_per_task", []),
        "mean_retrieval_similarity": final_stats.get("mean_retrieval_similarity", 0.0),
        "replay_steps_executed": final_stats.get("replay_steps_executed", 0),
        "replay_error_delta": final_stats.get("replay_error_delta", 0.0),
        
        # Metadata
        "state_reset_between_sequences": True,
        "copy_length": copy_length if args.condition == "copy" else None,
        "context_window": 2 if args.model == "mlp_context_window" else None,
        "runtime_seconds": runtime,
        "sensitivity_mode": args.sensitivity_mode,
        "write_error_threshold": config.memory.write_error_threshold,
        "write_novelty_threshold": config.memory.write_novelty_threshold,
    }

    # 6. Save results
    # Generate partitioning directory path if sensitivity_mode is enabled
    if args.sensitivity_mode:
        w_str = format_threshold(config.memory.write_error_threshold)
        n_str = format_threshold(config.memory.write_novelty_threshold)
        run_dir = os.path.join(
            config.results_dir,
            args.condition,
            args.model,
            f"write_{w_str}_novelty_{n_str}",
            f"seed_{args.seed}"
        )
    else:
        run_dir = os.path.join(config.results_dir, args.condition, args.model, f"seed_{args.seed}")
        
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results_schema, f, indent=2)

    config_path = os.path.join(run_dir, "config_used.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

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

    # 10. Generate plots
    from plot_phase2 import generate_all_plots
    generate_all_plots(
        run_dir=run_dir,
        accuracy_matrix=accuracy_matrix,
        prediction_errors=prediction_errors,
        recurrent_drives=recurrent_drive_norms,
        condition=args.condition,
        model_name=args.model,
        seed=args.seed,
    )

    print(f"[SeqBenchmark] Run finished successfully. Results saved in {run_dir}")


if __name__ == "__main__":
    main()
