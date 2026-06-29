"""
Raw AGNIS — experiments/phase1_associative/run_benchmark.py

Phase 1 Unified Benchmark Runner.
Executes a single model configuration on a specific benchmark condition and seed.

Supports:
- orthogonal, overlapping, clustered, capacity_stress conditions
- mlp, dense_hebbian, agnis_dense, agnis_kwta, agnis_memory, agnis_replay, agnis_full_fixed models
- local smoke tests and full sweeps
- --dry-run and --smoke options
"""

import sys
import os
import argparse
import time
import torch
import random
from typing import List, Tuple, Dict, Any, Optional

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agnis.utils.config import load_config, default_config, AGNISConfig
from agnis.evaluation.baselines import (
    NaiveMLPBaseline,
    DenseHebbianBaseline,
    AgnisBaseline,
)
from agnis.evaluation.continual_metrics import evaluate_model_on_task, compute_phase1_metrics
from agnis.evaluation.representation import (
    compute_task_prototypes,
    compute_pairwise_overlap,
    compute_interference_score,
)
from agnis.evaluation.phase1_logging import save_phase1_run_results, format_threshold
from task_generators import generate_phase1_tasks, AssociationTask
from plot_phase1 import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Raw AGNIS Phase 1 Unified Benchmark Runner")
    parser.add_argument(
        "--condition",
        type=str,
        default="orthogonal",
        choices=["orthogonal", "overlapping", "clustered", "capacity_stress"],
        help="Benchmark difficulty condition",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="agnis_full_fixed",
        choices=[
            "mlp",
            "dense_hebbian",
            "agnis_dense",
            "agnis_kwta",
            "agnis_memory",
            "agnis_replay",
            "agnis_full_fixed",
            "agnis_neurogenesis",
        ],
        help="Model baseline or ablation variant",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny local validation run")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit without running")
    parser.add_argument("--write-threshold", type=float, default=None, help="Overwrite write error threshold")
    parser.add_argument("--novelty-threshold", type=float, default=None, help="Overwrite write novelty threshold")
    parser.add_argument("--latent-dim", type=int, default=None, help="Overwrite model latent dimension d_z")
    parser.add_argument("--n-tasks", type=int, default=None, help="Overwrite number of tasks")
    parser.add_argument("--sensitivity-mode", action="store_true", help="Partition results directory by thresholds")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


def initialize_model(model_name: str, config: AGNISConfig, d_in: int, d_out: int):
    """Factory to initialize baseline and wrapper models."""
    m_cfg = config.model
    t_cfg = config.training
    mem_cfg = config.memory

    if model_name == "mlp":
        return NaiveMLPBaseline(
            d_in=d_in,
            d_out=d_out,
            d_hidden=64,
            lr=0.01,
        )
    elif model_name == "dense_hebbian":
        return DenseHebbianBaseline(
            d_in=d_in,
            d_out=d_out,
            eta=m_cfg.eta_D,
            n_settle=m_cfg.n_settle,
        )
    elif model_name == "agnis_dense":
        return AgnisBaseline(
            d_in=d_in,
            d_out=d_out,
            d_z=m_cfg.d_z,
            n_settle=m_cfg.n_settle,
            eta_z=m_cfg.eta_z,
            eta_D=m_cfg.eta_D,
            eta_E=m_cfg.eta_E,
            eta_R=m_cfg.eta_R,
            rho=m_cfg.rho,
            lambda_lat=m_cfg.lambda_lat,
            lambda_sparse=m_cfg.lambda_sparse,
            use_sparsity=False,
            use_memory=False,
            use_replay=False,
        )
    elif model_name == "agnis_kwta":
        return AgnisBaseline(
            d_in=d_in,
            d_out=d_out,
            d_z=m_cfg.d_z,
            k_sparse=m_cfg.k_sparse,
            n_settle=m_cfg.n_settle,
            eta_z=m_cfg.eta_z,
            eta_D=m_cfg.eta_D,
            eta_E=m_cfg.eta_E,
            eta_R=m_cfg.eta_R,
            rho=m_cfg.rho,
            lambda_lat=m_cfg.lambda_lat,
            lambda_sparse=m_cfg.lambda_sparse,
            use_sparsity=True,
            use_memory=False,
            use_replay=False,
        )
    elif model_name == "agnis_memory":
        return AgnisBaseline(
            d_in=d_in,
            d_out=d_out,
            d_z=m_cfg.d_z,
            k_sparse=m_cfg.k_sparse,
            n_settle=m_cfg.n_settle,
            eta_z=m_cfg.eta_z,
            eta_D=m_cfg.eta_D,
            eta_E=m_cfg.eta_E,
            eta_R=m_cfg.eta_R,
            rho=m_cfg.rho,
            lambda_lat=m_cfg.lambda_lat,
            lambda_sparse=m_cfg.lambda_sparse,
            use_sparsity=True,
            use_memory=True,
            use_replay=False,
            memory_capacity=mem_cfg.capacity,
            write_error_threshold=mem_cfg.write_error_threshold,
            write_novelty_threshold=mem_cfg.write_novelty_threshold,
        )
    elif model_name in ["agnis_replay", "agnis_full_fixed"]:
        return AgnisBaseline(
            d_in=d_in,
            d_out=d_out,
            d_z=m_cfg.d_z,
            k_sparse=m_cfg.k_sparse,
            n_settle=m_cfg.n_settle,
            eta_z=m_cfg.eta_z,
            eta_D=m_cfg.eta_D,
            eta_E=m_cfg.eta_E,
            eta_R=m_cfg.eta_R,
            rho=m_cfg.rho,
            lambda_lat=m_cfg.lambda_lat,
            lambda_sparse=m_cfg.lambda_sparse,
            use_sparsity=True,
            use_memory=True,
            use_replay=True,
            memory_capacity=mem_cfg.capacity,
            write_error_threshold=mem_cfg.write_error_threshold,
            write_novelty_threshold=mem_cfg.write_novelty_threshold,
            replay_buffer_size=mem_cfg.replay_buffer_size,
            sleep_lr_scale=t_cfg.sleep_lr_scale,
            importance_protect_threshold=t_cfg.importance_protect_threshold,
            n_sleep_steps=t_cfg.n_sleep_steps,
            n_sleep_replay=t_cfg.n_sleep_replay,
        )
    elif model_name == "agnis_neurogenesis":
        print("[Benchmark] Warning: Agnis Neurogenesis is not implemented yet in v0.2. Falling back to Full Fixed.")
        return initialize_model("agnis_full_fixed", config, d_in, d_out)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = default_config()

    # 2. Handle smoke-test override
    if args.smoke:
        print("[Benchmark] Smoke test active. Overriding to minimal settings.")
        config.training.n_tasks = 3
        config.training.pairs_per_task = 2
        config.training.n_repeats_per_task = 2
        config.training.n_sleep_steps = 1
        config.training.n_sleep_replay = 2
        config.model.d_z = 8
        config.memory.capacity = 10
        config.memory.replay_buffer_size = 5

    # 2.5 Apply CLI overrides
    if args.write_threshold is not None:
        config.memory.write_error_threshold = args.write_threshold
    if args.novelty_threshold is not None:
        config.memory.write_novelty_threshold = args.novelty_threshold
    if args.latent_dim is not None:
        config.model.d_z = args.latent_dim
    if args.n_tasks is not None:
        config.training.n_tasks = args.n_tasks

    # 3. Handle capacity stress overrides/validation
    if args.condition == "capacity_stress":
        print("[Benchmark] Capacity Stress condition selected.")
        if config.model.d_z > 8:
            print(f"[Benchmark] Overriding latent dimension d_z from {config.model.d_z} to 4 for stress bottleneck.")
            config.model.d_z = 4
        # Validate task size constraints
        config.training.n_tasks = max(5, config.training.n_tasks)
        config.training.pairs_per_task = max(4, config.training.pairs_per_task)
        print(f"[Benchmark] Stress task structure: {config.training.n_tasks} tasks, {config.training.pairs_per_task} pairs per task.")

    # 4. Dry-run early exit
    if args.dry_run:
        print(f"\n=== Dry Run Report ===")
        print(f"Condition: {args.condition}")
        print(f"Model: {args.model}")
        print(f"Seed: {args.seed}")
        print(f"Num Tasks: {config.training.n_tasks}")
        print(f"Pairs per Task: {config.training.pairs_per_task}")
        print(f"Latent Dim (d_z): {config.model.d_z}")
        print(f"Sleep Replay: {config.training.n_sleep_replay} samples, {config.training.n_sleep_steps} steps")
        print(f"Output Path: {os.path.join(config.results_dir, args.condition, args.model, f'seed_{args.seed}')}")
        print(f"======================\n")
        sys.exit(0)

    start_time = time.time()

    # Dimensions
    d_in = config.model.d_in
    d_out = config.model.d_in  # keeping targets one-hot in the same space size

    # Overlapping context option (checks configuration or defaults to True)
    overlap_context = getattr(config.training, "overlap_context", True)

    # 5. Generate tasks
    tasks = generate_phase1_tasks(
        condition=args.condition,
        num_tasks=config.training.n_tasks,
        pairs_per_task=config.training.pairs_per_task,
        d_in=d_in,
        d_out=d_out,
        overlap_context=overlap_context,
        seed=args.seed,
    )

    # Adjust model input dimension if task contains context vectors
    d_context = len(tasks[0].context) if (tasks[0].context is not None) else 0
    d_in_model = d_in + d_context

    print(f"\n[Benchmark] Initializing model {args.model} (d_in_model={d_in_model}, d_out={d_out})")
    model = initialize_model(args.model, config, d_in_model, d_out)

    # Run logs
    prediction_errors = []
    active_fractions = []
    memory_usages = []
    accuracy_matrix = []

    # 6. Sequential Learning Loop
    accuracy_before_sleep = []
    accuracy_after_sleep = []

    for t_idx, task in enumerate(tasks):
        print(f"[Benchmark] Training Task {t_idx}: {task.name}...")
        if hasattr(model, "start_task"):
            model.start_task(t_idx)
        
        # Present pairs sequentially
        n_repeats = config.training.n_repeats_per_task
        for repeat in range(n_repeats):
            # Shuffle pairs for this repeat
            indices = list(range(config.training.pairs_per_task))
            random.shuffle(indices)
            
            for idx in indices:
                x = task.inputs[idx]
                y = task.targets[idx]
                
                # train step
                train_metrics = model.train_pair(x, y, task_context=task.context)
                
                # record step logs
                prediction_errors.append(train_metrics.get("error", 0.0))
                
                stats = model.get_stats()
                sparsity = stats.get("sparsity_level", 0.0)
                active_fractions.append(1.0 - sparsity)
                memory_usages.append(stats.get("memory_size", 0))

        # Evaluation before sleep
        row_before = []
        for eval_idx in range(config.training.n_tasks):
            if eval_idx > t_idx:
                row_before.append(None)
            else:
                eval_task = tasks[eval_idx]
                acc = evaluate_model_on_task(
                    model=model,
                    inputs=eval_task.inputs,
                    targets=eval_task.targets,
                    context=eval_task.context,
                )
                row_before.append(acc)
        accuracy_before_sleep.append(row_before)
        print(f"  Accuracies before sleep: {row_before}")

        # Sleep/consolidation phase
        if hasattr(model, "sleep"):
            print(f"[Benchmark] Sleep consolidation after Task {t_idx}...")
            model.sleep()

        # Evaluation after sleep
        row_after = []
        for eval_idx in range(config.training.n_tasks):
            if eval_idx > t_idx:
                row_after.append(None)
            else:
                eval_task = tasks[eval_idx]
                acc = evaluate_model_on_task(
                    model=model,
                    inputs=eval_task.inputs,
                    targets=eval_task.targets,
                    context=eval_task.context,
                )
                row_after.append(acc)
        accuracy_after_sleep.append(row_after)
        print(f"  Accuracies after sleep:  {row_after}")

    accuracy_matrix = accuracy_after_sleep
    runtime = time.time() - start_time

    # 7. Post-experiment representation and metrics compilation
    print("[Benchmark] Compiling final metrics...")
    random_accuracy = 1.0 / d_out
    cl_metrics_before = compute_phase1_metrics(accuracy_before_sleep, random_accuracy=random_accuracy)
    cl_metrics = compute_phase1_metrics(accuracy_matrix, random_accuracy=random_accuracy)

    # Prototypes and Overlaps
    prototypes = []
    for t_idx, task in enumerate(tasks):
        task_pairs = [(task.inputs[i], task.targets[i]) for i in range(task.inputs.shape[0])]
        proto = compute_task_prototypes(model, task_pairs, task.context)
        prototypes.append(proto)

    overlap_tensor = compute_pairwise_overlap(prototypes)
    overlap_matrix = overlap_tensor.tolist()
    mean_overlap = overlap_tensor.mean().item()

    # Interference score on the last task
    interference_val = 0.0
    if len(prototypes) > 1:
        interference_val = compute_interference_score(model, prototypes[-1])

    # Final stats
    final_stats = model.get_stats()
    
    # 8. Build full results schema
    results_schema = {
        "condition": args.condition,
        "model": args.model,
        "seed": args.seed,
        "num_tasks": config.training.n_tasks,
        "pairs_per_task": config.training.pairs_per_task,
        "latent_dim": config.model.d_z,
        "accuracy_matrix": accuracy_matrix,
        "final_average_accuracy": cl_metrics.get("final_average_accuracy", 0.0),
        "average_forgetting": cl_metrics.get("average_forgetting", 0.0),
        "forgetting_per_task": cl_metrics.get("forgetting_per_task", {}),
        "backward_transfer": cl_metrics.get("backward_transfer", 0.0),
        "forward_transfer": cl_metrics.get("forward_transfer", 0.0),
        "mean_prediction_error": sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0,
        "final_prediction_error": prediction_errors[-1] if prediction_errors else 0.0,
        "mean_active_fraction": sum(active_fractions) / len(active_fractions) if active_fractions else 0.0,
        "final_memory_size": final_stats.get("memory_size", 0),
        "memory_hit_rate": final_stats.get("memory_hit_rate", 0.0),
        "representation_overlap_mean": mean_overlap,
        "interference_score": interference_val,
        "runtime_seconds": runtime,
        # New Diagnostic Metrics
        "accuracy_before_sleep": accuracy_before_sleep,
        "accuracy_after_sleep": accuracy_after_sleep,
        "forgetting_before_sleep": cl_metrics_before.get("average_forgetting", 0.0),
        "forgetting_after_sleep": cl_metrics.get("average_forgetting", 0.0),
        "replay_benefit": cl_metrics.get("final_average_accuracy", 0.0) - cl_metrics_before.get("final_average_accuracy", 0.0),
        "memory_writes_per_task": final_stats.get("memory_writes_per_task", []),
        "memory_retrievals_per_task": final_stats.get("memory_retrievals_per_task", []),
        "memory_hits_per_task": final_stats.get("memory_hits_per_task", []),
        "mean_retrieval_similarity": final_stats.get("mean_retrieval_similarity", 0.0),
        "replay_steps_executed": final_stats.get("replay_steps_executed", 0),
        "replay_error_delta": final_stats.get("replay_error_delta", 0.0),
        # Metadata configuration parameters
        "uses_context": overlap_context if args.condition == "overlapping" else False,
        "input_dim": d_in,
        "target_dim": d_out,
        "joint_dim": d_in_model + d_out,
        "kwta_k": config.model.k_sparse,
        "memory_capacity": config.memory.capacity,
        "train_steps_per_pair": config.training.n_repeats_per_task,
        "sleep_steps_per_task": config.training.n_sleep_steps,
        "evaluation_mode": "no_update",
        "write_error_threshold": config.memory.write_error_threshold,
        "write_novelty_threshold": config.memory.write_novelty_threshold,
        "sensitivity_mode": args.sensitivity_mode,
        "run_id": f"{args.condition}_{args.model}_write_{format_threshold(config.memory.write_error_threshold)}_novelty_{format_threshold(config.memory.write_novelty_threshold)}_seed_{args.seed}",
    }

    # 9. Save files
    run_dir = save_phase1_run_results(
        results_dir=config.results_dir,
        condition=args.condition,
        model_name=args.model,
        seed=args.seed,
        metrics_dict=results_schema,
        accuracy_matrix=accuracy_matrix,
        forgetting_list=list(cl_metrics.get("forgetting_per_task", {}).values()),
        prediction_errors=prediction_errors,
        active_fractions=active_fractions,
        memory_usages=memory_usages,
        overlap_matrix=overlap_matrix,
        config_data=config.to_dict(),
    )

    # 10. Generate plots
    generate_all_plots(
        run_dir=run_dir,
        accuracy_matrix=accuracy_matrix,
        forgetting_list=list(cl_metrics.get("forgetting_per_task", {}).values()),
        prediction_errors=prediction_errors,
        active_fractions=active_fractions,
        memory_usages=memory_usages,
        overlap_matrix=overlap_matrix,
    )

    print(f"\n[Benchmark] Run finished successfully. Results saved in {run_dir}\n")


if __name__ == "__main__":
    main()
