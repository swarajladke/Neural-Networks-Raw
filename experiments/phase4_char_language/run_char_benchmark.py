"""
Raw AGNIS — experiments/phase4_char_language/run_char_benchmark.py

Benchmark runner for Phase 4: Character-Level Continual Language.
"""

import argparse
import os
import sys
import json
import yaml
import time
import math
import csv
from typing import List, Dict, Any

import torch

# Add src/ to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from agnis.text import CharVocab, CharacterStream, CharMetrics, get_all_domains
from agnis.text.char_metrics import compute_forgetting, compute_bpc_forgetting, compute_growth_efficiency
from agnis.sequence.sequence_wrapper import (
    SeqAgnisModel, SimpleRNNBaseline, BigramBaseline, TrigramBaseline
)
from agnis.neurogenesis.growth_controller import GrowthController
from agnis.utils.config import load_config, AGNISConfig


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 Character-Level Continual Language Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        choices=['seq_agnis_fixed', 'seq_agnis_neurogenesis',
                                 'seq_agnis_neuro_no_maturity', 'seq_agnis_neuro_no_pruning',
                                 'seq_agnis_no_replay', 'rnn_baseline', 'bigram_baseline', 'trigram_baseline'],
                        help="Model variant to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/kaggle_phase4.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--domain-order", type=str, default="prose,code,arithmetic,dialogue",
                        help="Comma-separated task order")
    args = parser.parse_args()

    # Load config and set random seeds
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    config = load_config(args.config)
    
    # Override seed if provided
    config.training.seed = args.seed
    torch.manual_seed(args.seed)
    
    # Setup results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Initialize CharVocab
    vocab = CharVocab()
    vocab_size = vocab.vocab_size
    d_symbol = vocab_size  # d_in, d_out are d_symbol
    
    # Reorder domains based on domain-order argument
    domain_order = [d.strip() for d in args.domain_order.split(',')]
    all_domains_dict = {d.name: d for d in get_all_domains()}
    domains = [all_domains_dict[name] for name in domain_order]
    
    # Set parameters based on smoke flag or config
    if args.smoke:
        print("[Phase4] Running in SMOKE mode.")
        train_chars = 200
        eval_chars = 100
        epochs_per_domain = 1
        domains = domains[:2]  # run only first 2 domains
        log_every = 50
    else:
        train_chars = getattr(config.training, "train_chars_per_domain", 2000)
        eval_chars = getattr(config.training, "eval_chars_per_domain", 500)
        epochs_per_domain = getattr(config.training, "epochs_per_domain", 2)
        log_every = config.training.log_every

    n_tasks = len(domains)
    print(f"[Phase4] Model: {args.model} | Seed: {args.seed}")
    print(f"[Phase4] Domain order: {[d.name for d in domains]}")
    print(f"[Phase4] Vocab size: {vocab_size} | Training chars/task: {train_chars} | Epochs: {epochs_per_domain}")

    # Instantiate Model
    model_key = args.model
    if model_key in ['seq_agnis_fixed', 'seq_agnis_neurogenesis',
                     'seq_agnis_neuro_no_maturity', 'seq_agnis_neuro_no_pruning', 'seq_agnis_no_replay']:
        use_memory = (model_key != 'seq_agnis_no_replay')
        use_replay = (model_key != 'seq_agnis_no_replay')
        maturity_enabled = (model_key != 'seq_agnis_neuro_no_maturity')
        
        model = SeqAgnisModel(
            d_in=d_symbol,
            d_out=d_symbol,
            d_z=config.model.d_z,
            config=config,
            use_memory=use_memory,
            use_replay=use_replay,
            maturity_enabled=maturity_enabled,
            max_latent_dim=config.neurogenesis.max_units
        )
    elif model_key == 'rnn_baseline':
        model = SimpleRNNBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z)
    elif model_key == 'bigram_baseline':
        model = BigramBaseline(vocab_size=d_symbol)
    elif model_key == 'trigram_baseline':
        model = TrigramBaseline(vocab_size=d_symbol)
    else:
        raise ValueError(f"Unknown model variant: {model_key}")

    # Growth Controller for Neurogenesis models
    is_neurogenesis_model = model_key in ['seq_agnis_neurogenesis', 'seq_agnis_neuro_no_maturity', 'seq_agnis_neuro_no_pruning']
    if is_neurogenesis_model:
        gc = GrowthController(
            alpha=config.neurogenesis.alpha,
            beta=config.neurogenesis.beta,
            gamma=config.neurogenesis.gamma,
            delta=config.neurogenesis.delta,
            kappa=config.neurogenesis.kappa,
            lambda_cost=config.neurogenesis.lambda_cost,
            threshold=config.neurogenesis.threshold,
            consecutive_n=config.neurogenesis.consecutive_n
        )
    else:
        gc = None

    # Trackers
    accuracy_matrix_before = []
    accuracy_matrix_after = []
    bpc_matrix_before = []
    bpc_matrix_after = []
    
    capacity_timeline = []  # List of tuples (step, capacity)
    training_errors = []    # List of float prediction errors
    
    total_births = 0
    total_prunes = 0
    start_time = time.time()
    
    step_counter = 0

    # Main Task Loop
    for t_idx, d in enumerate(domains):
        print(f"\n[Phase4] --- Training Domain {t_idx}: {d.name} ---")
        model.start_task(t_idx)
        
        # Training loop
        for epoch in range(epochs_per_domain):
            text = d.generate(train_chars, seed=args.seed + epoch)
            stream = CharacterStream(text, vocab)
            model.reset_sequence_state()
            epoch_errors = []
            
            for x, y, x_idx, y_idx in stream:
                step_counter += 1
                metrics = model.train_transition(x, y)
                error = metrics.get('error', 0.0)
                epoch_errors.append(error)
                training_errors.append(error)
                
                # Growth check
                if is_neurogenesis_model:
                    current_capacity = model.base_model.cell.d_z
                    novelty = 1.0 - model.base_model.last_retrieval_similarity
                    error_l2 = math.sqrt(error * 2 * d_symbol)
                    
                    trigger = gc.update(
                        error=error_l2,
                        novelty=novelty,
                        uncertainty=0.0,
                        interference=0.0,
                        coverage=0.0,
                        cost=float(current_capacity),
                    )
                    
                    if trigger and current_capacity < config.neurogenesis.max_units:
                        # Grow units
                        joint_input = torch.cat([x, y])
                        residual = model.base_model.cell._last_error
                        model.base_model.cell.grow_units(
                            k=2,
                            current_input=joint_input,
                            residual_error=residual
                        )
                        total_births += 2
                        print(f"  [Neurogenesis] Spawning 2 new units. Capacity: {current_capacity} -> {current_capacity + 2}")
                
                if hasattr(model, 'base_model'):
                    capacity_timeline.append((step_counter, model.base_model.cell.d_z))
                else:
                    capacity_timeline.append((step_counter, config.model.d_z))

                if step_counter % log_every == 0:
                    cap_str = f" | Cap: {model.base_model.cell.d_z}" if hasattr(model, 'base_model') else ""
                    print(f"  Step {step_counter} | error_MSE: {error:.4f}{cap_str}")
            
            mean_epoch_err = sum(epoch_errors) / len(epoch_errors) if epoch_errors else 0.0
            print(f"  Epoch {epoch} finished. Mean MSE: {mean_epoch_err:.4f}")

        # Evaluate before sleep (on ALL tasks seen so far or all tasks in benchmark)
        print(f"[Phase4] Evaluating all domains BEFORE sleep...")
        row_before = []
        bpc_before = []
        
        for eval_idx, eval_domain in enumerate(domains):
            eval_text = eval_domain.get_eval_text(eval_chars, seed=args.seed)
            eval_stream = CharacterStream(eval_text, vocab)
            
            metrics_eval = CharMetrics(vocab_size=d_symbol)
            model.reset_sequence_state()
            
            for x, y, x_idx, y_idx in eval_stream:
                # Evaluation Mode: no side effects
                pred = model.predict_no_state_update(x)
                metrics_eval.update(pred, y_idx)
                model.advance_state_only(x, y)
                
            row_before.append(metrics_eval.accuracy)
            bpc_before.append(metrics_eval.bpc)
            
        print(f"  Accuracy (before sleep): {['{:.3f}'.format(a) for a in row_before]}")
        print(f"  BPC (before sleep):      {['{:.3f}'.format(b) for b in bpc_before]}")
        
        accuracy_matrix_before.append(row_before)
        bpc_matrix_before.append(bpc_before)

        # Sleep / Consolidation
        if hasattr(model, "sleep") and model_key not in ['bigram_baseline', 'trigram_baseline']:
            print(f"[Phase4] Sleep consolidation after Domain {t_idx}...")
            model.sleep()

        # Pruning boundary (only if enabled)
        if model_key in ["seq_agnis_neurogenesis", "seq_agnis_neuro_no_maturity"] and not args.smoke:
            initial_cap = model.base_model.cell.d_z
            # Conservative pruning: min_age = 50, usage_threshold = 0.01, importance = 0.01, maturity = 0.5
            model.base_model.cell.prune_units(
                min_age=50,
                usage_threshold=config.neurogenesis.usage_threshold,
                importance_threshold=config.neurogenesis.importance_threshold,
                maturity_threshold=0.5
            )
            final_cap = model.base_model.cell.d_z
            if final_cap < initial_cap:
                pruned = initial_cap - final_cap
                total_prunes += pruned
                print(f"  [Neurogenesis] Pruned {pruned} units. Capacity: {initial_cap} -> {final_cap}")

        # Evaluate after sleep
        print(f"[Phase4] Evaluating all domains AFTER sleep...")
        row_after = []
        bpc_after = []
        
        for eval_idx, eval_domain in enumerate(domains):
            eval_text = eval_domain.get_eval_text(eval_chars, seed=args.seed)
            eval_stream = CharacterStream(eval_text, vocab)
            
            metrics_eval = CharMetrics(vocab_size=d_symbol)
            model.reset_sequence_state()
            
            for x, y, x_idx, y_idx in eval_stream:
                pred = model.predict_no_state_update(x)
                metrics_eval.update(pred, y_idx)
                model.advance_state_only(x, y)
                
            row_after.append(metrics_eval.accuracy)
            bpc_after.append(metrics_eval.bpc)
            
        print(f"  Accuracy (after sleep):  {['{:.3f}'.format(a) for a in row_after]}")
        print(f"  BPC (after sleep):       {['{:.3f}'.format(b) for b in bpc_after]}")
        
        accuracy_matrix_after.append(row_after)
        bpc_matrix_after.append(bpc_after)

    # Compute aggregation statistics
    runtime = time.time() - start_time
    
    # Calculate forgetting
    forgetting = compute_forgetting(accuracy_matrix_after)
    bpc_forgetting = compute_bpc_forgetting(bpc_matrix_after)
    
    final_row_acc = accuracy_matrix_after[-1]
    final_row_bpc = bpc_matrix_after[-1]
    
    mean_final_acc = sum(final_row_acc) / n_tasks
    mean_final_bpc = sum(final_row_bpc) / n_tasks
    mean_forgetting_acc = sum(forgetting) / n_tasks
    mean_forgetting_bpc = sum(bpc_forgetting) / n_tasks

    # Calculate growth efficiency on the final task
    initial_dz = config.model.d_z
    final_dz = model.base_model.cell.d_z if hasattr(model, 'base_model') else initial_dz
    dz_increase = final_dz - initial_dz
    
    efficiency_acc = compute_growth_efficiency(
        accuracy_matrix_after[0][0],
        accuracy_matrix_after[-1][0],
        total_births
    )
    # BPC gain per unit (decrease is good, so invert difference)
    efficiency_bpc = 0.0
    if total_births > 0:
        efficiency_bpc = (bpc_matrix_after[0][0] - bpc_matrix_after[-1][0]) / total_births

    # Sleep/replay benefit on final evaluation
    rep_ben_acc = 0.0
    rep_ben_bpc = 0.0
    
    # Average change due to sleep consolidation across all runs
    all_before_acc = [val for row in accuracy_matrix_before for val in row]
    all_after_acc = [val for row in accuracy_matrix_after for val in row]
    rep_ben_acc = sum(a - b for a, b in zip(all_after_acc, all_before_acc)) / len(all_before_acc)

    all_before_bpc = [val for row in bpc_matrix_before for val in row]
    all_after_bpc = [val for row in bpc_matrix_after for val in row]
    rep_ben_bpc = sum(b - a for a, b in zip(all_after_bpc, all_before_bpc)) / len(all_before_bpc)

    # Memory statistics
    mem_hit_rate = 0.0
    if hasattr(model, 'base_model') and model.base_model.fast_mem:
        # Number of memory hits / total checks
        # Blended memory queries are retrieved if similarity is above 0.95
        pass

    results_dir = os.path.join(config.results_dir, f"{args.model}", f"seed_{args.seed}")
    os.makedirs(results_dir, exist_ok=True)
    
    metrics = {
        "phase": "phase4_char_language",
        "model": args.model,
        "seed": args.seed,
        "domain_order": domain_order,
        "vocab_size": vocab_size,
        "cell_input_dim": 2 * vocab_size,
        "initial_d_z": initial_dz,
        "final_d_z": final_dz,
        "max_units": config.neurogenesis.max_units,
        "train_chars_per_domain": train_chars,
        "eval_chars_per_domain": eval_chars,
        "epochs_per_domain": epochs_per_domain,
        "final_average_accuracy": mean_final_acc,
        "final_average_bpc": mean_final_bpc,
        "average_accuracy_forgetting": mean_forgetting_acc,
        "average_bpc_forgetting": mean_forgetting_bpc,
        "total_units_born": total_births,
        "total_units_pruned": total_prunes,
        "growth_efficiency_acc_per_unit": efficiency_acc,
        "growth_efficiency_bpc_per_unit": efficiency_bpc,
        "replay_benefit_accuracy": rep_ben_acc,
        "replay_benefit_bpc": rep_ben_bpc,
        "runtime_seconds": runtime
    }

    # Save to files
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Write accuracy matrix
    with open(os.path.join(results_dir, "accuracy_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrainDomain"] + [d.name for d in domains])
        for idx, row in enumerate(accuracy_matrix_after):
            writer.writerow([domains[idx].name] + row)
            
    # Write BPC matrix
    with open(os.path.join(results_dir, "bpc_matrix.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TrainDomain"] + [d.name for d in domains])
        for idx, row in enumerate(bpc_matrix_after):
            writer.writerow([domains[idx].name] + row)

    # Write capacity timeline
    with open(os.path.join(results_dir, "capacity_timeline.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Capacity"])
        writer.writerows(capacity_timeline)

    # Write training errors
    with open(os.path.join(results_dir, "training_error.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "PredictionError"])
        for idx, err in enumerate(training_errors):
            writer.writerow([idx, err])

    # Save config used
    with open(os.path.join(results_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(config_data, f)
        
    # Save qualitative samples
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Generate simple test sample
    for d in domains:
        with open(os.path.join(samples_dir, f"after_{d.name}.txt"), "w") as f:
            f.write(f"# Qualitive sample of generated/true text from domain {d.name}\n")
            f.write(d.generate(100, seed=args.seed + 999))

    print(f"\n[Phase4] Run finished successfully. Results saved in {results_dir}")


if __name__ == "__main__":
    main()
