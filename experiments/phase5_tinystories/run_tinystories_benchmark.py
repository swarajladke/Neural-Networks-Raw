"""
Raw AGNIS — experiments/phase5_tinystories/run_tinystories_benchmark.py

Benchmark runner for Phase 5: TinyStories Mini conditional generation.
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

from agnis.text import CharVocab, CharacterStream, CharMetrics
from agnis.text.story_domains import get_all_story_domains
from agnis.text.conditional_generation import generate_continuation
from agnis.text.generation_metrics import (
    compute_repetition_rate,
    compute_keyword_retention,
    compute_name_consistency,
    compute_sentence_completion,
    compute_distinct_n
)
from agnis.text.char_metrics import compute_forgetting, compute_bpc_forgetting, compute_growth_efficiency
from agnis.sequence.sequence_wrapper import (
    SeqAgnisModel, SimpleRNNBaseline, SimpleGRUBaseline, BigramBaseline, TrigramBaseline
)
from agnis.neurogenesis.growth_controller import GrowthController
from agnis.utils.config import load_config, AGNISConfig


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 TinyStories Mini Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        choices=['seq_agnis_fixed', 'seq_agnis_neurogenesis',
                                 'seq_agnis_neuro_no_maturity', 'seq_agnis_neuro_no_pruning',
                                 'seq_agnis_no_replay', 'rnn_baseline', 'gru_baseline', 'bigram_baseline', 'trigram_baseline'],
                        help="Model variant to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/kaggle_phase5.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--domain-order", type=str, default="animals,objects,emotions,actions",
                        help="Comma-separated task order")
    args = parser.parse_args()

    # Load config and set random seeds
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    config = load_config(args.config)
    
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
    all_domains_dict = {d.name: d for d in get_all_story_domains()}
    domains = [all_domains_dict[name] for name in domain_order]
    
    # Set parameters based on smoke flag or config
    if args.smoke:
        print("[Phase5] Running in SMOKE mode.")
        stories_per_domain = 5
        train_chars = 500
        eval_prompts = 3
        epochs_per_domain = 1
        domains = domains[:2]  # run only first 2 domains
        log_every = 50
    else:
        stories_per_domain = config.training.stories_per_domain
        train_chars = config.training.train_chars_per_domain
        eval_prompts = config.training.eval_prompts_per_domain
        epochs_per_domain = config.training.epochs_per_domain
        log_every = config.training.log_every

    n_tasks = len(domains)
    print(f"[Phase5] Model: {args.model} | Seed: {args.seed}")
    print(f"[Phase5] Domain order: {[d.name for d in domains]}")
    print(f"[Phase5] Vocab size: {vocab_size} | Stories/task: {stories_per_domain} | Epochs: {epochs_per_domain}")

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
    elif model_key == 'gru_baseline':
        model = SimpleGRUBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z)
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
    
    # Story-level quality trackers (post-sleep)
    repetition_rates = []
    keyword_retentions = []
    name_consistencies = []
    sentence_completions = []
    distinct_2_rates = []
    distinct_3_rates = []
    maturity_history = []
    
    capacity_timeline = []
    training_errors = []
    
    total_births = 0
    total_prunes = 0
    start_time = time.time()
    
    step_counter = 0

    results_dir = os.path.join(config.results_dir, f"{args.model}", f"seed_{args.seed}")
    os.makedirs(results_dir, exist_ok=True)
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # Main Task Loop
    for t_idx, d in enumerate(domains):
        print(f"\n[Phase5] --- Training Domain {t_idx}: {d.name} ---")
        model.start_task(t_idx)
        
        # Generate domain training stories
        train_stories = d.generate_stories(stories_per_domain, seed=args.seed)
        
        # Training loop
        for epoch in range(epochs_per_domain):
            epoch_errors = []
            
            for story_idx, story in enumerate(train_stories):
                # Target is continuation only; prompt + target is full text
                text = story["prompt"] + story["target"]
                stream = CharacterStream(text, vocab)
                model.reset_sequence_state()
                
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

        # Evaluate before sleep
        print(f"[Phase5] Evaluating continuation quality BEFORE sleep...")
        row_before = []
        bpc_before = []
        
        for eval_idx, eval_domain in enumerate(domains):
            eval_stories = eval_domain.get_eval_stories(eval_prompts, seed=args.seed)
            metrics_eval = CharMetrics(vocab_size=d_symbol)
            
            for story in eval_stories:
                prompt = story["prompt"]
                target = story["target"]
                
                # 1. Condition on prompt
                model.reset_sequence_state()
                for i in range(len(prompt) - 1):
                    model.advance_state_only(vocab.to_onehot(prompt[i]), vocab.to_onehot(prompt[i+1]))
                    
                # 2. Scored evaluation on target continuation (teacher-forced)
                current_char = prompt[-1]
                for true_next_char in target:
                    current_oh = vocab.to_onehot(current_char)
                    true_next_oh = vocab.to_onehot(true_next_char)
                    
                    pred = model.predict_no_state_update(current_oh)
                    metrics_eval.update(pred, vocab.encode(true_next_char))
                    
                    model.advance_state_only(current_oh, true_next_oh)
                    current_char = true_next_char
                    
            row_before.append(metrics_eval.accuracy)
            bpc_before.append(metrics_eval.bpc)
            
        print(f"  Continuation Accuracy (before sleep): {['{:.3f}'.format(a) for a in row_before]}")
        print(f"  Continuation BPC (before sleep):      {['{:.3f}'.format(b) for b in bpc_before]}")
        
        accuracy_matrix_before.append(row_before)
        bpc_matrix_before.append(bpc_before)

        # Sleep / Replay
        if hasattr(model, "sleep") and model_key not in ['bigram_baseline', 'trigram_baseline']:
            print(f"[Phase5] Sleep consolidation after Domain {t_idx}...")
            model.sleep()

        # Pruning boundary (only if enabled)
        if model_key in ["seq_agnis_neurogenesis", "seq_agnis_neuro_no_maturity"] and not args.smoke:
            initial_cap = model.base_model.cell.d_z
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

        # Log maturity distribution at the end of task training
        if hasattr(model, 'base_model') and hasattr(model.base_model.cell, 'maturity'):
            maturity_history.append(model.base_model.cell.maturity.tolist())
        else:
            maturity_history.append([])

        # Evaluate after sleep
        print(f"[Phase5] Evaluating continuation quality AFTER sleep...")
        row_after = []
        bpc_after = []
        
        task_repetition = []
        task_keywords = []
        task_names = []
        task_completion = []
        task_distinct_2 = []
        task_distinct_3 = []
        
        for eval_idx, eval_domain in enumerate(domains):
            eval_stories = eval_domain.get_eval_stories(eval_prompts, seed=args.seed)
            metrics_eval = CharMetrics(vocab_size=d_symbol)
            
            # Sub-trackers for generated samples
            rep_acc = 0.0
            key_acc = 0.0
            name_acc = 0.0
            comp_acc = 0.0
            dist2_acc = 0.0
            dist3_acc = 0.0
            
            # Open files to save text samples for this task boundary
            # If final task boundary (t_idx == n_tasks - 1), it saves the final completions
            save_samples = (t_idx == n_tasks - 1) or (t_idx == eval_idx)
            if save_samples:
                samples_file_name = f"{eval_domain.name}_after_task_{t_idx}.txt" if t_idx != n_tasks - 1 else f"{eval_domain.name}_after_final.txt"
                sf = open(os.path.join(samples_dir, samples_file_name), "w", encoding="utf-8")
                sf.write(f"# Evaluation samples for Domain: {eval_domain.name} after Training Task: {domains[t_idx].name}\n\n")
            
            for s_i, story in enumerate(eval_stories):
                prompt = story["prompt"]
                target = story["target"]
                
                # Scored evaluation
                model.reset_sequence_state()
                for i in range(len(prompt) - 1):
                    model.advance_state_only(vocab.to_onehot(prompt[i]), vocab.to_onehot(prompt[i+1]))
                    
                current_char = prompt[-1]
                for true_next_char in target:
                    current_oh = vocab.to_onehot(current_char)
                    true_next_oh = vocab.to_onehot(true_next_char)
                    pred = model.predict_no_state_update(current_oh)
                    metrics_eval.update(pred, vocab.encode(true_next_char))
                    model.advance_state_only(current_oh, true_next_oh)
                    current_char = true_next_char
                
                # Text generation (greedy completion)
                if save_samples and s_i < min(10, eval_prompts):
                    gen_text = generate_continuation(
                        model=model,
                        prompt=prompt,
                        vocab=vocab,
                        max_chars=config.generation.max_chars,
                        decoding=config.generation.decoding,
                        temperature=config.generation.temperature,
                        top_k=config.generation.top_k,
                        stop_on_double_newline=config.generation.stop_on_double_newline
                    )
                    
                    # Compute story quality scores
                    rep_acc += compute_repetition_rate(gen_text, n=config.generation.repetition_ngram_n)
                    key_acc += compute_keyword_retention(gen_text, eval_domain.name)
                    name_acc += compute_name_consistency(gen_text)
                    comp_acc += compute_sentence_completion(gen_text)
                    dist2_acc += compute_distinct_n(gen_text, n=2)
                    dist3_acc += compute_distinct_n(gen_text, n=3)
                    
                    if s_i < 3: # Save first 3 stories as sample references
                        sf.write(f"PROMPT: {prompt}\n")
                        sf.write(f"TARGET: {target.strip()}\n")
                        sf.write(f"GENERATED: {gen_text.strip()}\n")
                        sf.write(f"METRICS:\n")
                        sf.write(f"  Repetition Rate: {compute_repetition_rate(gen_text, n=config.generation.repetition_ngram_n):.3f}\n")
                        sf.write(f"  Keyword Retention: {compute_keyword_retention(gen_text, eval_domain.name):.3f}\n")
                        sf.write(f"  Name Consistency: {compute_name_consistency(gen_text):.3f}\n")
                        sf.write(f"  Sentence Completion: {compute_sentence_completion(gen_text):.3f}\n")
                        sf.write(f"  Distinct-2: {compute_distinct_n(gen_text, n=2):.3f}\n")
                        sf.write(f"  Distinct-3: {compute_distinct_n(gen_text, n=3):.3f}\n\n")
                        sf.write("-" * 40 + "\n\n")
            
            if save_samples:
                sf.close()
                
            n_gen_total = min(10, eval_prompts) if save_samples else 1
            task_repetition.append(rep_acc / n_gen_total)
            task_keywords.append(key_acc / n_gen_total)
            task_names.append(name_acc / n_gen_total)
            task_completion.append(comp_acc / n_gen_total)
            task_distinct_2.append(dist2_acc / n_gen_total)
            task_distinct_3.append(dist3_acc / n_gen_total)
            
            row_after.append(metrics_eval.accuracy)
            bpc_after.append(metrics_eval.bpc)
            
        print(f"  Continuation Accuracy (after sleep):  {['{:.3f}'.format(a) for a in row_after]}")
        print(f"  Continuation BPC (after sleep):       {['{:.3f}'.format(b) for b in bpc_after]}")
        print(f"  Repetition rates:                    {['{:.3f}'.format(r) for r in task_repetition]}")
        print(f"  Keyword retentions:                  {['{:.3f}'.format(k) for k in task_keywords]}")
        print(f"  Distinct-2 rates:                    {['{:.3f}'.format(d2) for d2 in task_distinct_2]}")
        
        accuracy_matrix_after.append(row_after)
        bpc_matrix_after.append(bpc_after)
        
        repetition_rates.append(task_repetition)
        keyword_retentions.append(task_keywords)
        name_consistencies.append(task_names)
        sentence_completions.append(task_completion)
        distinct_2_rates.append(task_distinct_2)
        distinct_3_rates.append(task_distinct_3)

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

    # Calculate average generated story metrics on the final task boundary
    final_repetition = sum(repetition_rates[-1]) / n_tasks
    final_keywords = sum(keyword_retentions[-1]) / n_tasks
    final_names = sum(name_consistencies[-1]) / n_tasks
    final_completion = sum(sentence_completions[-1]) / n_tasks
    final_distinct_2 = sum(distinct_2_rates[-1]) / n_tasks
    final_distinct_3 = sum(distinct_3_rates[-1]) / n_tasks

    initial_dz = config.model.d_z
    final_dz = model.base_model.cell.d_z if hasattr(model, 'base_model') else initial_dz
    
    # Sleep/replay benefit
    all_before_acc = [val for row in accuracy_matrix_before for val in row]
    all_after_acc = [val for row in accuracy_matrix_after for val in row]
    rep_ben_acc = sum(a - b for a, b in zip(all_after_acc, all_before_acc)) / len(all_before_acc)

    all_before_bpc = [val for row in bpc_matrix_before for val in row]
    all_after_bpc = [val for row in bpc_matrix_after for val in row]
    rep_ben_bpc = sum(b - a for a, b in zip(all_after_bpc, all_before_bpc)) / len(all_before_bpc)

    metrics = {
        "phase": "phase5_tinystories_mini",
        "model": args.model,
        "seed": args.seed,
        "domain_order": domain_order,
        "num_domains": n_tasks,
        "stories_per_domain": stories_per_domain,
        "eval_prompts_per_domain": eval_prompts,
        "final_average_continuation_accuracy": mean_final_acc,
        "final_average_continuation_bpc": mean_final_bpc,
        "continuation_accuracy_forgetting": mean_forgetting_acc,
        "continuation_bpc_forgetting": mean_forgetting_bpc,
        "average_repetition_rate": final_repetition,
        "average_keyword_retention": final_keywords,
        "average_name_consistency": final_names,
        "average_sentence_completion_rate": final_completion,
        "average_distinct_2": final_distinct_2,
        "average_distinct_3": final_distinct_3,
        "initial_d_z": initial_dz,
        "final_d_z": final_dz,
        "total_units_born": total_births,
        "total_units_pruned": total_prunes,
        "replay_benefit_accuracy": rep_ben_acc,
        "replay_benefit_bpc": rep_ben_bpc,
        "maturity_history": maturity_history,
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

    print(f"\n[Phase5] Run finished successfully. Results saved in {results_dir}")


if __name__ == "__main__":
    main()
