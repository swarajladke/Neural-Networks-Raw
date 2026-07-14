"""
Raw AGNIS — experiments/phase6_deep_stack/run_deep_benchmark.py

Benchmark runner for Phase 6: Deep Hierarchical Predictive Coding (Deep AGNIS).
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
from agnis.text.char_metrics import (
    compute_forgetting, compute_bpc_forgetting, compute_growth_efficiency,
    summarize_learning_vs_forgetting,
)
from agnis.sequence.sequence_wrapper import (
    SeqAgnisModel, DeepSeqAgnisModel, SimpleRNNBaseline, SimpleGRUBaseline,
    RNNReplayBaseline, GRUReplayBaseline, RNNEWCBaseline, GRUEWCBaseline,
    BigramBaseline, TrigramBaseline
)
from agnis.evaluation.probes import WordBoundaryProbe
from agnis.utils.config import load_config, AGNISConfig


def main():
    parser = argparse.ArgumentParser(description="Run Phase 6 Deep AGNIS Benchmark")
    parser.add_argument("--model", type=str, required=True,
                        choices=['gru_baseline', 'rnn_baseline', 'rnn_replay_baseline', 'gru_replay_baseline',
                                 'rnn_ewc_baseline', 'gru_ewc_baseline', 'seq_agnis_flat_wide',
                                 'deep_agnis_2L', 'deep_agnis_3L', 'deep_agnis_3L_neurogenesis',
                                 'sparc_task_id_oracle', 'sparc_nearest_prototype'],
                        help="Model variant to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/kaggle_phase6.yaml", help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test")
    parser.add_argument("--domain-order", type=str, default="animals,objects,emotions,actions",
                        help="Comma-separated task order")
    args = parser.parse_args()

    # Load config and set random seeds
    config = load_config(args.config)
    config.training.seed = args.seed
    torch.manual_seed(args.seed)

    # Initialize CharVocab
    vocab = CharVocab()
    vocab_size = vocab.vocab_size
    d_symbol = vocab_size

    # Reorder domains based on domain-order argument
    domain_order = [d.strip() for d in args.domain_order.split(',')]
    all_domains_dict = {d.name: d for d in get_all_story_domains()}
    domains = [all_domains_dict[name] for name in domain_order]

    # Set parameters based on smoke flag or config
    if args.smoke:
        print("[Phase6] Running in SMOKE mode.")
        stories_per_domain = 2
        train_chars = 200
        eval_prompts = 2
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
    print(f"[Phase6] Model: {args.model} | Seed: {args.seed}")
    print(f"[Phase6] Domain order: {[d.name for d in domains]}")
    print(f"[Phase6] Vocab size: {vocab_size} | Stories/task: {stories_per_domain} | Epochs: {epochs_per_domain}")

    # Configure hierarchy settings based on model variant
    model_key = args.model
    if model_key == 'seq_agnis_flat_wide':
        # Parameter-matched control (112 flat units)
        config.hierarchy.enabled = False
        config.model.d_z = 112
        config.model.k_sparse = 11
        config.neurogenesis.enabled = False
        model = SeqAgnisModel(
            d_in=d_symbol,
            d_out=d_symbol,
            d_z=112,
            config=config,
            use_memory=True,
            use_replay=True,
            maturity_enabled=False,
            max_latent_dim=112,
        )
    elif model_key == 'deep_agnis_2L':
        config.hierarchy.enabled = True
        config.hierarchy.layer_dims = [64, 32]
        config.hierarchy.k_sparse_per_layer = [8, 4]
        config.hierarchy.commit_strides = [1, 4]
        config.hierarchy.max_units_per_layer = [128, 48]
        config.neurogenesis.enabled = False
        model = DeepSeqAgnisModel(
            d_in=d_symbol,
            d_out=d_symbol,
            config=config,
            use_memory=True,
            use_replay=True,
        )
    elif model_key == 'deep_agnis_3L':
        config.hierarchy.enabled = True
        config.hierarchy.layer_dims = [64, 32, 16]
        config.hierarchy.k_sparse_per_layer = [8, 4, 2]
        config.hierarchy.commit_strides = [1, 4, 16]
        config.hierarchy.max_units_per_layer = [128, 48, 24]
        config.neurogenesis.enabled = False
        model = DeepSeqAgnisModel(
            d_in=d_symbol,
            d_out=d_symbol,
            config=config,
            use_memory=True,
            use_replay=True,
        )
    elif model_key == 'deep_agnis_3L_neurogenesis':
        config.hierarchy.enabled = True
        config.hierarchy.layer_dims = [64, 32, 16]
        config.hierarchy.k_sparse_per_layer = [8, 4, 2]
        config.hierarchy.commit_strides = [1, 4, 16]
        config.hierarchy.max_units_per_layer = [128, 48, 24]
        config.neurogenesis.enabled = True
        model = DeepSeqAgnisModel(
            d_in=d_symbol,
            d_out=d_symbol,
            config=config,
            use_memory=True,
            use_replay=True,
        )
    elif model_key == 'rnn_baseline':
        model = SimpleRNNBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z)
    elif model_key == 'gru_baseline':
        model = SimpleGRUBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z)
    elif model_key == 'rnn_replay_baseline':
        model = RNNReplayBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z, sleep_lr_scale=config.training.sleep_lr_scale)
    elif model_key == 'gru_replay_baseline':
        model = GRUReplayBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z, sleep_lr_scale=config.training.sleep_lr_scale)
    elif model_key == 'rnn_ewc_baseline':
        model = RNNEWCBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z, ewc_lambda=100.0)
    elif model_key == 'gru_ewc_baseline':
        model = GRUEWCBaseline(d_in=d_symbol, d_out=d_symbol, d_hidden=config.model.d_z, ewc_lambda=100.0)
    elif model_key == 'sparc_task_id_oracle':
        from agnis.sequence.sequence_wrapper import SPARCSequenceWrapper
        model = SPARCSequenceWrapper(
            d_in=d_symbol,
            d_out=d_symbol,
            num_columns=n_tasks,
            d_latent=32,
            alpha=0.01,
            beta=0.5,
            eta_D=0.01,
            eta_R=0.01,
            eta_Q=0.01,
            step_c=0.5,
            n_settle=15,
            routing_mode="task_id_oracle",
        )
    elif model_key == 'sparc_nearest_prototype':
        from agnis.sequence.sequence_wrapper import SPARCSequenceWrapper
        model = SPARCSequenceWrapper(
            d_in=d_symbol,
            d_out=d_symbol,
            num_columns=n_tasks,
            d_latent=32,
            alpha=0.01,
            beta=0.5,
            eta_D=0.01,
            eta_R=0.01,
            eta_Q=0.01,
            step_c=0.5,
            n_settle=15,
            routing_mode="nearest_prototype",
        )
    else:
        raise ValueError(f"Unknown model variant: {model_key}")

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
    capacity_timeline = []
    training_errors = []
    per_layer_error_timelines = []

    # Linear probe trackers
    probe_accuracies_before = []
    probe_accuracies_after = []

    total_births = 0
    total_prunes = 0
    start_time = time.time()
    step_counter = 0

    results_dir = os.path.join(config.results_dir, f"{args.model}", f"seed_{args.seed}")
    os.makedirs(results_dir, exist_ok=True)
    samples_dir = os.path.join(results_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    is_deep = hasattr(model, "hierarchy")

    # Main Task Loop
    for t_idx, d in enumerate(domains):
        print(f"\n[Phase6] --- Training Domain {t_idx}: {d.name} ---")
        model.start_task(t_idx)

        # Generate domain training stories
        train_stories = d.generate_stories(stories_per_domain, seed=args.seed)

        # Training loop
        for epoch in range(epochs_per_domain):
            epoch_errors = []
            for story_idx, story in enumerate(train_stories):
                text = story["prompt"] + story["target"]
                stream = CharacterStream(text, vocab)
                model.reset_sequence_state()

                for x, y, x_idx, y_idx in stream:
                    step_counter += 1
                    metrics = model.train_transition(x, y)
                    error = metrics.get('error', 0.0)
                    epoch_errors.append(error)
                    training_errors.append(error)

                    if is_deep:
                        # Log per-layer error timeline
                        per_layer_error_timelines.append(list(model.hierarchy.per_layer_errors))
                        capacity_timeline.append((step_counter, list(model.hierarchy.layer_dims)))
                    else:
                        capacity_timeline.append((step_counter, [config.model.d_z]))

                    if step_counter % log_every == 0:
                        cap_str = f" | Layers: {model.hierarchy.layer_dims}" if is_deep else ""
                        print(f"  Step {step_counter} | error_MSE: {error:.4f}{cap_str}")

            mean_epoch_err = sum(epoch_errors) / len(epoch_errors) if epoch_errors else 0.0
            print(f"  Epoch {epoch} finished. Mean MSE: {mean_epoch_err:.4f}")

        # Evaluate before sleep
        print(f"[Phase6] Evaluating BEFORE sleep...")
        row_before = []
        bpc_before = []
        probes_before = []

        for eval_idx, eval_domain in enumerate(domains):
            eval_stories = eval_domain.get_eval_stories(eval_prompts, seed=args.seed)
            metrics_eval = CharMetrics(vocab_size=d_symbol)

            # Initialize word boundary probes per layer
            if is_deep:
                word_probes = [WordBoundaryProbe() for _ in range(model.hierarchy.n_layers)]
            else:
                word_probes = []

            for story in eval_stories:
                prompt = story["prompt"]
                target = story["target"]

                # 1. Condition on prompt
                model.reset_sequence_state()
                for i in range(len(prompt) - 1):
                    model.advance_state_only(vocab.to_onehot(prompt[i]), vocab.to_onehot(prompt[i+1]))

                # 2. Scored evaluation on target continuation
                current_char = prompt[-1]
                for true_next_char in target:
                    current_oh = vocab.to_onehot(current_char)
                    true_next_oh = vocab.to_onehot(true_next_char)

                    pred = model.predict_no_state_update(current_oh)
                    metrics_eval.update(pred, vocab.encode(true_next_char))

                    model.advance_state_only(current_oh, true_next_oh)

                    # Record probe features if deep hierarchy
                    if is_deep:
                        for l in range(model.hierarchy.n_layers):
                            word_probes[l].record(model.hierarchy._z(l), true_next_char == ' ')

                    current_char = true_next_char

            row_before.append(metrics_eval.accuracy)
            bpc_before.append(metrics_eval.bpc)

            if is_deep:
                probes_before.append([p.evaluate() for p in word_probes])

        accuracy_matrix_before.append(row_before)
        bpc_matrix_before.append(bpc_before)
        probe_accuracies_before.append(probes_before)

        print(f"  Continuation Accuracy (before sleep): {['{:.3f}'.format(a) for a in row_before]}")
        if is_deep:
            print(f"  Word boundary probe accuracies (before sleep, per layer): {probes_before[-1]}")

        # Sleep / Replay
        if hasattr(model, "sleep"):
            print(f"[Phase6] Sleep consolidation after Domain {t_idx}...")
            model.sleep()

        # Pruning boundary (only if neurogenesis enabled)
        if model_key == "deep_agnis_3L_neurogenesis" and not args.smoke:
            initial_caps = list(model.hierarchy.layer_dims)
            model.hierarchy.prune_units(
                min_age=50,
                usage_threshold=config.neurogenesis.usage_threshold,
                importance_threshold=config.neurogenesis.importance_threshold,
                maturity_threshold=0.5
            )
            final_caps = list(model.hierarchy.layer_dims)
            if final_caps != initial_caps:
                total_prunes += sum(initial_caps) - sum(final_caps)
                print(f"  [Neurogenesis] Pruned units. Capacity: {initial_caps} -> {final_caps}")

        # Evaluate after sleep
        print(f"[Phase6] Evaluating AFTER sleep...")
        row_after = []
        bpc_after = []
        probes_after = []

        # We also generate stories for this domain after sleep to evaluate text quality
        eval_stories = domains[t_idx].get_eval_stories(eval_prompts, seed=args.seed)
        generated_samples = []

        for eval_idx, eval_domain in enumerate(domains):
            eval_stories_domain = eval_domain.get_eval_stories(eval_prompts, seed=args.seed)
            metrics_eval = CharMetrics(vocab_size=d_symbol)

            if is_deep:
                word_probes = [WordBoundaryProbe() for _ in range(model.hierarchy.n_layers)]

            for story_idx, story in enumerate(eval_stories_domain):
                prompt = story["prompt"]
                target = story["target"]

                # 1. Condition on prompt
                model.reset_sequence_state()
                for i in range(len(prompt) - 1):
                    model.advance_state_only(vocab.to_onehot(prompt[i]), vocab.to_onehot(prompt[i+1]))

                # 2. Scored evaluation
                current_char = prompt[-1]
                generated_text = ""
                for true_next_char in target:
                    current_oh = vocab.to_onehot(current_char)
                    true_next_oh = vocab.to_onehot(true_next_char)

                    pred = model.predict_no_state_update(current_oh)
                    metrics_eval.update(pred, vocab.encode(true_next_char))

                    model.advance_state_only(current_oh, true_next_oh)

                    if is_deep:
                        for l in range(model.hierarchy.n_layers):
                            word_probes[l].record(model.hierarchy._z(l), true_next_char == ' ')

                    current_char = true_next_char

                # 3. Generate autoregressive continuation sample for quality tracking (from task domain)
                if eval_idx == t_idx:
                    gen_cont = generate_continuation(
                        model=model,
                        prompt=prompt,
                        max_chars=config.generation.max_chars,
                        vocab=vocab,
                        decoding=config.generation.decoding,
                        temperature=config.generation.temperature,
                        top_k=config.generation.top_k,
                    )
                    generated_samples.append({
                        "prompt": prompt,
                        "target": target,
                        "generated": gen_cont,
                    })

            row_after.append(metrics_eval.accuracy)
            bpc_after.append(metrics_eval.bpc)

            if is_deep:
                probes_after.append([p.evaluate() for p in word_probes])

        accuracy_matrix_after.append(row_after)
        bpc_matrix_after.append(bpc_after)
        probe_accuracies_after.append(probes_after)

        print(f"  Continuation Accuracy (after sleep):  {['{:.3f}'.format(a) for a in row_after]}")
        print(f"  Continuation BPC (after sleep):       {['{:.3f}'.format(b) for b in bpc_after]}")
        if is_deep:
            print(f"  Word boundary probe accuracies (after sleep, per layer):  {probes_after[-1]}")

        # Compute and write generation samples
        sample_file = os.path.join(samples_dir, f"{d.name}_after_{domains[t_idx].name}.txt")
        with open(sample_file, "w") as sf:
            sf.write(f"# Evaluation samples for Domain: {d.name} after Training Task: {domains[t_idx].name}\n\n")
            for gs in generated_samples:
                sf.write(f"PROMPT: {gs['prompt']}\n")
                sf.write(f"TARGET: {gs['target']}\n")
                sf.write(f"GENERATED: {gs['generated']}\n")

                # Metrics for this story
                rep = compute_repetition_rate(gs['generated'], n=config.generation.repetition_ngram_n)
                kw = compute_keyword_retention(gs['generated'], gs['prompt'] + gs['target'])
                nc = compute_name_consistency(gs['generated'])
                sc = compute_sentence_completion(gs['generated'])
                d2 = compute_distinct_n(gs['generated'], n=2)
                d3 = compute_distinct_n(gs['generated'], n=3)

                sf.write("METRICS:\n")
                sf.write(f"  Repetition Rate: {rep:.3f}\n")
                sf.write(f"  Keyword Retention: {kw:.3f}\n")
                sf.write(f"  Name Consistency: {nc:.3f}\n")
                sf.write(f"  Sentence Completion: {sc:.3f}\n")
                sf.write(f"  Distinct-2: {d2:.3f}\n")
                sf.write(f"  Distinct-3: {d3:.3f}\n\n")
                sf.write("-" * 40 + "\n\n")

            # Collect metrics for summary
            repetition_rates.append(sum(compute_repetition_rate(g['generated'], config.generation.repetition_ngram_n) for g in generated_samples) / max(len(generated_samples), 1))
            distinct_2_rates.append(sum(compute_distinct_n(g['generated'], 2) for g in generated_samples) / max(len(generated_samples), 1))
            distinct_3_rates.append(sum(compute_distinct_n(g['generated'], 3) for g in generated_samples) / max(len(generated_samples), 1))

    runtime = time.time() - start_time
    print(f"\n[Phase6] Sweep completed in {runtime:.1f}s.")

    # Compute final consolidation stats
    avg_acc_after = sum(accuracy_matrix_after[-1]) / n_tasks
    avg_bpc_after = sum(bpc_matrix_after[-1]) / n_tasks
    forgetting = compute_forgetting(accuracy_matrix_after)
    bpc_forgetting = compute_bpc_forgetting(bpc_matrix_after)

    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0
    avg_bpc_forgetting = sum(bpc_forgetting) / len(bpc_forgetting) if bpc_forgetting else 0.0

    # Learning-vs-retention summary
    random_acc = 1.0 / d_symbol
    lvf = summarize_learning_vs_forgetting(accuracy_matrix_after, random_accuracy=random_acc)

    print(f"[Phase6] Final average accuracy: {avg_acc_after:.4f}")
    print(f"[Phase6] Final average BPC: {avg_bpc_after:.4f}")
    print(f"[Phase6] Average forgetting: {avg_forgetting:.4f}")
    print(f"[Phase6] Average BPC forgetting: {avg_bpc_forgetting:.4f}")
    print(f"[Phase6] Mean peak accuracy: {lvf['mean_peak_accuracy']:.4f}")
    print(f"[Phase6] Mean retained accuracy: {lvf['mean_retained_accuracy']:.4f}")
    print(f"[Phase6] Mean forward transfer: {lvf['mean_forward_transfer']:.4f}")

    # Build results dictionary
    results = {
        "model": args.model,
        "seed": args.seed,
        "runtime_seconds": runtime,
        "domain_order": domain_order,
        "accuracy_matrix_before": accuracy_matrix_before,
        "accuracy_matrix_after": accuracy_matrix_after,
        "bpc_matrix_before": bpc_matrix_before,
        "bpc_matrix_after": bpc_matrix_after,
        "probe_accuracies_before": probe_accuracies_before,
        "probe_accuracies_after": probe_accuracies_after,
        "average_accuracy_after": avg_acc_after,
        "average_bpc_after": avg_bpc_after,
        "forgetting": forgetting,
        "bpc_forgetting": bpc_forgetting,
        "average_forgetting": avg_forgetting,
        "average_bpc_forgetting": avg_bpc_forgetting,
        "random_accuracy": random_acc,
        "mean_peak_accuracy": lvf["mean_peak_accuracy"],
        "mean_retained_accuracy": lvf["mean_retained_accuracy"],
        "mean_forward_transfer": lvf["mean_forward_transfer"],
        "learning_headroom": lvf["learning_headroom"],
        "repetition_rate_mean": sum(repetition_rates) / max(len(repetition_rates), 1),
        "distinct_2_mean": sum(distinct_2_rates) / max(len(distinct_2_rates), 1),
        "distinct_3_mean": sum(distinct_3_rates) / max(len(distinct_3_rates), 1),
        "births": model.total_births if hasattr(model, "total_births") else 0,
        "prunes": model.total_prunes if hasattr(model, "total_prunes") else 0,
        "final_dims": list(model.hierarchy.layer_dims) if is_deep else [config.model.d_z],
    }

    # Save results to json
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save timelines
    with open(os.path.join(results_dir, "capacity_timeline.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "dims"])
        for row in capacity_timeline:
            writer.writerow([row[0], str(row[1])])

    with open(os.path.join(results_dir, "training_error.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "error"])
        for idx, err in enumerate(training_errors):
            writer.writerow([idx, err])

    if is_deep:
        with open(os.path.join(results_dir, "per_layer_errors.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step"] + [f"layer{l}" for l in range(model.hierarchy.n_layers)])
            for idx, row in enumerate(per_layer_error_timelines):
                writer.writerow([idx] + row)

    print(f"[Phase6] Done. Metrics saved to {results_dir}/metrics.json")


if __name__ == "__main__":
    main()
