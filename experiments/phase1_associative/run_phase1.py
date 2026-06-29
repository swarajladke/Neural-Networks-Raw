"""
Raw AGNIS — experiments/phase1_associative/run_phase1.py

Phase 1: Associative Continual Memory Experiment

Protocol:
    1. Train Task 0 (A→B, C→D)   → Evaluate Task 0
    2. Train Task 1 (E→F, G→H)   → Evaluate Task 0, Task 1
    3. Train Task 2 (I→J, K→L)   → Evaluate Task 0, Task 1, Task 2

Models compared:
    - Naive MLP (backprop, no memory)
    - Dense Hebbian (Hebbian, no sparsity)
    - Raw AGNIS (no sparsity) — ablation
    - Raw AGNIS (no memory) — ablation
    - Raw AGNIS (full)

Metrics reported:
    - Accuracy per task per checkpoint
    - Average forgetting per model
    - Prediction error curves
    - Sparsity levels (where applicable)

Usage:
    python experiments/phase1_associative/run_phase1.py
    python experiments/phase1_associative/run_phase1.py --config configs/phase1_micro.yaml
    python experiments/phase1_associative/run_phase1.py --seed 42 --n_repeats 100
"""

import sys
import os
import argparse
import torch
import random

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from agnis.core.predictive_cell import PredictiveCell
from agnis.memory.fast_memory import FastMemory
from agnis.memory.replay_buffer import ReplayBuffer
from agnis.training.curriculum import build_phase1_tasks, ContinualCurriculum
from agnis.training.online_trainer import OnlineTrainer
from agnis.training.sleep_trainer import SleepTrainer
from agnis.evaluation.baselines import NaiveMLP, DenseHebbian
from agnis.evaluation.forgetting import ForgettingTracker
from agnis.evaluation.metrics import ContinualLearningMetrics
from agnis.utils.visualization import plot_forgetting_curves, plot_forgetting_comparison


# ── Configuration ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Raw AGNIS Phase 1: Associative Continual Memory")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_repeats", type=int, default=50,
                        help="Number of times to present each pair per task")
    parser.add_argument("--d_z", type=int, default=32, help="Latent dimension")
    parser.add_argument("--k_sparse", type=int, default=3, help="kWTA k value")
    parser.add_argument("--n_settle", type=int, default=10, help="Settling iterations")
    parser.add_argument("--eval_threshold", type=float, default=0.3,
                        help="MSE threshold for 'correct' in accuracy computation")
    parser.add_argument("--results_dir", type=str, default="results/phase1/")
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)


# ── Evaluation Helper ─────────────────────────────────────────────────────────

def evaluate_all_tasks_on_model(model, all_task_data, threshold=0.3):
    """Evaluate model on all tasks. Returns list of accuracies."""
    accs = []
    for task_data in all_task_data:
        correct = 0
        for s, target in task_data:
            with torch.no_grad():
                a = model.forward(s) if hasattr(model, 'forward') else model.forward(s)
                if hasattr(model, 'D'):
                    pred = model.D @ a
                else:
                    pred = model.predict(s)
                err = ((pred - target) ** 2).mean().item()
                if err < threshold:
                    correct += 1
        accs.append(correct / len(task_data) if task_data else 0.0)
    return accs


def evaluate_baseline(model, all_task_data, threshold=0.3):
    """Evaluate a baseline model (NaiveMLP, DenseHebbian) on all tasks."""
    accs = []
    for task_data in all_task_data:
        correct = 0
        for s, target in task_data:
            pred = model.predict(s)
            err = ((pred - target) ** 2).mean().item()
            if err < threshold:
                correct += 1
        accs.append(correct / len(task_data) if task_data else 0.0)
    return accs


# ── Main Experiment ───────────────────────────────────────────────────────────

def run_experiment(args):
    set_seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Raw AGNIS - Phase 1: Associative Continual Memory")
    print(f"  Seed: {args.seed} | Repeats/task: {args.n_repeats}")
    print(f"  d_z: {args.d_z} | k_sparse: {args.k_sparse} | n_settle: {args.n_settle}")
    print(f"{'='*70}\n")

    vocab_size = 12  # 12 symbols: A,B,C,D,...,L

    # ── Build Tasks ───────────────────────────────────────────────────────────
    tasks = build_phase1_tasks(vocab_size=vocab_size)
    curriculum = ContinualCurriculum(tasks, n_repeats_per_task=args.n_repeats)
    all_task_data_full = curriculum.all_tasks_data()
    # Evaluation uses smaller set (1 repeat)
    all_task_data_eval = curriculum.all_tasks_data(n_repeats=1)

    print(f"  Tasks: {[t.name for t in tasks]}")
    print(f"  Training samples per task: {len(all_task_data_full[0])}")
    print()

    # ── Models ────────────────────────────────────────────────────────────────

    # 1. Naive MLP baseline
    mlp = NaiveMLP(d_in=vocab_size, d_out=vocab_size, d_hidden=64, lr=0.01)

    # 2. Dense Hebbian baseline
    dense_hebb = DenseHebbian(d_in=vocab_size, d_z=args.d_z, eta=0.01)

    # 3. Raw AGNIS (no sparsity) - ablation
    agnis_no_sparse = PredictiveCell(
        d_in=vocab_size, d_z=args.d_z, k_sparse=args.d_z,
        n_settle=args.n_settle, eta_D=0.01, eta_E=0.01,
        use_sparsity=False,
    )

    # 4. Raw AGNIS (no memory) - ablation
    agnis_no_mem = PredictiveCell(
        d_in=vocab_size, d_z=args.d_z, k_sparse=args.k_sparse,
        n_settle=args.n_settle, eta_D=0.01, eta_E=0.01,
        use_sparsity=True,
    )

    # 5. Raw AGNIS (full: sparsity + memory + replay)
    agnis_full = PredictiveCell(
        d_in=vocab_size, d_z=args.d_z, k_sparse=args.k_sparse,
        n_settle=args.n_settle, eta_D=0.01, eta_E=0.01,
        use_sparsity=True,
    )
    fast_mem = FastMemory(
        capacity=128,
        write_error_threshold=0.2,
        write_novelty_threshold=0.15,
    )
    replay_buf = ReplayBuffer(max_size=64)

    # ── Forgetting Trackers ───────────────────────────────────────────────────
    n_tasks = curriculum.n_tasks
    trackers = {
        "NaiveMLP": ForgettingTracker(n_tasks),
        "DenseHebbian": ForgettingTracker(n_tasks),
        "AGNIS_noSparse": ForgettingTracker(n_tasks),
        "AGNIS_noMem": ForgettingTracker(n_tasks),
        "AGNIS_full": ForgettingTracker(n_tasks),
    }

    # Accuracy matrices for plotting
    matrices = {name: [] for name in trackers}

    # ── Sequential Training Loop ──────────────────────────────────────────────
    for task_idx in range(n_tasks):
        task = tasks[task_idx]
        task_data = all_task_data_full[task_idx]

        print(f"[Task {task_idx}] Training: {task.name}  ({len(task_data)} samples)")

        # ── Train each model ──────────────────────────────────────────────────

        # 1. NaiveMLP
        for s, target in task_data:
            mlp.train_on(s, target)

        # 2. Dense Hebbian
        for s, target in task_data:
            dense_hebb.train_on(s, target)

        # 3. AGNIS no sparse
        for s, target in task_data:
            a = agnis_no_sparse.forward(s)
            agnis_no_sparse.update_weights(s, a)

        # 4. AGNIS no memory (training only)
        for s, target in task_data:
            a = agnis_no_mem.forward(s)
            agnis_no_mem.update_weights(s, a)

        # 5. AGNIS full (training + memory write)
        for s, target in task_data:
            a = agnis_full.forward(s)
            agnis_full.update_weights(s, a)
            err_val = agnis_full.prediction_error or 0.0
            fast_mem.tick()
            fast_mem.write(
                key=a.detach().clone(),
                value=s.detach().clone(),
                error_val=err_val,
                task_id=task_idx,
            )

        # Sleep phase for AGNIS full
        replay_buf.add_from_memory(fast_mem, n=32)
        sleep_trainer = SleepTrainer(
            model=agnis_full,
            replay_buffer=replay_buf,
            sleep_lr_scale=0.3,
            importance_protect_threshold=0.5,
        )
        sleep_trainer.sleep(n_replay=16, n_steps=1)

        # ── Evaluate all tasks ────────────────────────────────────────────────
        print(f"[Task {task_idx}] Evaluating all tasks...")

        checkpoint_row_mlp = []
        checkpoint_row_dense = []
        checkpoint_row_nosparse = []
        checkpoint_row_nomem = []
        checkpoint_row_full = []

        for eval_task_idx in range(n_tasks):
            eval_data = all_task_data_eval[eval_task_idx]

            if eval_task_idx > task_idx:
                # Not yet trained - record None
                checkpoint_row_mlp.append(None)
                checkpoint_row_dense.append(None)
                checkpoint_row_nosparse.append(None)
                checkpoint_row_nomem.append(None)
                checkpoint_row_full.append(None)
                continue

            acc_mlp = evaluate_baseline(mlp, [eval_data], args.eval_threshold)[0]
            acc_dense = evaluate_baseline(dense_hebb, [eval_data], args.eval_threshold)[0]
            acc_nosparse = evaluate_all_tasks_on_model(agnis_no_sparse, [eval_data], args.eval_threshold)[0]
            acc_nomem = evaluate_all_tasks_on_model(agnis_no_mem, [eval_data], args.eval_threshold)[0]
            acc_full = evaluate_all_tasks_on_model(agnis_full, [eval_data], args.eval_threshold)[0]

            checkpoint_row_mlp.append(acc_mlp)
            checkpoint_row_dense.append(acc_dense)
            checkpoint_row_nosparse.append(acc_nosparse)
            checkpoint_row_nomem.append(acc_nomem)
            checkpoint_row_full.append(acc_full)

        matrices["NaiveMLP"].append(checkpoint_row_mlp)
        matrices["DenseHebbian"].append(checkpoint_row_dense)
        matrices["AGNIS_noSparse"].append(checkpoint_row_nosparse)
        matrices["AGNIS_noMem"].append(checkpoint_row_nomem)
        matrices["AGNIS_full"].append(checkpoint_row_full)

        print(f"  Accuracies at checkpoint {task_idx}:")
        for name, row in [
            ("NaiveMLP", checkpoint_row_mlp),
            ("DenseHebbian", checkpoint_row_dense),
            ("AGNIS_noSparse", checkpoint_row_nosparse),
            ("AGNIS_noMem", checkpoint_row_nomem),
            ("AGNIS_full", checkpoint_row_full),
        ]:
            row_str = "  ".join([
                f"T{i}:{v:.3f}" if v is not None else f"T{i}:----"
                for i, v in enumerate(row)
            ])
            print(f"    {name:16s}: {row_str}")
        print()

    # ── Final Results ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Final Results: Forgetting Summary")
    print(f"{'='*70}")

    task_order = list(range(n_tasks))
    from agnis.evaluation.forgetting import compute_forgetting, compute_average_forgetting

    all_forgettings = {}
    for name, matrix in matrices.items():
        forgetting = compute_forgetting(matrix, task_order)
        avg_f = compute_average_forgetting(forgetting)
        all_forgettings[name] = forgetting
        print(f"  {name:20s}: avg_forgetting={avg_f:.4f}  per_task={[f'{v:.3f}' for v in forgetting]}")

    print(f"\n  Memory size (AGNIS_full): {fast_mem.size}")
    print(f"  Memory utilization: {fast_mem.utilization:.1%}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        # Forgetting curve per model
        for name, matrix in matrices.items():
            plot_forgetting_curves(
                accuracy_matrix=matrix,
                task_names=[f"Task {i}" for i in range(n_tasks)],
                title=f"Phase 1 Forgetting Curves - {name}",
                save_path=os.path.join(args.results_dir, f"forgetting_curves_{name}.png"),
            )

        # Comparison bar chart
        avg_forgettings = {
            name: [compute_average_forgetting(compute_forgetting(matrix, task_order))]
            for name, matrix in matrices.items()
        }
        plot_forgetting_comparison(
            {name: compute_forgetting(matrix, task_order) for name, matrix in matrices.items()},
            title="Phase 1: Average Forgetting by Model",
            save_path=os.path.join(args.results_dir, "forgetting_comparison.png"),
        )

    print(f"\n  Results saved to: {args.results_dir}")
    print(f"{'='*70}\n")

    return matrices, all_forgettings


if __name__ == "__main__":
    args = parse_args()
    matrices, forgettings = run_experiment(args)
