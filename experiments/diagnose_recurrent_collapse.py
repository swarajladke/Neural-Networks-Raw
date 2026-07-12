"""
Raw AGNIS — experiments/diagnose_recurrent_collapse.py

Diagnostic tool to analyze representation overlap, Hebbian recurrent R-matrix rank collapse,
and Hebbian vs. backpropagation gradient alignment.
"""

import os
import sys
import torch
import numpy as np
import random
import yaml
from typing import Dict, List, Any

# Add src folder to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agnis.utils.config import AGNISConfig, load_config
from agnis.sequence.sequence_tasks import generate_doublet_tasks
from agnis.sequence.sequence_wrapper import SeqAgnisModel


def calculate_representational_overlap(model: SeqAgnisModel, seq: List[int], vocab_size: int) -> Dict[str, float]:
    """Calculate the cosine similarity between z-states of identical symbols in different contexts.
    
    In a doublet sequence [A, A, B, B, C, C...], symbol A appears twice.
    - z_first: latent state for the first 'A' (context: previous was C)
    - z_second: latent state for the second 'A' (context: previous was A)
    If cosine similarity between z_first and z_second is near 1.0, the model
    failed to disambiguate the context (representational collapse).
    """
    model.reset_sequence_state()
    z_states = {}
    
    # Run the sequence and collect latent activation vectors
    # We step through the sequence without training
    for t in range(len(seq) - 1):
        x = torch.zeros(vocab_size)
        x[seq[t]] = 1.0
        
        # Get latent activation (the activation vector 'a' representing the latent code z)
        # AgnisBaseline.get_latent returns a = cell.forward(s_query, observed_mask=self.observed_mask)
        a_latent = model.base_model.get_latent(x)
        
        # Track symbol and its position (first vs second occurrence in doublet)
        symbol = seq[t]
        is_second = (t > 0 and seq[t-1] == symbol)
        
        pos_key = "second" if is_second else "first"
        if symbol not in z_states:
            z_states[symbol] = {}
        if pos_key not in z_states[symbol]:
            z_states[symbol][pos_key] = []
        
        z_states[symbol][pos_key].append(a_latent.detach().clone())
    
    # Compute average cosine similarity for each symbol
    similarities = []
    for sym, positions in z_states.items():
        if "first" in positions and "second" in positions:
            firsts = torch.stack(positions["first"])
            seconds = torch.stack(positions["second"])
            
            # Mean representation for first and second position
            mean_first = firsts.mean(dim=0)
            mean_second = seconds.mean(dim=0)
            
            # Cosine similarity
            denom = mean_first.norm() * mean_second.norm()
            if denom > 1e-8:
                sim = torch.dot(mean_first, mean_second) / denom
                similarities.append(sim.item())
                
    return {
        "mean_overlap": float(np.mean(similarities)) if similarities else 1.0,
        "raw_overlaps": similarities
    }


def analyze_recurrent_weights(model: SeqAgnisModel) -> Dict[str, Any]:
    """Compute SVD and matrix norm properties of R to detect rank collapse."""
    R = model.base_model.cell.R.detach()
    
    # Frobenius norm and Spectral norm (ord=2)
    f_norm = torch.linalg.matrix_norm(R, ord="fro").item()
    spectral_norm = torch.linalg.matrix_norm(R, ord=2).item()
    
    # SVD
    s_vals = torch.linalg.svdvals(R)
    s_vals_np = s_vals.cpu().numpy()
    
    # Singular value entropy: measure of rank concentration
    # If entropy is near 0, the matrix has collapsed to a rank-1 subspace.
    total_s = s_vals_np.sum()
    if total_s > 1e-8:
        probs = s_vals_np / total_s
        entropy = -np.sum(probs * np.log(probs + 1e-8))
    else:
        entropy = 0.0
        
    # Fraction of variance explained by top-k singular values
    top3_ratio = float(s_vals_np[:3].sum() / max(total_s, 1e-8))
    
    return {
        "fro_norm": f_norm,
        "spectral_norm": spectral_norm,
        "singular_values": s_vals_np.tolist(),
        "svd_entropy": entropy,
        "top3_variance_ratio": top3_ratio,
        "effective_rank": int(np.exp(entropy))
    }


def compute_gradient_alignment(model: SeqAgnisModel, x: torch.Tensor, y_idx: int) -> float:
    """Measure the cosine angle between the Hebbian update delta_R and the ideal backprop gradient.
    
    - delta_R: Hebbian recurrent update.
    - R_grad: Standard gradient of cross entropy loss with respect to R.
    A positive cosine similarity means the Hebbian update is aligned with gradient descent.
    """
    cell = model.base_model.cell
    if not cell.use_recurrent:
        return 0.0
        
    # 1. Capture and detach cell states to isolate the single-step computation graph
    z_prev_backup = cell.z_prev.detach().clone() if cell.z_prev is not None else None
    z_backup = cell.z.detach().clone() if cell.z is not None else None
    
    if cell.z_prev is not None:
        cell.z_prev = cell.z_prev.detach().requires_grad_(False)
    if cell.z is not None:
        cell.z = cell.z.detach().requires_grad_(False)
        
    z_prev = cell.z_prev.clone() if cell.z_prev is not None else None
    
    # Make a copy of R that requires gradient
    R_var = cell.R.clone().detach().requires_grad_(True)
    
    # Backup original R
    R_orig = cell.R
    cell.R = R_var  # inject parameter with gradient tracking
    
    # 2. Run query pass
    zeros_target = torch.zeros(model.base_model.d_out_y)
    s_query = torch.cat([x, zeros_target])
    
    # Run settling using our gradient-enabled R
    a = cell.forward(s_query, observed_mask=model.base_model.observed_mask)
    
    # Target prediction
    pred_joint = cell.D @ a
    pred_target = pred_joint[model.base_model.d_in_x:]
    pred_probs = torch.softmax(pred_target, dim=-1)
    
    # Cross-entropy loss
    loss = -torch.log(pred_probs[y_idx] + 1e-8)
    loss.backward()
    
    # Get descent direction (negative gradient)
    R_descent = -R_var.grad.detach() if R_var.grad is not None else torch.zeros_like(R_orig)
    
    # Restore original R
    cell.R = R_orig
    
    # Restore cell states from backup
    cell.z_prev = z_prev_backup
    cell.z = z_backup
    
    # 3. Calculate local Hebbian update delta_R
    # Recurrent update: delta_R = Hebbian update = eta_R * outer(z - R @ a_prev, a_prev)
    a_prev = cell.activation(z_prev_backup) if z_prev_backup is not None else torch.zeros_like(a)
    if cell.use_sparsity:
        from agnis.core.sparsity import kwta
        a_prev = kwta(a_prev, cell.k_sparse)
        
    z_target = cell.z.detach() if cell.z is not None else torch.zeros_like(a)
    z_pred = cell.R @ a_prev
    r_error = z_target - z_pred
    delta_R = cell.eta_R * torch.outer(r_error, a_prev)
    
    # Compute cosine similarity between delta_R and R_descent
    denom = delta_R.norm() * R_descent.norm()
    if denom > 1e-8:
        similarity = torch.sum(delta_R * R_descent) / denom
        return similarity.item()
    return 0.0


def run_diagnostics(epochs: int = 15):
    # Load configuration from existing yaml file
    config = load_config("configs/phase6_smoke.yaml")
    
    # Force isolate recurrent dynamics
    config.model.use_recurrent = True
    config.model.d_z = 32
    config.model.k_sparse = 4
    
    # Vocab size = 4
    vocab_size = 4
    tasks = generate_doublet_tasks(num_tasks=1, sequences_per_task=10, seq_length=40, vocab_size_per_task=vocab_size)
    task = tasks[0]
    
    for label, maturity_on in [("Baseline (No Maturity Gating)", False), ("Maturity Gated Model", True)]:
        print(f"\n=== Running Configuration: {label} ===")
        
        # Initialize baseline wrapper
        model = SeqAgnisModel(
            d_in=vocab_size,
            d_out=vocab_size,
            d_z=32,
            config=config,
            R_update_enabled=True,
            R_drive_enabled=True,
            use_recurrent=True,
            use_memory=False,  # Isolate recurrence by disabling retrieval bypass
            use_replay=False,
            maturity_enabled=maturity_on
        )
        
        print("\n--- Epoch-by-Epoch Diagnostic Log ---")
        print(f"| Epoch | Accuracy | Cos Overlap | SV Entropy | Eff Rank | Gradient Align |")
        print(f"|---|---|---|---|---|---|")
        
        for epoch in range(epochs):
            alignments = []
            correct_count = 0
            total_predictions = 0
            
            seqs = list(task.sequences)
            random.shuffle(seqs)
            
            for seq in seqs:
                model.reset_sequence_state()
                for t in range(len(seq) - 1):
                    x = torch.zeros(vocab_size)
                    x[seq[t]] = 1.0
                    y = torch.zeros(vocab_size)
                    y[seq[t+1]] = 1.0
                    
                    # Check accuracy before training on this pair without mutating sequence state
                    pred = model.predict_no_state_update(x)
                    if pred.argmax().item() == seq[t+1]:
                        correct_count += 1
                    total_predictions += 1
                    
                    # Measure alignment before updating weights
                    align = compute_gradient_alignment(model, x, seq[t+1])
                    alignments.append(align)
                    
                    # Update weights
                    model.train_transition(x, y)
                    
                    # Force detach internal latent variables in-place to break any BPTT graph
                    if model.base_model.cell.z_prev is not None:
                        model.base_model.cell.z_prev = model.base_model.cell.z_prev.detach()
                    if model.base_model.cell.z is not None:
                        model.base_model.cell.z = model.base_model.cell.z.detach()
                    
            # Calculate statistics
            acc = correct_count / max(total_predictions, 1)
            mean_align = np.mean(alignments)
            
            # Representational overlap
            overlap_stats = calculate_representational_overlap(model, task.sequences[0], vocab_size)
            overlap = overlap_stats["mean_overlap"]
            
            # SVD analysis of R
            r_stats = analyze_recurrent_weights(model)
            
            print(f"| {epoch+1:5d} | {acc:8.2%} | {overlap:11.3f} | {r_stats['svd_entropy']:10.3f} | {r_stats['effective_rank']:8d} | {mean_align:14.3f} |")
            
        print("\n=== Diagnosis Summary ===")
        if overlap > 0.8:
            print("[WARNING] High representation overlap detected. The representations for identical characters")
            print("          in different contexts are nearly collinear. Hebbian R-update updates collide.")
        if r_stats['effective_rank'] < 5:
            print("[WARNING] Recurrent matrix rank collapse detected. The effective rank of R is extremely low,")
            print("          meaning temporal context propagation is confined to a tiny projection space.")
        if mean_align < 0.1:
            print("[WARNING] Poor Hebbian gradient alignment. The local Hebbian update updates are nearly orthogonal")
            print("          or opposing to the true error-minimizing gradient descent directions.")
        print("=========================")


if __name__ == "__main__":
    run_diagnostics()
