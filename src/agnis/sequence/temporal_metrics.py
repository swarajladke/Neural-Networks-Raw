"""
Raw AGNIS — src/agnis/sequence/temporal_metrics.py

Metrics for sequence learning accuracy, forgetting, and condition-specific temporal consistency.
"""

import torch
from typing import List, Dict, Any
from agnis.sequence.sequence_tasks import SequenceTask


def evaluate_model_on_sequence_task(
    model,
    sequences: List[List[int]],
    vocab_size: int,
) -> float:
    """Evaluate next-symbol prediction accuracy over a list of sequences."""
    correct = 0
    total = 0
    
    for seq in sequences:
        model.reset_sequence_state()
        for t in range(len(seq) - 1):
            x = torch.zeros(vocab_size)
            x[seq[t]] = 1.0
            
            # Predict next token distribution
            pred = model.predict_transition(x)
            if pred.argmax().item() == seq[t + 1]:
                correct += 1
            total += 1
            
    return correct / total if total > 0 else 0.0


def compute_temporal_consistency(
    model,
    condition: str,
    task: SequenceTask,
    vocab_size: int,
) -> float:
    """
    Compute condition-specific temporal consistency:
    - periodic: cycle transition correctness
    - doublet: repeated symbol duration rule correctness
    - copy: autoregressive matches of copied segment
    - palindrome: autoregressive matches of symmetric segment
    """
    # Use a subset of sequences for consistency testing
    test_seqs = task.sequences[:5]
    if not test_seqs:
        return 0.0

    # Get local task symbols range
    # Since symbols are disjoint: task_id * vocab_size_per_task
    # We can infer it from the sequence values
    all_syms = set()
    for seq in task.sequences:
        all_syms.update(seq)
    
    if not all_syms:
        return 0.0
        
    task_symbols = sorted(list(all_syms))

    if condition == "periodic":
        # Check transition correctness for each symbol in the cycle
        correct = 0
        total = 0
        model.reset_sequence_state()
        
        # Cycle transition: sym_i -> sym_{i+1}
        for idx, sym in enumerate(task_symbols):
            x = torch.zeros(vocab_size)
            x[sym] = 1.0
            pred = model.predict_transition(x)
            next_target = task_symbols[(idx + 1) % len(task_symbols)]
            if pred.argmax().item() == next_target:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0

    elif condition == "doublet":
        # Check repeated symbol duration rule:
        # A A B B C C ...
        # Even index: A -> predict A
        # Odd index:  A -> predict B
        correct = 0
        total = 0
        
        for seq in test_seqs:
            model.reset_sequence_state()
            for t in range(len(seq) - 1):
                # Determine position inside task doublet cycle
                # Since symbols cycle as: sym0, sym0, sym1, sym1, sym2, sym2...
                # Current index in task_symbols
                curr_sym = seq[t]
                curr_pos = task_symbols.index(curr_sym)
                
                # Check if this is the first or second doublet token
                # Look back to see if we just saw this symbol
                is_first = (t == 0) or (seq[t - 1] != curr_sym)
                
                x = torch.zeros(vocab_size)
                x[curr_sym] = 1.0
                pred = model.predict_transition(x)
                
                if is_first:
                    # First doublet token: target is the same symbol
                    target = curr_sym
                else:
                    # Second doublet token: target is the next symbol in cycle
                    target = task_symbols[(curr_pos + 1) % len(task_symbols)]
                    
                if pred.argmax().item() == target:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0.0

    elif condition == "copy":
        # Autoregressive generation test:
        # Feed source segment A B C SEP -> predict next L steps, check if they match A B C
        correct = 0
        total = 0
        # Determine L (copy_length)
        # Sequence is: source + SEP + target
        # Hence len(seq) = 2 * L + 1 -> L = (len(seq) - 1) // 2
        for seq in test_seqs:
            L = (len(seq) - 1) // 2
            sep_token = seq[L]
            source = seq[:L]
            
            # 1. Feed source and separator
            model.reset_sequence_state()
            for t in range(L + 1):
                x = torch.zeros(vocab_size)
                x[seq[t]] = 1.0
                pred_dist = model.predict_transition(x)
                
            # 2. Autoregressively generate next L steps
            generated = []
            curr_input = sep_token
            for _ in range(L):
                x = torch.zeros(vocab_size)
                x[curr_input] = 1.0
                pred_dist = model.predict_transition(x)
                next_sym = pred_dist.argmax().item()
                generated.append(next_sym)
                curr_input = next_sym
                
            # Compare generated to source
            for gen_sym, src_sym in zip(generated, source):
                if gen_sym == src_sym:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0.0

    elif condition == "palindrome":
        # Autoregressive generation test:
        # Feed first half A B C -> predict next L steps, check if they match C B A
        correct = 0
        total = 0
        for seq in test_seqs:
            L = len(seq) // 2
            first_half = seq[:L]
            expected_symmetric = list(reversed(first_half))
            
            # 1. Feed first half
            model.reset_sequence_state()
            for t in range(L):
                x = torch.zeros(vocab_size)
                x[seq[t]] = 1.0
                pred_dist = model.predict_transition(x)
                
            # 2. Autoregressively generate next L steps
            generated = []
            curr_input = seq[L - 1] # last of first half
            for _ in range(L):
                x = torch.zeros(vocab_size)
                x[curr_input] = 1.0
                pred_dist = model.predict_transition(x)
                next_sym = pred_dist.argmax().item()
                generated.append(next_sym)
                curr_input = next_sym
                
            # Compare generated to expected symmetric reverse
            for gen_sym, sym_target in zip(generated, expected_symmetric):
                if gen_sym == sym_target:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0.0

    return 0.0
