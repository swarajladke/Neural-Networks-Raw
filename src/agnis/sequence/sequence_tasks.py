"""
Raw AGNIS — src/agnis/sequence/sequence_tasks.py

Sequence task generators for periodic, doublet, copy, and palindrome conditions.
"""

import random
from typing import List, Dict, Any


class SequenceTask:
    """Represents a continual sequence learning task containing a list of token sequences."""

    def __init__(self, name: str, sequences: List[List[int]], vocab_size: int, task_id: int):
        self.name = name
        self.sequences = sequences  # List of List[int] (token indices)
        self.vocab_size = vocab_size
        self.task_id = task_id


def generate_periodic_tasks(
    num_tasks: int,
    sequences_per_task: int,
    seq_length: int,
    vocab_size_per_task: int = 4,
) -> List[SequenceTask]:
    """
    Generate periodic sequence tasks (e.g. ABCABCABC...).
    Each task i uses a disjoint vocabulary offset.
    """
    tasks = []
    total_vocab_size = num_tasks * vocab_size_per_task

    for i in range(num_tasks):
        vocab_start = i * vocab_size_per_task
        task_symbols = list(range(vocab_start, vocab_start + vocab_size_per_task))

        sequences = []
        for _ in range(sequences_per_task):
            # Create a simple cycle of task symbols repeated to seq_length
            seq = []
            while len(seq) < seq_length:
                seq.extend(task_symbols)
            sequences.append(seq[:seq_length])

        tasks.append(
            SequenceTask(
                name=f"Periodic_Task_{i}",
                sequences=sequences,
                vocab_size=total_vocab_size,
                task_id=i,
            )
        )
    return tasks


def generate_doublet_tasks(
    num_tasks: int,
    sequences_per_task: int,
    seq_length: int,
    vocab_size_per_task: int = 4,
) -> List[SequenceTask]:
    """
    Generate doublet sequence tasks (e.g. AABBCCAABBCC...).
    Each symbol is repeated twice before moving to the next.
    """
    tasks = []
    total_vocab_size = num_tasks * vocab_size_per_task

    for i in range(num_tasks):
        vocab_start = i * vocab_size_per_task
        task_symbols = list(range(vocab_start, vocab_start + vocab_size_per_task))

        sequences = []
        for _ in range(sequences_per_task):
            seq = []
            symbol_idx = 0
            while len(seq) < seq_length:
                sym = task_symbols[symbol_idx]
                seq.extend([sym, sym])  # repeat twice
                symbol_idx = (symbol_idx + 1) % len(task_symbols)
            sequences.append(seq[:seq_length])

        tasks.append(
            SequenceTask(
                name=f"Doublet_Task_{i}",
                sequences=sequences,
                vocab_size=total_vocab_size,
                task_id=i,
            )
        )
    return tasks


def generate_copy_tasks(
    num_tasks: int,
    sequences_per_task: int,
    copy_length: int = 3,
    vocab_size_per_task: int = 4,
) -> List[SequenceTask]:
    """
    Generate copy/delayed recall sequence tasks (e.g. A B C SEP A B C).
    Requires a reserved separator token at the end of each task's local vocab.
    """
    tasks = []
    total_vocab_size = num_tasks * vocab_size_per_task

    for i in range(num_tasks):
        vocab_start = i * vocab_size_per_task
        # Last index is reserved for local SEP token
        sep_token = vocab_start + vocab_size_per_task - 1
        content_symbols = list(range(vocab_start, sep_token))

        sequences = []
        for _ in range(sequences_per_task):
            # Sample a random source sequence of length copy_length
            source = [random.choice(content_symbols) for _ in range(copy_length)]
            # Target is the same
            seq = source + [sep_token] + source
            sequences.append(seq)

        tasks.append(
            SequenceTask(
                name=f"Copy_Task_{i}",
                sequences=sequences,
                vocab_size=total_vocab_size,
                task_id=i,
            )
        )
    return tasks


def generate_palindrome_tasks(
    num_tasks: int,
    sequences_per_task: int,
    half_length: int = 3,
    vocab_size_per_task: int = 4,
) -> List[SequenceTask]:
    """
    Generate palindrome sequence tasks (e.g. A B C C B A).
    """
    tasks = []
    total_vocab_size = num_tasks * vocab_size_per_task

    for i in range(num_tasks):
        vocab_start = i * vocab_size_per_task
        task_symbols = list(range(vocab_start, vocab_start + vocab_size_per_task))

        sequences = []
        for _ in range(sequences_per_task):
            # Generate random first half
            first_half = [random.choice(task_symbols) for _ in range(half_length)]
            # Palindrome is first half followed by its reverse
            seq = first_half + list(reversed(first_half))
            sequences.append(seq)

        tasks.append(
            SequenceTask(
                name=f"Palindrome_Task_{i}",
                sequences=sequences,
                vocab_size=total_vocab_size,
                task_id=i,
            )
        )
    return tasks
