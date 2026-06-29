"""
Raw AGNIS — tests/test_phase2_sequence_generators.py

Unit tests for Phase 2 sequence task generators.
"""

import pytest
from agnis.sequence.sequence_tasks import (
    generate_periodic_tasks,
    generate_doublet_tasks,
    generate_copy_tasks,
    generate_palindrome_tasks,
)


def test_periodic_task_generator():
    tasks = generate_periodic_tasks(num_tasks=2, sequences_per_task=3, seq_length=8, vocab_size_per_task=4)
    assert len(tasks) == 2
    assert tasks[0].vocab_size == 8
    assert len(tasks[0].sequences) == 3
    assert len(tasks[0].sequences[0]) == 8
    
    # Check disjoint vocab
    # Task 0 should use symbols 0-3; Task 1 should use symbols 4-7
    for seq in tasks[0].sequences:
        assert all(0 <= token < 4 for token in seq)
    for seq in tasks[1].sequences:
        assert all(4 <= token < 8 for token in seq)
        
    # Check periodicity pattern: ABCABC...
    seq0 = tasks[0].sequences[0]
    assert seq0[:4] == [0, 1, 2, 3]
    assert seq0[4:] == [0, 1, 2, 3]


def test_doublet_task_generator():
    tasks = generate_doublet_tasks(num_tasks=2, sequences_per_task=2, seq_length=8, vocab_size_per_task=3)
    # Check doublets pattern: AABBCCAABB...
    # Task 0 uses symbols 0, 1, 2
    seq = tasks[0].sequences[0]
    assert seq == [0, 0, 1, 1, 2, 2, 0, 0]


def test_copy_task_generator():
    tasks = generate_copy_tasks(num_tasks=2, sequences_per_task=2, copy_length=3, vocab_size_per_task=4)
    # Vocab size per task = 4. Task 0 uses 0, 1, 2 and SEP = 3.
    # Total vocab size = 8.
    seq = tasks[0].sequences[0]
    assert len(seq) == 7  # 3 (source) + 1 (SEP) + 3 (target)
    assert seq[3] == 3  # SEP token
    assert seq[:3] == seq[4:]  # source matches target


def test_palindrome_task_generator():
    tasks = generate_palindrome_tasks(num_tasks=2, sequences_per_task=2, half_length=3, vocab_size_per_task=4)
    seq = tasks[0].sequences[0]
    assert len(seq) == 6
    assert seq[:3] == list(reversed(seq[3:]))  # symmetric palindrome
