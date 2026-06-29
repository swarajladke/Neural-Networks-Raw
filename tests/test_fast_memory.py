"""
Raw AGNIS — tests/test_fast_memory.py

Unit tests for FastMemory episodic prototype store.

Tests:
  1. test_fast_memory_write_retrieval
  2. test_fast_memory_skips_duplicate
  3. test_fast_memory_capacity_eviction
  4. test_fast_memory_importance_update
  5. test_fast_memory_stats
  6. test_fast_memory_sample_by_importance
"""

import pytest
import torch
from agnis.memory.fast_memory import FastMemory


@pytest.fixture
def memory():
    """Small FastMemory for testing."""
    return FastMemory(
        capacity=10,
        write_error_threshold=0.0,   # always write for testing
        write_novelty_threshold=0.0,
        min_similarity_to_skip_write=0.99,  # very high threshold to skip duplicates
    )


def test_fast_memory_write_and_retrieve(memory):
    """Store a key-value pair and retrieve it with a similar key."""
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    value = torch.tensor([0.0, 1.0, 0.0, 0.0])
    written = memory.write(key, value, error_val=0.5)
    assert written, "Memory should write high-error entry"
    assert memory.size == 1, f"Expected 1 entry, got {memory.size}"

    # Retrieve with same key
    result = memory.retrieve(key)
    assert result is not None, "Retrieval should return a result"
    retrieved_value, sim, entry = result
    assert sim > 0.99, f"Similarity with same key should be high, got {sim}"
    assert torch.allclose(retrieved_value, value), "Retrieved value should match stored value"


def test_fast_memory_returns_none_when_empty():
    """Retrieve on empty memory should return None."""
    memory = FastMemory(capacity=10)
    result = memory.retrieve(torch.randn(4))
    assert result is None, "Empty memory should return None"


def test_fast_memory_skips_near_duplicate(memory):
    """Should not write a near-duplicate entry (high cosine similarity)."""
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    value = torch.tensor([0.0, 1.0, 0.0, 0.0])

    # Write first entry
    memory.write(key, value, error_val=0.8)
    assert memory.size == 1

    # Try to write very similar key (should be skipped)
    key_similar = torch.tensor([1.0, 0.001, 0.0, 0.0])  # almost identical
    memory.write(key_similar, value, error_val=0.8)
    # Should still be 1 (duplicate detected, not written)
    assert memory.size == 1, f"Duplicate should not be written, size={memory.size}"


def test_fast_memory_capacity_eviction():
    """When full, writing a new entry should evict the lowest-importance entry."""
    memory = FastMemory(
        capacity=3,
        write_error_threshold=0.0,
        write_novelty_threshold=0.0,
        min_similarity_to_skip_write=0.999,
    )
    # Write 3 distinct entries
    for i in range(3):
        key = torch.zeros(4)
        key[i % 4] = 1.0
        key = key + torch.randn(4) * 0.001  # add tiny noise to prevent similarity skip
        memory.write(key, torch.randn(4), error_val=float(i + 1))

    assert memory.size == 3, f"Expected 3 entries, got {memory.size}"

    # Write 4th entry — should evict the lowest-importance (entry with error=1.0)
    new_key = torch.tensor([0.0, 0.0, 0.0, 1.0]) + torch.randn(4) * 0.001
    memory.write(new_key, torch.randn(4), error_val=10.0)
    assert memory.size == 3, f"Capacity should remain 3, got {memory.size}"


def test_fast_memory_importance_update():
    """Retrieving an entry should increment its usage_count and importance."""
    memory = FastMemory(
        capacity=10,
        write_error_threshold=0.0,
        write_novelty_threshold=0.0,
        min_similarity_to_skip_write=0.999,
    )
    key = torch.randn(4)
    value = torch.randn(4)
    memory.write(key, value, error_val=0.5)

    entry_before_usage = memory.get_all_entries()[0].usage_count
    memory.retrieve(key, update_importance=True)
    entry_after_usage = memory.get_all_entries()[0].usage_count
    assert entry_after_usage > entry_before_usage, "Usage count should increase after retrieval"


def test_fast_memory_stats(memory):
    """Stats should return correct size and utilization."""
    assert memory.size == 0
    assert memory.utilization == pytest.approx(0.0)

    key = torch.randn(4)
    memory.write(key, torch.randn(4), error_val=0.5)

    stats = memory.stats()
    assert stats["size"] == 1
    assert stats["utilization"] == pytest.approx(0.1)  # 1/10


def test_fast_memory_sample_by_importance(memory):
    """sample_by_importance should return at most n entries."""
    keys = [torch.randn(4) for _ in range(5)]
    for i, key in enumerate(keys):
        memory.write(key, torch.randn(4), error_val=float(i + 1))

    sampled = memory.sample_by_importance(3)
    assert len(sampled) <= 3, f"Should return at most 3 samples, got {len(sampled)}"
    assert len(sampled) > 0, "Should return at least 1 sample"


def test_fast_memory_task_entries(memory):
    """task_entries should filter by task_id correctly."""
    k1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    k2 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    k3 = torch.tensor([0.0, 0.0, 1.0, 0.0])

    memory.write(k1, torch.randn(4), error_val=0.9, task_id=0)
    memory.write(k2, torch.randn(4), error_val=0.9, task_id=1)
    memory.write(k3, torch.randn(4), error_val=0.9, task_id=0)

    task0_entries = memory.task_entries(0)
    task1_entries = memory.task_entries(1)

    assert len(task0_entries) == 2, f"Task 0 should have 2 entries, got {len(task0_entries)}"
    assert len(task1_entries) == 1, f"Task 1 should have 1 entry, got {len(task1_entries)}"
