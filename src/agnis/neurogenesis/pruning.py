"""
Raw AGNIS — src/agnis/neurogenesis/pruning.py

Pruning and merging of unused, redundant, or low-importance units.

Pruning condition (all three must hold):
  - usage_j < usage_threshold
  - max(importance_j[:]) < importance_threshold
  - redundancy_j > redundancy_threshold

Additional condition:
  - Past probation period AND maturity_j < maturity_floor

See NOTES/neurogenesis_design.md for full specification.
"""

import torch
from typing import List, Tuple


def compute_redundancy(D: torch.Tensor, unit_idx: int, eps: float = 1e-8) -> float:
    """
    Compute the redundancy of unit unit_idx as max cosine similarity
    to any other unit's generative weights.

    Parameters
    ----------
    D : torch.Tensor of shape (d_in, d_z)
        Generative weight matrix. Each column is one unit.
    unit_idx : int
        The unit to compute redundancy for.

    Returns
    -------
    float
        Maximum cosine similarity to any other unit. 1.0 = fully redundant.
    """
    d_in, d_z = D.shape
    if d_z <= 1:
        return 0.0

    col_j = D[:, unit_idx]
    norm_j = col_j.norm().item()
    if norm_j < eps:
        return 0.0

    max_sim = 0.0
    for k in range(d_z):
        if k == unit_idx:
            continue
        col_k = D[:, k]
        norm_k = col_k.norm().item()
        if norm_k < eps:
            continue
        sim = abs((col_j @ col_k).item() / (norm_j * norm_k))
        if sim > max_sim:
            max_sim = sim

    return max_sim


def find_prune_candidates(
    usage: torch.Tensor,
    importance: torch.Tensor,
    D: torch.Tensor,
    usage_threshold: float = 0.05,
    importance_threshold: float = 0.01,
    redundancy_threshold: float = 0.90,
) -> List[int]:
    """
    Find units that are candidates for pruning.

    Parameters
    ----------
    usage : torch.Tensor of shape (d_z,)
        Per-unit usage (EMA of active fraction).
    importance : torch.Tensor of shape (d_z,) or (d_in, d_z)
        Per-unit importance. If 2D, max over d_in dimension is used.
    D : torch.Tensor of shape (d_in, d_z)
        Generative weight matrix for redundancy computation.
    usage_threshold, importance_threshold, redundancy_threshold : float
        Thresholds for each pruning criterion.

    Returns
    -------
    list of int
        Indices of units to prune (sorted in reverse order for safe deletion).
    """
    d_z = usage.shape[0]

    # Per-unit importance (max over rows if 2D)
    if importance.dim() == 2:
        max_importance = importance.max(dim=0).values  # (d_z,)
    else:
        max_importance = importance  # already (d_z,)

    candidates = []
    for j in range(d_z):
        low_usage = usage[j].item() < usage_threshold
        low_importance = max_importance[j].item() < importance_threshold
        high_redundancy = compute_redundancy(D, j) > redundancy_threshold

        if low_usage and low_importance and high_redundancy:
            candidates.append(j)

    # Return in reverse order so we can safely delete from end
    return sorted(candidates, reverse=True)


def find_merge_candidates(
    D: torch.Tensor,
    merge_threshold: float = 0.95,
) -> List[Tuple[int, int]]:
    """
    Find pairs of units that are nearly identical (merge candidates).

    Parameters
    ----------
    D : torch.Tensor of shape (d_in, d_z)
    merge_threshold : float

    Returns
    -------
    list of (keep_idx, merge_idx) pairs
        merge_idx should be removed after its weights are folded into keep_idx.
    """
    d_z = D.shape[1]
    pairs = []
    merged = set()

    for i in range(d_z):
        if i in merged:
            continue
        for j in range(i + 1, d_z):
            if j in merged:
                continue
            sim = compute_redundancy_pair(D, i, j)
            if sim > merge_threshold:
                pairs.append((i, j))  # keep i, merge j into i
                merged.add(j)

    return pairs


def compute_redundancy_pair(D: torch.Tensor, i: int, j: int, eps: float = 1e-8) -> float:
    """Cosine similarity between columns i and j of D."""
    ci = D[:, i]
    cj = D[:, j]
    ni = ci.norm().item()
    nj = cj.norm().item()
    if ni < eps or nj < eps:
        return 0.0
    return abs((ci @ cj).item() / (ni * nj))
