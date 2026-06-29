"""
Raw AGNIS — src/agnis/evaluation/representation.py

Implements representation analysis for Continual Learning:
- Computes mean task prototypes in latent space
- Computes pairwise cosine similarity / representation overlap
- Computes unit-level importance and interference risk scores
"""

import torch
from typing import List, Dict, Optional, Tuple


def compute_task_prototypes(
    model,
    task_data_list: List[Tuple[torch.Tensor, torch.Tensor]],
    task_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the mean latent representation (prototype) for a task.

    Parameters
    ----------
    model : AssociativeModel
    task_data_list : list of (input, target) pairs
    task_context : torch.Tensor, optional

    Returns
    -------
    torch.Tensor of shape (d_z,)
        Mean latent activation vector.
    """
    latents = []
    for x, y in task_data_list:
        with torch.no_grad():
            a = model.get_latent(x, y=y, task_context=task_context)
            latents.append(a.detach().clone())
            
    if not latents:
        return torch.zeros(1)
        
    return torch.stack(latents).mean(dim=0)


def compute_pairwise_overlap(prototypes: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between task prototypes.

    Parameters
    ----------
    prototypes : list of torch.Tensor
        List of task prototypes (each of shape (d_z,)).

    Returns
    -------
    torch.Tensor of shape (num_tasks, num_tasks)
        Symmetric overlap matrix.
    """
    num_tasks = len(prototypes)
    overlap_matrix = torch.zeros(num_tasks, num_tasks)
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            p_i = prototypes[i]
            p_j = prototypes[j]
            norm_i = p_i.norm().item()
            norm_j = p_j.norm().item()
            
            if norm_i > 1e-8 and norm_j > 1e-8:
                overlap_matrix[i, j] = (p_i @ p_j).item() / (norm_i * norm_j)
            else:
                overlap_matrix[i, j] = 0.0
                
    return overlap_matrix


def compute_interference_score(
    model,
    new_task_prototype: torch.Tensor,
) -> float:
    """
    Compute the interference score: how much the new task activates units 
    that are highly important for past tasks.
    
    Interference = (new_task_prototype @ unit_importance) / (norms + eps)
    
    Where unit_importance is estimated from the EMA of generative weight changes.

    Parameters
    ----------
    model : AssociativeModel
    new_task_prototype : torch.Tensor of shape (d_z,)

    Returns
    -------
    float
        Interference score in [0, 1].
    """
    # Expose weight importance if model is AgnisBaseline
    if not hasattr(model, 'cell') or not hasattr(model.cell, 'importance_D'):
        return 0.0
        
    # Sum/max over input dimension to get unit-level importance
    # importance_D shape: (d_in, d_z)
    importance_D = model.cell.importance_D
    unit_importance = importance_D.max(dim=0).values # shape (d_z,)
    
    norm_p = new_task_prototype.norm().item()
    norm_imp = unit_importance.norm().item()
    
    if norm_p > 1e-8 and norm_imp > 1e-8:
        score = (new_task_prototype @ unit_importance).item() / (norm_p * norm_imp)
        return max(0.0, min(1.0, score))
        
    return 0.0
