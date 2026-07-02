"""
Raw AGNIS — src/agnis/text/char_metrics.py

Metrics tracking character prediction accuracy, Bits-Per-Character (BPC), and forgetting.
"""

import torch
import math
from typing import Dict, List, Optional

class CharMetrics:
    """Accumulates character-level prediction metrics."""
    
    def __init__(self, vocab_size: int, eps: float = 1e-8):
        self.vocab_size = vocab_size
        self.eps = eps
        self.reset()
    
    def reset(self):
        self.total = 0
        self.correct = 0
        self.correct_top3 = 0
        self.total_log_prob = 0.0  # sum of log2(p(y))
    
    def update(self, pred_probs: torch.Tensor, y_idx: int):
        """Update metrics with a single prediction.
        
        Args:
            pred_probs: softmax probability distribution over vocab (shape: vocab_size)
            y_idx: ground truth character index
        """
        self.total += 1
        
        # Top-1 accuracy
        if pred_probs.argmax().item() == y_idx:
            self.correct += 1
        
        # Top-3 accuracy
        topk_size = min(3, self.vocab_size)
        top3 = pred_probs.topk(topk_size).indices.tolist()
        if y_idx in top3:
            self.correct_top3 += 1
        
        # BPC (bits per character)
        p_y = pred_probs[y_idx].item()
        self.total_log_prob += math.log2(max(p_y, self.eps))
    
    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)
    
    @property
    def top3_accuracy(self) -> float:
        return self.correct_top3 / max(self.total, 1)
    
    @property
    def bpc(self) -> float:
        """Bits per character. Lower is better."""
        return -self.total_log_prob / max(self.total, 1)
    
    def summary(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'top3_accuracy': self.top3_accuracy,
            'bpc': self.bpc,
            'n_chars': self.total,
        }


def compute_forgetting(accuracy_matrix: List[List[float]]) -> List[float]:
    """Compute per-domain forgetting from an accuracy matrix.
    
    accuracy_matrix[i][j] = accuracy on domain j after training on domain i.
    Forgetting for domain j = max accuracy on j across rows 0..j  minus  final accuracy on j.
    """
    n_domains = len(accuracy_matrix)
    forgetting = []
    for j in range(n_domains):
        best_acc = max(accuracy_matrix[i][j] for i in range(j + 1))
        final_acc = accuracy_matrix[-1][j]
        forgetting.append(best_acc - final_acc)
    return forgetting


def compute_bpc_forgetting(bpc_matrix: List[List[float]]) -> List[float]:
    """Compute per-domain BPC forgetting. Forgetting = increase in BPC.
    
    bpc_matrix[i][j] = BPC on domain j after training on domain i.
    Forgetting for domain j = final BPC on j  minus  best (lowest) BPC on j.
    """
    n_domains = len(bpc_matrix)
    forgetting = []
    for j in range(n_domains):
        best_bpc = min(bpc_matrix[i][j] for i in range(j + 1))
        final_bpc = bpc_matrix[-1][j]
        forgetting.append(final_bpc - best_bpc)
    return forgetting


def compute_growth_efficiency(accuracy_before: float, accuracy_after: float,
                               units_born: int) -> float:
    """Accuracy gain per new unit born."""
    if units_born == 0:
        return 0.0
    return (accuracy_after - accuracy_before) / units_born
