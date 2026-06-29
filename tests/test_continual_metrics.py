"""
Raw AGNIS — tests/test_continual_metrics.py

Unit tests for continual learning metrics calculation (BWT, FWT, accuracy).
"""

import pytest
from agnis.evaluation.continual_metrics import compute_phase1_metrics


def test_metrics_calculation_simple():
    # 3 tasks
    accuracy_matrix = [
        [0.9, None, None],   # ckpt 0
        [0.7, 0.85, None],   # ckpt 1
        [0.6, 0.80, 0.95],   # ckpt 2
    ]
    metrics = compute_phase1_metrics(accuracy_matrix, random_accuracy=0.1)
    
    # 1. Final average accuracy: average of last row [0.6, 0.80, 0.95] = 0.7833
    assert abs(metrics["final_average_accuracy"] - 0.7833) < 1e-4
    
    # 2. Forgetting:
    # Task 0: Peak after training is at ckpt 0 (0.9), final is 0.6. Forgetting = 0.3
    # Task 1: Peak after training is at ckpt 1 (0.85), final is 0.80. Forgetting = 0.05
    # Average forgetting = (0.3 + 0.05) / 2 = 0.175
    assert abs(metrics["average_forgetting"] - 0.175) < 1e-4
    assert abs(metrics["forgetting_per_task"]["task_0"] - 0.3) < 1e-4
    assert abs(metrics["forgetting_per_task"]["task_1"] - 0.05) < 1e-4
    
    # 3. BWT: ( (A_0,T - A_0,0) + (A_1,T - A_1,1) ) / 2
    # Task 0: 0.6 - 0.9 = -0.3
    # Task 1: 0.8 - 0.85 = -0.05
    # BWT = (-0.3 - 0.05) / 2 = -0.175
    assert abs(metrics["backward_transfer"] - (-0.175)) < 1e-4


def test_metrics_fwt():
    # FWT measures if training task 0 helps with task 1 before training task 1
    accuracy_matrix = [
        [0.8, 0.25, None],   # ckpt 0 (after task 0)
        [0.7, 0.90, 0.30],   # ckpt 1 (after task 1)
        [0.6, 0.80, 0.95],   # ckpt 2 (after task 2)
    ]
    # random accuracy is 0.1
    # FWT component 1: task 1 at ckpt 0 (0.25 - 0.1 = 0.15)
    # FWT component 2: task 2 at ckpt 1 (0.30 - 0.1 = 0.20)
    # Average FWT = (0.15 + 0.20) / 2 = 0.175
    metrics = compute_phase1_metrics(accuracy_matrix, random_accuracy=0.1)
    assert abs(metrics["forward_transfer"] - 0.175) < 1e-4
