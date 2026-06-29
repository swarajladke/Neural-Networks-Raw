"""
Raw AGNIS — src/agnis/evaluation/continual_metrics.py

Implements metrics for continual learning evaluation:
- Forgetting tracker and average forgetting
- Backward Transfer (BWT) and Forward Transfer (FWT)
- Task classification accuracy computation
- Active fraction / sparsity tracking
- Memory statistics (hit rate, size, retrieval similarity)
"""

import torch
from typing import List, Dict, Optional, Tuple


def evaluate_model_on_task(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    context: Optional[torch.Tensor] = None,
) -> float:
    """
    Evaluate accuracy of a model on a specific AssociationTask.
    Uses discrete argmax-based classification accuracy.
    
    Parameters
    ----------
    model : AssociativeModel
    inputs : torch.Tensor of shape (num_pairs, d_in)
    targets : torch.Tensor of shape (num_pairs, d_out)
    context : torch.Tensor of shape (d_context,), optional
    
    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    correct = 0
    num_pairs = inputs.shape[0]
    
    for i in range(num_pairs):
        x = inputs[i]
        y_true = targets[i]
        
        with torch.no_grad():
            y_pred = model.predict(x, task_context=context)
            
        pred_idx = torch.argmax(y_pred).item()
        true_idx = torch.argmax(y_true).item()
        
        if pred_idx == true_idx:
            correct += 1
            
    return correct / num_pairs


def compute_phase1_metrics(
    accuracy_matrix: List[List[Optional[float]]],
    random_accuracy: float = 0.0833,  # 1/12 default for 12 classes
) -> dict:
    """
    Compute standard continual learning metrics from the accuracy matrix.
    
    Parameters
    ----------
    accuracy_matrix : list of list of float
        accuracy_matrix[ckpt_idx][task_idx] = accuracy on task_idx at checkpoint ckpt_idx.
        checkpoints are evaluated sequentially after training each task.
    random_accuracy : float
        Accuracy of a random model (used for FWT computation).
        
    Returns
    -------
    dict
        CL metrics (avg_forgetting, bwt, fwt, final_avg_accuracy, etc.)
    """
    if not accuracy_matrix:
        return {}
        
    num_checkpoints = len(accuracy_matrix)
    num_tasks = len(accuracy_matrix[0])
    
    # 1. Final Average Accuracy (at the last checkpoint)
    final_row = accuracy_matrix[-1]
    final_accs = [acc for acc in final_row if acc is not None]
    final_avg_accuracy = sum(final_accs) / len(final_accs) if final_accs else 0.0
    
    # 2. Forgetting per task: peak_accuracy - final_accuracy
    # Only computed for tasks that finished training before the final checkpoint
    forgetting_per_task = {}
    forgetting_list = []
    
    for task_idx in range(num_tasks - 1):
        # find checkpoint where task was first trained (checkpoint index == task_idx)
        train_ckpt = task_idx
        if train_ckpt >= num_checkpoints:
            continue
            
        accuracies_after_train = []
        for ckpt in range(train_ckpt, num_checkpoints):
            val = accuracy_matrix[ckpt][task_idx]
            if val is not None:
                accuracies_after_train.append(val)
                
        if len(accuracies_after_train) >= 2:
            peak = max(accuracies_after_train)
            final = accuracies_after_train[-1]
            forgetting = peak - final
            forgetting_per_task[f"task_{task_idx}"] = forgetting
            forgetting_list.append(forgetting)
            
    avg_forgetting = sum(forgetting_list) / len(forgetting_list) if forgetting_list else 0.0
    
    # 3. Backward Transfer (BWT)
    # BWT = (1 / (N - 1)) * sum_{i=0}^{N-2} (A_{i, T} - A_{i, i})
    bwt_list = []
    for task_idx in range(num_tasks - 1):
        if task_idx < num_checkpoints:
            acc_at_train = accuracy_matrix[task_idx][task_idx]
            acc_final = accuracy_matrix[-1][task_idx]
            if acc_at_train is not None and acc_final is not None:
                bwt_list.append(acc_final - acc_at_train)
                
    bwt = sum(bwt_list) / len(bwt_list) if bwt_list else 0.0
    
    # 4. Forward Transfer (FWT)
    # FWT = (1 / (N - 1)) * sum_{i=1}^{N-1} (A_{i, i-1} - random_acc)
    # A_{i, i-1} is the accuracy on task i right before it starts training
    fwt_list = []
    for task_idx in range(1, num_tasks):
        prev_ckpt = task_idx - 1
        if prev_ckpt < num_checkpoints:
            acc_before_train = accuracy_matrix[prev_ckpt][task_idx]
            if acc_before_train is not None:
                fwt_list.append(acc_before_train - random_accuracy)
                
    fwt = sum(fwt_list) / len(fwt_list) if fwt_list else 0.0
    
    return {
        "final_average_accuracy": final_avg_accuracy,
        "average_forgetting": avg_forgetting,
        "forgetting_per_task": forgetting_per_task,
        "backward_transfer": bwt,
        "forward_transfer": fwt,
    }
