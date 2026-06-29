"""
Raw AGNIS — tests/test_phase1_logging.py

Unit tests for result logging and schema validation.
"""

import os
import json
import shutil
import pytest
from agnis.evaluation.phase1_logging import save_phase1_run_results


def test_metrics_json_schema(tmp_path):
    # Setup dummy directory
    results_dir = str(tmp_path)
    
    # Required metrics schema keys
    required_keys = [
        "condition", "model", "seed", "num_tasks", "pairs_per_task", "latent_dim",
        "accuracy_matrix", "final_average_accuracy", "average_forgetting",
        "forgetting_per_task", "backward_transfer", "forward_transfer",
        "mean_prediction_error", "final_prediction_error", "mean_active_fraction",
        "final_memory_size", "memory_hit_rate", "representation_overlap_mean",
        "runtime_seconds", "uses_context", "input_dim", "target_dim",
        "joint_dim", "kwta_k", "memory_capacity", "train_steps_per_pair",
        "sleep_steps_per_task", "evaluation_mode"
    ]
    
    # Create mock metrics dictionary matching the schema
    mock_metrics = {key: 0 for key in required_keys}
    mock_metrics["condition"] = "orthogonal"
    mock_metrics["model"] = "agnis_kwta"
    mock_metrics["accuracy_matrix"] = [[1.0, 1.0]]
    mock_metrics["forgetting_per_task"] = {"task_0": 0.0}

    accuracy_matrix = [[1.0, 1.0]]
    forgetting_list = [0.0]
    prediction_errors = [0.1, 0.05]
    active_fractions = [0.125, 0.125]
    memory_usages = [1, 2]
    overlap_matrix = [[1.0, 0.0], [0.0, 1.0]]
    config_data = {"dummy_config": True}
    
    # Save results
    run_dir = save_phase1_run_results(
        results_dir=results_dir,
        condition="orthogonal",
        model_name="agnis_kwta",
        seed=0,
        metrics_dict=mock_metrics,
        accuracy_matrix=accuracy_matrix,
        forgetting_list=forgetting_list,
        prediction_errors=prediction_errors,
        active_fractions=active_fractions,
        memory_usages=memory_usages,
        overlap_matrix=overlap_matrix,
        config_data=config_data,
    )
    
    # Check directory structure exists
    assert os.path.exists(run_dir)
    assert os.path.exists(os.path.join(run_dir, "metrics.json"))
    assert os.path.exists(os.path.join(run_dir, "accuracy_matrix.csv"))
    assert os.path.exists(os.path.join(run_dir, "representation_overlap.csv"))
    
    # Load and validate json schema keys
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        loaded_metrics = json.load(f)
        
    for key in required_keys:
        assert key in loaded_metrics, f"Missing required metrics key: {key}"
