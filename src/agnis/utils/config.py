"""
Raw AGNIS — src/agnis/utils/config.py

Configuration loading and validation from YAML files.
All hyperparameters should be set via configs, not hardcoded.
"""

import yaml
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """PredictiveCell / PredictiveHierarchy configuration."""
    d_in: int = 12
    d_z: int = 32
    k_sparse: int = 3          # top-3 active out of 32 = ~9% sparsity
    n_settle: int = 10
    eta_z: float = 0.05
    eta_D: float = 0.01
    eta_E: float = 0.01
    eta_R: float = 0.005
    rho: float = 0.3
    lambda_lat: float = 0.1
    lambda_sparse: float = 0.01
    use_sparsity: bool = True
    use_recurrent: bool = False
    use_lateral: bool = False
    importance_decay: float = 0.01


@dataclass
class MemoryConfig:
    """FastMemory and ReplayBuffer configuration."""
    capacity: int = 256
    write_error_threshold: float = 0.3
    write_novelty_threshold: float = 0.2
    importance_decay: float = 0.999
    min_similarity_to_skip_write: float = 0.95
    replay_buffer_size: int = 128


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    n_tasks: int = 3
    pairs_per_task: int = 2
    n_repeats_per_task: int = 50
    n_sleep_steps: int = 1
    n_sleep_replay: int = 32
    sleep_lr_scale: float = 0.3
    importance_protect_threshold: float = 0.5
    eval_threshold: float = 0.3    # MSE threshold for "correct" in accuracy computation
    log_every: int = 10
    seed: int = 42
    overlap_context: bool = True
    sequences_per_task: int = 16
    sequence_length: int = 24
    epochs_per_task: int = 50
    sensitivity_mode: bool = False





@dataclass
class NeurogenesisConfig:
    """Growth controller configuration (Phase 3)."""
    enabled: bool = False
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.3
    delta: float = 0.4
    kappa: float = 0.3
    lambda_cost: float = 0.2
    threshold: float = 0.6
    consecutive_n: int = 10
    max_units: int = 64
    prune_interval: int = 100
    usage_threshold: float = 0.05
    importance_threshold: float = 0.01
    redundancy_threshold: float = 0.90
    maturity_floor: float = 0.05


@dataclass
class AGNISConfig:
    """Full Raw AGNIS experiment configuration."""
    experiment_name: str = "phase1_associative"
    phase: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    neurogenesis: NeurogenesisConfig = field(default_factory=NeurogenesisConfig)
    results_dir: str = "results/"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str) -> AGNISConfig:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to YAML config file.

    Returns
    -------
    AGNISConfig
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return AGNISConfig()

    # Parse nested configs
    config = AGNISConfig(
        experiment_name=raw.get("experiment_name", "unnamed"),
        phase=raw.get("phase", 1),
        results_dir=raw.get("results_dir", "results/"),
    )

    if "model" in raw:
        config.model = ModelConfig(**raw["model"])
    if "memory" in raw:
        config.memory = MemoryConfig(**raw["memory"])
    if "training" in raw:
        config.training = TrainingConfig(**raw["training"])
    if "neurogenesis" in raw:
        config.neurogenesis = NeurogenesisConfig(**raw["neurogenesis"])

    return config


def default_config() -> AGNISConfig:
    """Return the default configuration."""
    return AGNISConfig()
