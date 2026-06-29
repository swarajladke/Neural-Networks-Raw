"""
Raw AGNIS — src/agnis/training/__init__.py

Training subpackage:
  - OnlineTrainer: online sequential task training
  - SleepTrainer: offline replay-based consolidation
  - Curriculum: task sequencing and ordering
"""

from agnis.training.online_trainer import OnlineTrainer
from agnis.training.sleep_trainer import SleepTrainer

__all__ = [
    "OnlineTrainer",
    "SleepTrainer",
]
