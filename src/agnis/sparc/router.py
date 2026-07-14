"""
Raw AGNIS — src/agnis/sparc/router.py

SPARC v0.1 column routers.
Implements TaskIDOracleRouter and NearestPrototypeRouter.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TaskIDOracleRouter:
    """
    Task-ID Oracle Router.
    Directly routes to the column mapped to the ground truth task_id.
    """

    def __init__(self):
        pass

    def route(self, task_id: int, num_columns: int) -> int:
        """Selects column index using task_id (modular routing upper bound)."""
        # Map task_id directly to column index, capping at total available columns
        col_idx = min(task_id, num_columns - 1)
        return col_idx


class NearestPrototypeRouter:
    """
    Nearest Prototype Router.
    A non-oracle, nonparametric router that maps input vectors to columns
    based on distance to running task centroids (prototypes).
    """

    def __init__(self, d_latent: int, num_columns: int, decay: float = 0.99):
        self.d_latent = d_latent
        self.num_columns = num_columns
        self.decay = decay

        # Initialize prototypes as zero vectors
        self.register_buffer("prototypes", torch.zeros(num_columns, d_latent))
        # Keep track of count of updates per column
        self.register_buffer("counts", torch.zeros(num_columns))

    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Helper to store persistent state as buffers."""
        setattr(self, name, tensor)

    def route(self, z: torch.Tensor) -> int:
        """Finds the nearest column prototype to input z."""
        with torch.no_grad():
            z_det = z.detach()
            # If prototypes are uninitialized, route randomly or to column 0
            if torch.sum(self.counts) == 0:
                return 0

            # Compute Euclidean distances: shape (num_columns,)
            # z_det is (d_latent,), prototypes is (num_columns, d_latent)
            distances = torch.sum((self.prototypes - z_det) ** 2, dim=-1)
            # Return index of nearest prototype
            return torch.argmin(distances).item()

    def update_prototype(self, col_idx: int, z: torch.Tensor):
        """Update the running prototype centroid for column col_idx."""
        with torch.no_grad():
            z_det = z.detach()
            if self.counts[col_idx] == 0:
                self.prototypes[col_idx] = z_det
            else:
                # Running average update
                self.prototypes[col_idx] = self.decay * self.prototypes[col_idx] + (1 - self.decay) * z_det

            self.counts[col_idx] += 1
