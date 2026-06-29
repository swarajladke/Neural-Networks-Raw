"""
Raw AGNIS — src/agnis/sequence/sequence_wrapper.py

Model wrappers for Seq AGNIS and MLP/RNN baselines.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from agnis.utils.config import AGNISConfig


class SequenceModel:
    """Base interface for all sequence learning models."""

    def reset_sequence_state(self):
        """Reset sequence context/hidden state between independent sequences."""
        pass

    def start_task(self, task_id: int):
        """Notify the model that a new task has started."""
        pass

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Train on a transition from x (current token) to y (next token)."""
        raise NotImplementedError

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next token distribution given current token x."""
        raise NotImplementedError

    def sleep(self) -> Dict[str, Any]:
        """Consolidate weights (offline sleep phase)."""
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve model diagnostics."""
        return {}


class SeqAgnisModel(SequenceModel):
    """Sequence prediction wrapper for the AgnisBaseline system."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_z: int,
        config: AGNISConfig,
        R_update_enabled: bool = True,
        R_drive_enabled: bool = True,
        use_recurrent: bool = True,
        use_memory: bool = True,
        use_replay: bool = True,
    ):
        from agnis.evaluation.baselines import AgnisBaseline
        # We initialize AgnisBaseline
        self.base_model = AgnisBaseline(
            d_in=d_in,
            d_out=d_out,
            d_z=d_z,
            k_sparse=config.model.k_sparse,
            n_settle=config.model.n_settle,
            eta_z=config.model.eta_z,
            eta_D=config.model.eta_D,
            eta_E=config.model.eta_E,
            eta_R=config.model.eta_R,
            rho=config.model.rho,
            lambda_lat=config.model.lambda_lat,
            lambda_sparse=config.model.lambda_sparse,
            use_sparsity=config.model.use_sparsity,
            use_recurrent=use_recurrent,
            use_lateral=config.model.use_lateral,
            importance_decay=config.model.importance_decay,
            use_memory=use_memory,
            use_replay=use_replay,
            memory_capacity=config.memory.capacity,
            write_error_threshold=config.memory.write_error_threshold,
            write_novelty_threshold=config.memory.write_novelty_threshold,
            replay_buffer_size=config.memory.replay_buffer_size,
            sleep_lr_scale=config.training.sleep_lr_scale,
            importance_protect_threshold=config.training.importance_protect_threshold,
            n_sleep_steps=config.training.n_sleep_steps,
            n_sleep_replay=config.training.n_sleep_replay,
        )

        # Override R flags
        self.base_model.cell.R_update_enabled = R_update_enabled
        self.base_model.cell.R_drive_enabled = R_drive_enabled
        self.use_memory = use_memory
        self.use_replay = use_replay

    def reset_sequence_state(self):
        self.base_model.cell.reset_state()

    def start_task(self, task_id: int):
        self.base_model.start_task(task_id)

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        return self.base_model.train_pair(x, y)

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        # Note: predict() handles evaluation guard internally
        raw_pred = self.base_model.predict(x)
        return torch.softmax(raw_pred, dim=-1)

    def sleep(self) -> Dict[str, Any]:
        return self.base_model.sleep()

    def get_stats(self) -> Dict[str, Any]:
        stats = self.base_model.get_stats()
        # Add recurrent R diagnostics
        stats.update({
            "rho": self.base_model.cell.rho if self.base_model.cell.use_recurrent else 0.0,
            "eta_R": self.base_model.cell.eta_R,
            "R_norm_final": torch.linalg.matrix_norm(self.base_model.cell.R, ord=2).item(),
            "R_update_norm_mean": sum(self.base_model.cell.R_update_norms) / len(self.base_model.cell.R_update_norms) if self.base_model.cell.R_update_norms else 0.0,
            "recurrent_drive_norm_mean": sum(self.base_model.cell.recurrent_drive_norms) / len(self.base_model.cell.recurrent_drive_norms) if self.base_model.cell.recurrent_drive_norms else 0.0,
        })
        return stats


class SimpleRNNBaseline(SequenceModel):
    """Standard PyTorch Recurrent Neural Network baseline."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 32, lr: float = 0.01):
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.use_memory = False
        self.use_replay = False

        self.rnn = nn.RNNCell(d_in, d_hidden)
        self.fc = nn.Linear(d_hidden, d_out)

        self.h = torch.zeros(1, d_hidden)
        self.optimizer = torch.optim.Adam(
            list(self.rnn.parameters()) + list(self.fc.parameters()), lr=lr
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def reset_sequence_state(self):
        self.h = torch.zeros(1, self.d_hidden)

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        self.optimizer.zero_grad()
        x_in = x.unsqueeze(0)  # (1, d_in)
        self.h = self.rnn(x_in, self.h)
        logits = self.fc(self.h)
        target = y.argmax().unsqueeze(0)
        loss = self.loss_fn(logits, target)
        loss.backward()
        self.optimizer.step()
        self.h = self.h.detach()
        return {"error": loss.item()}

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x_in = x.unsqueeze(0)
            self.h = self.rnn(x_in, self.h)
            logits = self.fc(self.h)
            prob = torch.softmax(logits, dim=-1).squeeze(0)
            return prob


class MLPWindowBaseline(SequenceModel):
    """MLP baseline that maps a sliding context window of history to predictions."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_window: int = 2,
        d_hidden: int = 64,
        lr: float = 0.01,
    ):
        self.d_in = d_in
        self.d_out = d_out
        self.context_window = context_window
        self.use_memory = False
        self.use_replay = False

        self.fc1 = nn.Linear(context_window * d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.optimizer = torch.optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()), lr=lr
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.history = []

    def reset_sequence_state(self):
        self.history.clear()

    def _get_window_input(self, x: torch.Tensor) -> torch.Tensor:
        self.history.append(x)
        if len(self.history) > self.context_window:
            self.history.pop(0)

        padded_history = self.history.copy()
        while len(padded_history) < self.context_window:
            padded_history.insert(0, torch.zeros_like(x))

        return torch.cat(padded_history)

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        self.optimizer.zero_grad()
        window_in = self._get_window_input(x).unsqueeze(0)  # (1, C * d_in)
        hidden = torch.relu(self.fc1(window_in))
        logits = self.fc2(hidden)
        target = y.argmax().unsqueeze(0)
        loss = self.loss_fn(logits, target)
        loss.backward()
        self.optimizer.step()
        return {"error": loss.item()}

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            window_in = self._get_window_input(x).unsqueeze(0)
            hidden = torch.relu(self.fc1(window_in))
            logits = self.fc2(hidden)
            prob = torch.softmax(logits, dim=-1).squeeze(0)
            return prob
