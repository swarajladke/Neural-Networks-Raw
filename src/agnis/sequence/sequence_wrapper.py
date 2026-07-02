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

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next token without modifying internal state (default: delegates to predict_transition)."""
        return self.predict_transition(x)

    def advance_state_only(self, x: torch.Tensor, y: torch.Tensor):
        """Advance recurrent/internal state without updating weights (default: no-op)."""
        pass


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
        maturity_enabled: bool = True,
        max_latent_dim: int = 128,
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
            maturity_enabled=maturity_enabled,
            max_latent_dim=max_latent_dim,
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

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next token without modifying internal recurrent state.
        Used during evaluation to get predictions without side effects."""
        z_prev_backup = self.base_model.cell.z_prev.clone() if self.base_model.cell.z_prev is not None else None
        z_backup = self.base_model.cell.z.clone() if self.base_model.cell.z is not None else None

        pred = self.predict_transition(x)

        self.base_model.cell.z_prev = z_prev_backup
        self.base_model.cell.z = z_backup

        return pred

    def advance_state_only(self, x: torch.Tensor, y: torch.Tensor):
        """Advance recurrent state using true (x, y) pair without updating
        weights, memory, importance, or growth. Used during evaluation."""
        s_joint = torch.cat([x, y])
        _ = self.base_model.cell.forward(s_joint)
        # Do NOT call update_weights, do NOT write to memory


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

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        """Predict without modifying hidden state."""
        with torch.no_grad():
            h_backup = self.h.clone()
            x_in = x.unsqueeze(0)
            h_new = self.rnn(x_in, self.h)
            logits = self.fc(h_new)
            prob = torch.softmax(logits, dim=-1).squeeze(0)
            self.h = h_backup  # restore
            return prob

    def advance_state_only(self, x: torch.Tensor, y: torch.Tensor):
        """Advance hidden state with true input, no weight update."""
        with torch.no_grad():
            x_in = x.unsqueeze(0)
            self.h = self.rnn(x_in, self.h).detach()


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

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        """Predict without modifying history."""
        with torch.no_grad():
            history_backup = self.history.copy()
            window_in = self._get_window_input(x).unsqueeze(0)
            hidden = torch.relu(self.fc1(window_in))
            logits = self.fc2(hidden)
            prob = torch.softmax(logits, dim=-1).squeeze(0)
            self.history = history_backup  # restore
            return prob

    def advance_state_only(self, x: torch.Tensor, y: torch.Tensor):
        """Advance window history with true input."""
        self.history.append(x)
        if len(self.history) > self.context_window:
            self.history.pop(0)


class BigramBaseline(SequenceModel):
    """Empirical bigram frequency table baseline.

    Counts P(next_char | current_char) from training data.
    This is the maximum-likelihood reference for one-step context models.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.01):
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.use_memory = False
        self.use_replay = False
        # Count matrix: counts[i][j] = number of times char j follows char i
        self.counts = torch.zeros(vocab_size, vocab_size)
        self.total_per_context = torch.zeros(vocab_size)

    def reset_sequence_state(self):
        pass  # stateless

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        x_idx = x.argmax().item()
        y_idx = y.argmax().item()
        self.counts[x_idx, y_idx] += 1.0
        self.total_per_context[x_idx] += 1.0
        return {"error": 0.0}

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        x_idx = x.argmax().item()
        row = self.counts[x_idx] + self.smoothing
        total = row.sum()
        if total > 0:
            return row / total
        else:
            return torch.ones(self.vocab_size) / self.vocab_size

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_transition(x)

    def advance_state_only(self, x: torch.Tensor, y: torch.Tensor):
        pass

    def sleep(self) -> Dict[str, Any]:
        return {}

    def get_stats(self) -> Dict[str, Any]:
        nonzero = (self.counts > 0).sum().item()
        total = self.counts.sum().item()
        return {"nonzero_bigrams": nonzero, "total_observations": total}


class TrigramBaseline(SequenceModel):
    """Empirical trigram frequency table baseline.

    Counts P(next_char | prev_char, current_char) from training data.
    Reference for two-step context models.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.01):
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.use_memory = False
        self.use_replay = False
        # Trigram counts: indexed by (prev_idx * vocab_size + curr_idx)
        self.counts = {}  # dict of {(prev, curr): counts_tensor}
        self.prev_idx = None  # previous character index

    def reset_sequence_state(self):
        self.prev_idx = None

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        x_idx = x.argmax().item()
        y_idx = y.argmax().item()
        if self.prev_idx is not None:
            key = (self.prev_idx, x_idx)
            if key not in self.counts:
                self.counts[key] = torch.zeros(self.vocab_size)
            self.counts[key][y_idx] += 1.0
        self.prev_idx = x_idx
        return {"error": 0.0}

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        x_idx = x.argmax().item()
        if self.prev_idx is not None:
            key = (self.prev_idx, x_idx)
            if key in self.counts:
                row = self.counts[key] + self.smoothing
                return row / row.sum()
        # Fall back to uniform
        return torch.ones(self.vocab_size) / self.vocab_size

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_transition(x)  # prediction doesn't change state

    def advance_state_only(self, x: torch.Tensor, y: torch.Tensor):
        x_idx = x.argmax().item()
        self.prev_idx = x_idx

    def sleep(self) -> Dict[str, Any]:
        return {}

    def get_stats(self) -> Dict[str, Any]:
        total_contexts = len(self.counts)
        total_obs = sum(c.sum().item() for c in self.counts.values())
        return {"trigram_contexts": total_contexts, "total_observations": total_obs}
