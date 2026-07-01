"""
Raw AGNIS — src/agnis/evaluation/baselines.py

Baseline models for comparison against Raw AGNIS in continual learning experiments.

Baselines:
  1. NaiveMLP          — Sequential MLP trained with standard backprop (no memory)
  2. DenseHebbian      — Dense associative memory with Hebbian updates (no sparsity)
  3. SimpleRNN         — Simple RNN for sequence tasks

All baselines implement the same interface:
  - train_on(s, target) — one training step
  - predict(s) → output — inference
  - reset_state() — reset temporal state (for RNNs)

Design notes:
  - NaiveMLP uses PyTorch autograd (legitimate — it's a baseline, not the core model).
  - DenseHebbian uses manual Hebbian updates (like Raw AGNIS core, but without sparsity).
  - These are for comparison only. Keep them minimal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class NaiveMLP(nn.Module):
    """
    Naive sequential MLP baseline.

    Trained with standard backprop (Adam optimizer) on each input-target pair.
    No memory, no replay, no sparsity. Pure catastrophic forgetting baseline.

    This is the "worst case" baseline — it should have the highest forgetting.

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_out : int
        Output dimension.
    d_hidden : int
        Hidden layer size.
    lr : float
        Learning rate for Adam optimizer.
    """

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 64, lr: float = 0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

    def train_on(self, s: torch.Tensor, target: torch.Tensor) -> float:
        """One gradient update step."""
        self.optimizer.zero_grad()
        pred = self.forward(s.unsqueeze(0)).squeeze(0)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, s: torch.Tensor) -> torch.Tensor:
        """Inference (no gradient)."""
        with torch.no_grad():
            return self.forward(s.unsqueeze(0)).squeeze(0)

    def reset_state(self):
        """No temporal state to reset."""
        pass


class DenseHebbian(nn.Module):
    """
    Dense Hebbian associative memory baseline.

    Uses the same core Hebbian update as Raw AGNIS (local, no backprop)
    but WITHOUT kWTA sparsity. Acts as the "no sparsity" ablation.

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_z : int
        Latent/association dimension.
    eta : float
        Hebbian learning rate.
    n_settle : int
        Number of settling iterations.
    """

    def __init__(self, d_in: int, d_z: int, eta: float = 0.01, n_settle: int = 5):
        super().__init__()
        self.d_in = d_in
        self.d_z = d_z
        self.eta = eta
        self.n_settle = n_settle

        # Hebbian weight matrix (d_in x d_z)
        self.W = torch.randn(d_in, d_z) * 0.01
        self.z = torch.zeros(d_z)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Forward: encode s through W, settle."""
        z = self.W.T @ s
        return torch.tanh(z)

    def train_on(self, s: torch.Tensor, target: torch.Tensor) -> float:
        """Hebbian update: ΔW = η * outer(s, a)."""
        a = self.forward(s)
        e = target - self.W @ a  # reconstruction error
        delta_W = self.eta * torch.outer(e, a)
        self.W = self.W + delta_W
        return (e ** 2).mean().item()

    def predict(self, s: torch.Tensor) -> torch.Tensor:
        """Predict: reconstruct target from s."""
        a = self.forward(s)
        return self.W @ a

    def reset_state(self):
        """No temporal state."""
        self.z = torch.zeros(self.d_z)


class SimpleRNN(nn.Module):
    """
    Simple Elman RNN baseline for sequence tasks.

    Trained with truncated BPTT (backprop through time, 1 step).
    Used as baseline for Phase 2 sequence tasks.

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_hidden : int
        Hidden state size.
    d_out : int
        Output dimension.
    lr : float
        Learning rate for Adam.
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int, lr: float = 0.01):
        super().__init__()
        self.d_hidden = d_hidden
        self.rnn_cell = nn.RNNCell(d_in, d_hidden)
        self.output = nn.Linear(d_hidden, d_out)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.h = torch.zeros(1, d_hidden)  # (1, d_hidden)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """One RNN step."""
        self.h = self.rnn_cell(s.unsqueeze(0), self.h)
        return self.output(self.h).squeeze(0)

    def train_on(self, s: torch.Tensor, target: torch.Tensor) -> float:
        """One BPTT step (single timestep)."""
        self.optimizer.zero_grad()
        # Detach h to do truncated BPTT (1-step)
        self.h = self.h.detach()
        pred = self.forward(s)
        loss = self.loss_fn(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, s: torch.Tensor) -> torch.Tensor:
        """Inference."""
        with torch.no_grad():
            return self.forward(s)

    def reset_state(self):
        """Reset hidden state (call between sequences/tasks)."""
        self.h = torch.zeros(1, self.d_hidden)


# ── Unified Associative Interface & Wrappers ──────────────────────────────────

class AssociativeModel:
    """Base class for all models participating in the continual learning benchmark."""
    
    def train_pair(self, x: torch.Tensor, y: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> dict:
        raise NotImplementedError

    def predict(self, x: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def sleep(self) -> dict:
        raise NotImplementedError

    def get_latent(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def start_task(self, task_idx: int):
        pass

    def get_stats(self) -> dict:
        raise NotImplementedError


class NaiveMLPBaseline(AssociativeModel):
    """Wrapper for standard backprop MLP baseline."""
    
    def __init__(self, d_in: int, d_out: int, d_hidden: int = 64, lr: float = 0.01):
        self.mlp = NaiveMLP(d_in, d_out, d_hidden, lr)
        self.d_in = d_in
        self.d_out = d_out

    def train_pair(self, x: torch.Tensor, y: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> dict:
        if task_context is not None:
            x = torch.cat([x, task_context])
        loss = self.mlp.train_on(x, y)
        return {"error": loss}

    def predict(self, x: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task_context is not None:
            x = torch.cat([x, task_context])
        return self.mlp.predict(x)

    def sleep(self) -> dict:
        return {}

    def get_latent(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task_context is not None:
            x = torch.cat([x, task_context])
        with torch.no_grad():
            # First linear activation as latent proxy
            return torch.tanh(self.mlp.net[0](x))

    def get_stats(self) -> dict:
        return {}


class DenseHebbianBaseline(AssociativeModel):
    """Wrapper for dense Hebbian auto-association baseline."""
    
    def __init__(self, d_in: int, d_out: int, eta: float = 0.01, n_settle: int = 5):
        self.hebb = DenseHebbian(d_in, d_out, eta, n_settle)
        self.d_in = d_in
        self.d_out = d_out

    def train_pair(self, x: torch.Tensor, y: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> dict:
        if task_context is not None:
            x = torch.cat([x, task_context])
        loss = self.hebb.train_on(x, y)
        return {"error": loss}

    def predict(self, x: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task_context is not None:
            x = torch.cat([x, task_context])
        return self.hebb.predict(x)

    def sleep(self) -> dict:
        return {}

    def get_latent(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task_context is not None:
            x = torch.cat([x, task_context])
        with torch.no_grad():
            return self.hebb.forward(x)

    def get_stats(self) -> dict:
        return {}


class AgnisBaseline(AssociativeModel):
    """Generic wrapper for all Raw AGNIS variants."""
    
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_z: int,
        k_sparse: int = 3,
        n_settle: int = 10,
        eta_z: float = 0.05,
        eta_D: float = 0.01,
        eta_E: float = 0.01,
        eta_R: float = 0.005,
        rho: float = 0.3,
        lambda_lat: float = 0.1,
        lambda_sparse: float = 0.01,
        use_sparsity: bool = True,
        use_recurrent: bool = False,
        use_lateral: bool = False,
        importance_decay: float = 0.01,
        use_memory: bool = True,
        use_replay: bool = True,
        memory_capacity: int = 128,
        write_error_threshold: float = 0.2,
        write_novelty_threshold: float = 0.15,
        replay_buffer_size: int = 64,
        sleep_lr_scale: float = 0.3,
        importance_protect_threshold: float = 0.5,
        n_sleep_steps: int = 1,
        n_sleep_replay: int = 16,
        maturity_enabled: bool = True,
        max_latent_dim: int = 128,
    ):
        self.d_in_x = d_in
        self.d_out_y = d_out
        self.joint_dim = d_in + d_out

        # Initialize PredictiveCell with joint_dim
        from agnis.core.predictive_cell import PredictiveCell
        self.cell = PredictiveCell(
            d_in=self.joint_dim,
            d_z=d_z,
            k_sparse=k_sparse,
            n_settle=n_settle,
            eta_z=eta_z,
            eta_D=eta_D,
            eta_E=eta_E,
            eta_R=eta_R,
            rho=rho,
            lambda_lat=lambda_lat,
            lambda_sparse=lambda_sparse,
            use_sparsity=use_sparsity,
            use_recurrent=use_recurrent,
            use_lateral=use_lateral,
            importance_decay=importance_decay,
            maturity_enabled=maturity_enabled,
            max_latent_dim=max_latent_dim,
        )

        # Observed mask for prediction (1 on x/context, 0 on y)
        self.observed_mask = torch.zeros(self.joint_dim)
        self.observed_mask[:self.d_in_x] = 1.0

        # Memory and Replay setup
        self.use_memory = use_memory
        self.use_replay = use_replay

        from agnis.memory.fast_memory import FastMemory
        from agnis.memory.replay_buffer import ReplayBuffer
        from agnis.training.sleep_trainer import SleepTrainer

        if self.use_memory:
            self.fast_mem = FastMemory(
                capacity=memory_capacity,
                write_error_threshold=write_error_threshold,
                write_novelty_threshold=write_novelty_threshold,
            )
            self.replay_buf = ReplayBuffer(max_size=replay_buffer_size)
        else:
            self.fast_mem = None
            self.replay_buf = None

        if self.use_replay and self.use_memory:
            self.sleep_trainer = SleepTrainer(
                model=self.cell,
                replay_buffer=self.replay_buf,
                sleep_lr_scale=sleep_lr_scale,
                importance_protect_threshold=importance_protect_threshold,
            )
        else:
            self.sleep_trainer = None

        self.n_sleep_steps = n_sleep_steps
        self.n_sleep_replay = n_sleep_replay

        # Track retrieval stats
        self.memory_retrievals = 0
        self.memory_hits = 0
        self.last_retrieval_similarity = 0.0

        # Diagnostic metrics per task
        self.current_task_idx = 0
        self.memory_writes_per_task = [0]
        self.memory_retrievals_per_task = [0]
        self.memory_hits_per_task = [0]
        self.retrieval_similarities = []
        self.replay_steps_executed = 0
        self.replay_error_delta = 0.0

    def start_task(self, task_idx: int):
        self.current_task_idx = task_idx
        while len(self.memory_writes_per_task) <= task_idx:
            self.memory_writes_per_task.append(0)
            self.memory_retrievals_per_task.append(0)
            self.memory_hits_per_task.append(0)

    def train_pair(self, x: torch.Tensor, y: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> dict:
        if task_context is not None:
            x = torch.cat([x, task_context])

        # Joint representation
        s_joint = torch.cat([x, y])

        # Settle on full joint representation
        a = self.cell.forward(s_joint)
        metrics = self.cell.update_weights(s_joint, a)
        error_val = self.cell.prediction_error or 0.0

        # Memory write
        if self.use_memory and self.fast_mem is not None:
            self.fast_mem.tick()
            written = self.fast_mem.write(
                key=a.detach().clone(),
                value=s_joint.detach().clone(),
                error_val=error_val,
                task_id=self.current_task_idx,
            )
            if written:
                self.memory_writes_per_task[self.current_task_idx] += 1

        return {"error": error_val, **metrics}

    def predict(self, x: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task_context is not None:
            x = torch.cat([x, task_context])

        # Joint input for prediction (target slice is zeros)
        zeros_target = torch.zeros(self.d_out_y)
        s_query = torch.cat([x, zeros_target])

        # Forward pass using observed_mask
        a = self.cell.forward(s_query, observed_mask=self.observed_mask)

        # Default prediction from weights
        pred_joint = self.cell.D @ a
        pred_target = pred_joint[self.d_in_x:]

        # Hybrid lookup retrieval (optional validation)
        if self.use_memory and self.fast_mem is not None:
            self.memory_retrievals_per_task[self.current_task_idx] += 1
            retrieved = self.fast_mem.retrieve(a, update_importance=False)
            self.memory_retrievals += 1
            if retrieved is not None:
                val, sim, entry = retrieved
                self.last_retrieval_similarity = sim
                self.retrieval_similarities.append(sim)
                if sim > 0.8:
                    self.memory_hits += 1
                    self.memory_hits_per_task[self.current_task_idx] += 1
                    # Blend or override if memory retrieval is extremely strong
                    if sim > 0.95:
                        pred_target = val[self.d_in_x:]

        return pred_target

    def sleep(self) -> dict:
        if self.use_replay and self.sleep_trainer is not None and self.replay_buf is not None and self.fast_mem is not None:
            # Populate buffer
            self.replay_buf.add_from_memory(self.fast_mem, n=self.n_sleep_replay)
            errors = self.sleep_trainer.sleep(n_replay=self.n_sleep_replay, n_steps=self.n_sleep_steps)
            if errors:
                start_err = errors[0]
                end_err = errors[-1]
                self.replay_error_delta += (start_err - end_err)
                self.replay_steps_executed += len(errors)
            mean_error = sum(errors) / len(errors) if errors else 0.0
            return {"sleep_mean_error": mean_error, "sleep_steps": len(errors)}
        return {}

    def get_latent(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if task_context is not None:
            x = torch.cat([x, task_context])

        if y is not None:
            s_joint = torch.cat([x, y])
            a = self.cell.forward(s_joint)
        else:
            zeros_target = torch.zeros(self.d_out_y)
            s_query = torch.cat([x, zeros_target])
            a = self.cell.forward(s_query, observed_mask=self.observed_mask)

        return a

    def get_stats(self) -> dict:
        stats = {
            "sparsity_level": self.cell.sparsity_level,
            "prediction_error": self.cell.prediction_error,
            "memory_size": self.fast_mem.size if self.fast_mem is not None else 0,
            "memory_writes_per_task": self.memory_writes_per_task,
            "memory_retrievals_per_task": self.memory_retrievals_per_task,
            "memory_hits_per_task": self.memory_hits_per_task,
            "mean_retrieval_similarity": sum(self.retrieval_similarities) / len(self.retrieval_similarities) if self.retrieval_similarities else 0.0,
            "replay_steps_executed": self.replay_steps_executed,
            "replay_error_delta": self.replay_error_delta,
        }
        if self.use_memory and self.fast_mem is not None:
            stats.update({
                "memory_utilization": self.fast_mem.utilization,
                "memory_hits": self.memory_hits,
                "memory_retrievals": self.memory_retrievals,
                "memory_hit_rate": self.memory_hits / max(1, self.memory_retrievals),
                "last_retrieval_similarity": self.last_retrieval_similarity,
            })
        return stats
