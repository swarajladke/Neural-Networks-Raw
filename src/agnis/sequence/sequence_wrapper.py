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
            use_softmax_output=getattr(config.model, "use_softmax_output", False),
            use_fatigue=getattr(config.model, "use_fatigue", False),
            fatigue_decay=getattr(config.model, "fatigue_decay", 0.9),
            gamma_fatigue=getattr(config.model, "gamma_fatigue", 0.5),
            use_precision_gating=getattr(config.model, "use_precision_gating", False),
            gate_alpha_min=getattr(config.model, "gate_alpha_min", 0.2),
            gate_alpha_max=getattr(config.model, "gate_alpha_max", 0.8),
            gate_beta=getattr(config.model, "gate_beta", 1.0),
            gate_ema=getattr(config.model, "gate_ema", 0.05),
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


class SimpleGRUBaseline(SequenceModel):
    """Standard PyTorch Gated Recurrent Unit baseline."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 32, lr: float = 0.01):
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.use_memory = False
        self.use_replay = False

        self.rnn = nn.GRUCell(d_in, d_hidden)
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


class DeepSeqAgnisModel(SequenceModel):
    """Sequence prediction wrapper for the multi-layer PredictiveHierarchy system (Phase 6)."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        config: AGNISConfig,
        use_recurrent: bool = True,
        use_memory: bool = True,
        use_replay: bool = True,
    ):
        from agnis.core.predictive_hierarchy import PredictiveHierarchy
        from agnis.memory.fast_memory import FastMemory
        from agnis.memory.replay_buffer import ReplayBuffer, LatentReplayBuffer
        from agnis.training.sleep_trainer import SleepTrainer
        from agnis.neurogenesis.growth_controller import GrowthController
        from agnis.neurogenesis.routing import NoveltyAttributor
        import math

        self.d_in_x = d_in
        self.d_out_y = d_out
        self.use_memory = use_memory
        self.use_replay = use_replay
        self.step_counter = 0
        self.current_task_id = 0
        self.total_births = 0
        self.total_prunes = 0
        self.config = config

        # 1. Initialize PredictiveHierarchy
        self.hierarchy = PredictiveHierarchy(
            d_input=d_in + d_out,
            layer_dims=config.hierarchy.layer_dims,
            k_sparse_per_layer=config.hierarchy.k_sparse_per_layer,
            commit_strides=config.hierarchy.commit_strides,
            lambda_td=config.hierarchy.lambda_td,
            n_settle=config.hierarchy.n_settle,
            eta_z=config.model.eta_z,
            eta_d=config.model.eta_D,
            eta_e=config.model.eta_E,
            eta_r=config.model.eta_R,
            lr_decay=config.hierarchy.learning_rate_decay,
            use_precision_gating=config.model.use_precision_gating,
            use_fatigue=config.model.use_fatigue,
            fatigue_decay=config.model.fatigue_decay,
            gamma_fatigue=config.model.gamma_fatigue,
            importance_decay=config.model.importance_decay,
        )

        # 2. FastMemory and Replay buffers (layer 0 stimulus-level)
        self.fast_mem = FastMemory(capacity=config.memory.capacity) if use_memory else None
        self.replay_buffer = ReplayBuffer(max_size=config.memory.replay_buffer_size) if use_replay else None
        self.latent_replay_buffer = LatentReplayBuffer(max_per_domain=64) if use_replay else None

        # 3. Sleep Trainer
        self.sleep_trainer = None
        if use_replay:
            self.sleep_trainer = SleepTrainer(
                model=self.hierarchy,
                fast_memory=self.fast_mem,
                replay_buffer=self.replay_buffer,
                sleep_lr_scale=config.training.sleep_lr_scale,
                importance_protect_threshold=config.training.importance_protect_threshold,
                latent_replay_buffer=self.latent_replay_buffer,
            )

        # 4. Neurogenesis Controllers
        self.neurogenesis_enabled = config.neurogenesis.enabled
        if self.neurogenesis_enabled:
            self.gcs = [
                GrowthController(
                    alpha=config.neurogenesis.alpha,
                    beta=config.neurogenesis.beta,
                    gamma=config.neurogenesis.gamma,
                    delta=config.neurogenesis.delta,
                    kappa=config.neurogenesis.kappa,
                    lambda_cost=config.neurogenesis.lambda_cost,
                    threshold=config.neurogenesis.threshold,
                    consecutive_n=config.neurogenesis.consecutive_n,
                )
                for _ in range(self.hierarchy.n_layers)
            ]
            self.routing = NoveltyAttributor(
                n_layers=self.hierarchy.n_layers,
                max_units_per_layer=config.hierarchy.max_units_per_layer,
            )
        else:
            self.gcs = []
            self.routing = None

        self.last_retrieval_similarity = 0.5
        self.observed_mask = torch.cat([torch.ones(d_in), torch.zeros(d_out)])

    def reset_sequence_state(self):
        self.hierarchy.reset_state()

    def start_task(self, task_id: int):
        self.current_task_id = task_id

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        import math
        self.step_counter += 1
        s_joint = torch.cat([x, y])

        # 1. Forward pass (settling)
        z_settled = self.hierarchy.forward(s_joint, t=self.step_counter)

        # 2. Compute prediction error
        s_pred = self.hierarchy.get_output_prediction()
        error_mse = ((s_joint - s_pred) ** 2).mean().item()

        # 3. Memory write check
        self.last_retrieval_similarity = 0.5
        if self.use_memory and self.fast_mem is not None:
            retrieved = self.fast_mem.retrieve(z_settled[0], update_importance=False)
            if retrieved is not None:
                _, sim, _ = retrieved
                self.last_retrieval_similarity = sim
            else:
                self.last_retrieval_similarity = 0.0

            # FastMemory.write internally checks novelty and error thresholds
            self.fast_mem.write(
                key=z_settled[0].detach().clone(),
                value=s_joint.detach().clone(),
                error_val=error_mse,
                task_id=self.current_task_id
            )

        # 4. Weight update
        metrics = self.hierarchy.update_all_weights(s_joint, t=self.step_counter)

        # 5. Latent snapshot recording (every 8th step)
        if self.use_replay and self.latent_replay_buffer is not None and self.step_counter % 8 == 0:
            self.latent_replay_buffer.add(
                self.current_task_id,
                [self.hierarchy._z(l) for l in range(self.hierarchy.n_layers)]
            )

        # 6. Neurogenesis growth check
        if self.neurogenesis_enabled:
            per_layer_trigger = []
            for l in range(self.hierarchy.n_layers):
                current_capacity = self.hierarchy.layer_dims[l]
                novelty = 1.0 - self.last_retrieval_similarity
                error_l2 = math.sqrt(self.hierarchy.per_layer_prediction_error[l])

                trigger = self.gcs[l].update(
                    error=error_l2,
                    novelty=novelty,
                    uncertainty=0.0,
                    interference=0.0,
                    coverage=0.0,
                    cost=float(current_capacity),
                )
                per_layer_trigger.append(trigger)

            # Novelty routing decision
            self.routing.update(self.hierarchy.per_layer_prediction_error)
            target_layer = self.routing.route_growth(per_layer_trigger, self.hierarchy.layer_dims)
            if target_layer >= 0:
                # Find input/error for birth initialization
                if target_layer == 0:
                    current_input = s_joint
                    residual_error = s_joint - s_pred
                else:
                    current_input = self.hierarchy._compute_bottom_up_error(target_layer, s_joint)
                    residual_error = current_input

                self.hierarchy.grow_units(target_layer, current_input, residual_error)
                self.total_births += 1
                print(f"  [Hierarchy Neurogenesis] Birth at layer {target_layer}. Capacity: {self.hierarchy.layer_dims}")

        return {"error": error_mse, **metrics}

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        # Construct joint query
        zeros_target = torch.zeros(self.d_out_y, device=x.device)
        s_query = torch.cat([x, zeros_target])

        # Forward pass on query with observed mask
        _ = self.hierarchy.forward(s_query, t=self.step_counter, observed_mask=self.observed_mask.to(x.device))

        # Get output prediction target slice
        pred_joint = self.hierarchy.get_output_prediction()
        pred_target = pred_joint[self.d_in_x:]

        # Softmax normalized prediction
        return torch.softmax(pred_target, dim=-1)

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        # Save state backups
        z_backups = [self.hierarchy._z(l).clone() for l in range(self.hierarchy.n_layers)]
        z_prev_backups = [self.hierarchy._z_prev(l).clone() for l in range(self.hierarchy.n_layers)]
        err_accum_backups = [self.hierarchy._error_accum(l).clone() for l in range(self.hierarchy.n_layers)]
        err_count_backups = [self.hierarchy._error_count(l).clone() for l in range(self.hierarchy.n_layers)]

        pred = self.predict_transition(x)

        # Restore
        for l in range(self.hierarchy.n_layers):
            self.hierarchy._set_z(l, z_backups[l])
            self.hierarchy._set_z_prev(l, z_prev_backups[l])
            self.hierarchy._set_error_accum(l, err_accum_backups[l])
            self.hierarchy._set_error_count(l, err_count_backups[l])

        return pred

    def advance_state_only(self, x: torch.Tensor, y: torch.Tensor):
        s_joint = torch.cat([x, y])
        _ = self.hierarchy.forward(s_joint, t=self.step_counter)

    def sleep(self) -> Dict[str, Any]:
        if self.use_replay and self.sleep_trainer is not None and self.fast_mem is not None:
            self.replay_buffer.add_from_memory(self.fast_mem, n=self.config.training.n_sleep_replay)
            errors = self.sleep_trainer.sleep(n_replay=self.config.training.n_sleep_replay, n_steps=self.config.training.n_sleep_steps)
            if errors:
                return {"sleep_start_error": errors[0], "sleep_end_error": errors[-1]}
        return {}

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "step": self.step_counter,
            "layer_dims": list(self.hierarchy.layer_dims),
            "total_births": self.total_births,
            "total_prunes": self.total_prunes,
        }
        for l in range(self.hierarchy.n_layers):
            stats[f"layer{l}/error"] = self.hierarchy.per_layer_prediction_error[l] if l < len(self.hierarchy.per_layer_errors) else 0.0
            stats[f"layer{l}/usage_mean"] = self.hierarchy._usage(l).mean().item()
            stats[f"layer{l}/maturity_mean"] = self.hierarchy._maturity(l).mean().item()
        return stats

