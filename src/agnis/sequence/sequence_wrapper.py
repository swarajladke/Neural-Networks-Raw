"""
Raw AGNIS — src/agnis/sequence/sequence_wrapper.py

Model wrappers for Seq AGNIS and MLP/RNN baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class RNNReplayBaseline(SimpleRNNBaseline):
    """RNN baseline with Sleep Replay Buffer (experience replay)."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 32, lr: float = 0.01, sleep_lr_scale: float = 0.3):
        super().__init__(d_in, d_out, d_hidden, lr)
        self.use_replay = True
        self.replay_buffer = []
        self.max_buffer_size = 128
        self.current_sequence = []
        self.sleep_lr_scale = sleep_lr_scale

    def reset_sequence_state(self):
        super().reset_sequence_state()
        if self.current_sequence:
            import random
            if len(self.replay_buffer) < self.max_buffer_size:
                self.replay_buffer.append(self.current_sequence)
            else:
                idx = random.randint(0, len(self.replay_buffer))
                if idx < self.max_buffer_size:
                    self.replay_buffer[idx] = self.current_sequence
            self.current_sequence = []

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        self.current_sequence.append(x.argmax().item())
        return super().train_transition(x, y)

    def sleep(self) -> Dict[str, Any]:
        if not self.replay_buffer:
            return {}
        import random
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.sleep_lr_scale

        losses = []
        n_sleep_steps = 30
        for _ in range(n_sleep_steps):
            seq = random.choice(self.replay_buffer)
            if len(seq) < 2:
                continue
            
            self.optimizer.zero_grad()
            h = torch.zeros(1, self.d_hidden, device=self.h.device)
            loss = 0.0
            for t in range(len(seq) - 1):
                x_t = torch.zeros(self.d_in, device=self.h.device)
                x_t[seq[t]] = 1.0
                x_in = x_t.unsqueeze(0)
                h = self.rnn(x_in, h)
                logits = self.fc(h)
                target = torch.tensor([seq[t+1]], device=self.h.device)
                loss += self.loss_fn(logits, target)
            
            loss = loss / (len(seq) - 1)
            loss.backward()
            # Gradient clipping to avoid exploding gradients during sleep BPTT
            torch.nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), max_norm=2.0)
            self.optimizer.step()
            losses.append(loss.item())

        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= self.sleep_lr_scale

        return {"sleep_mean_loss": sum(losses) / len(losses) if losses else 0.0}


class GRUReplayBaseline(SimpleGRUBaseline):
    """GRU baseline with Sleep Replay Buffer (experience replay)."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 32, lr: float = 0.01, sleep_lr_scale: float = 0.3):
        super().__init__(d_in, d_out, d_hidden, lr)
        self.use_replay = True
        self.replay_buffer = []
        self.max_buffer_size = 128
        self.current_sequence = []
        self.sleep_lr_scale = sleep_lr_scale

    def reset_sequence_state(self):
        super().reset_sequence_state()
        if self.current_sequence:
            import random
            if len(self.replay_buffer) < self.max_buffer_size:
                self.replay_buffer.append(self.current_sequence)
            else:
                idx = random.randint(0, len(self.replay_buffer))
                if idx < self.max_buffer_size:
                    self.replay_buffer[idx] = self.current_sequence
            self.current_sequence = []

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        self.current_sequence.append(x.argmax().item())
        return super().train_transition(x, y)

    def sleep(self) -> Dict[str, Any]:
        if not self.replay_buffer:
            return {}
        import random
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.sleep_lr_scale

        losses = []
        n_sleep_steps = 30
        for _ in range(n_sleep_steps):
            seq = random.choice(self.replay_buffer)
            if len(seq) < 2:
                continue
            
            self.optimizer.zero_grad()
            h = torch.zeros(1, self.d_hidden, device=self.h.device)
            loss = 0.0
            for t in range(len(seq) - 1):
                x_t = torch.zeros(self.d_in, device=self.h.device)
                x_t[seq[t]] = 1.0
                x_in = x_t.unsqueeze(0)
                h = self.rnn(x_in, h)
                logits = self.fc(h)
                target = torch.tensor([seq[t+1]], device=self.h.device)
                loss += self.loss_fn(logits, target)
            
            loss = loss / (len(seq) - 1)
            loss.backward()
            # Gradient clipping to avoid exploding gradients during sleep BPTT
            torch.nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), max_norm=2.0)
            self.optimizer.step()
            losses.append(loss.item())

        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= self.sleep_lr_scale

        return {"sleep_mean_loss": sum(losses) / len(losses) if losses else 0.0}


class RNNEWCBaseline(SimpleRNNBaseline):
    """RNN baseline with Elastic Weight Consolidation (EWC)."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 32, lr: float = 0.01, ewc_lambda: float = 100.0):
        super().__init__(d_in, d_out, d_hidden, lr)
        self.ewc_lambda = ewc_lambda
        self.task_optimal_params = []
        self.fishers = []
        self.current_task_sequences = []
        self.current_sequence = []

    def reset_sequence_state(self):
        super().reset_sequence_state()
        if self.current_sequence:
            self.current_task_sequences.append(self.current_sequence)
            self.current_sequence = []

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        self.current_sequence.append(x.argmax().item())
        
        self.optimizer.zero_grad()
        x_in = x.unsqueeze(0)
        self.h = self.rnn(x_in, self.h)
        logits = self.fc(self.h)
        target = y.argmax().unsqueeze(0)
        
        loss = self.loss_fn(logits, target)
        
        # Calculate EWC penalty
        ewc_penalty = torch.tensor(0.0, device=x.device)
        for task_idx in range(len(self.task_optimal_params)):
            opt_params = self.task_optimal_params[task_idx]
            fisher = self.fishers[task_idx]
            # Accumulate RNN parameters
            for name, param in self.rnn.named_parameters():
                key = f"rnn.{name}"
                if key in opt_params and key in fisher:
                    ewc_penalty += (fisher[key] * (param - opt_params[key]) ** 2).sum()
            # Accumulate FC parameters
            for name, param in self.fc.named_parameters():
                key = f"fc.{name}"
                if key in opt_params and key in fisher:
                    ewc_penalty += (fisher[key] * (param - opt_params[key]) ** 2).sum()
                    
        total_loss = loss + (self.ewc_lambda / 2.0) * ewc_penalty
        total_loss.backward()
        # Gradient clipping to avoid exploding gradients due to EWC penalty scale
        torch.nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), max_norm=2.0)
        self.optimizer.step()
        self.h = self.h.detach()
        
        return {"error": loss.item(), "ewc_penalty": ewc_penalty.item()}

    def start_task(self, task_id: int):
        # When a task transition occurs (task_id > 0), estimate Fisher and save optimal parameters for previous task
        if task_id > 0 and self.current_task_sequences:
            self.compute_ewc_task_data()
        self.current_task_sequences = []

    def compute_ewc_task_data(self):
        # Save copy of current parameters
        opt_params = {}
        for name, param in self.rnn.named_parameters():
            opt_params[f"rnn.{name}"] = param.clone().detach()
        for name, param in self.fc.named_parameters():
            opt_params[f"fc.{name}"] = param.clone().detach()
            
        self.task_optimal_params.append(opt_params)
        
        # Estimate diagonal of Fisher
        fisher = {}
        for name, param in self.rnn.named_parameters():
            fisher[f"rnn.{name}"] = torch.zeros_like(param)
        for name, param in self.fc.named_parameters():
            fisher[f"fc.{name}"] = torch.zeros_like(param)
        
        n_samples = 0
        for seq in self.current_task_sequences:
            if len(seq) < 2:
                continue
            n_samples += 1
            
            self.optimizer.zero_grad()
            h = torch.zeros(1, self.d_hidden, device=self.h.device)
            loss = 0.0
            
            for t in range(len(seq) - 1):
                x_t = torch.zeros(self.d_in, device=self.h.device)
                x_t[seq[t]] = 1.0
                x_in = x_t.unsqueeze(0)
                h = self.rnn(x_in, h)
                logits = self.fc(h)
                target = torch.tensor([seq[t+1]], device=self.h.device)
                loss += self.loss_fn(logits, target)
                
            loss = loss / (len(seq) - 1)
            loss.backward()
            # Gradient clipping to avoid exploding gradients during Fisher estimation BPTT
            torch.nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), max_norm=2.0)
            
            for name, param in self.rnn.named_parameters():
                if param.grad is not None:
                    fisher[f"rnn.{name}"] += param.grad.detach() ** 2
            for name, param in self.fc.named_parameters():
                if param.grad is not None:
                    fisher[f"fc.{name}"] += param.grad.detach() ** 2
                    
        for name in fisher:
            fisher[name] = fisher[name] / max(n_samples, 1)
            
        self.fishers.append(fisher)


class GRUEWCBaseline(SimpleGRUBaseline):
    """GRU baseline with Elastic Weight Consolidation (EWC)."""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 32, lr: float = 0.01, ewc_lambda: float = 100.0):
        super().__init__(d_in, d_out, d_hidden, lr)
        self.ewc_lambda = ewc_lambda
        self.task_optimal_params = []
        self.fishers = []
        self.current_task_sequences = []
        self.current_sequence = []

    def reset_sequence_state(self):
        super().reset_sequence_state()
        if self.current_sequence:
            self.current_task_sequences.append(self.current_sequence)
            self.current_sequence = []

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        self.current_sequence.append(x.argmax().item())
        
        self.optimizer.zero_grad()
        x_in = x.unsqueeze(0)
        self.h = self.rnn(x_in, self.h)
        logits = self.fc(self.h)
        target = y.argmax().unsqueeze(0)
        
        loss = self.loss_fn(logits, target)
        
        # Calculate EWC penalty
        ewc_penalty = torch.tensor(0.0, device=x.device)
        for task_idx in range(len(self.task_optimal_params)):
            opt_params = self.task_optimal_params[task_idx]
            fisher = self.fishers[task_idx]
            # Accumulate GRU parameters
            for name, param in self.rnn.named_parameters():
                key = f"rnn.{name}"
                if key in opt_params and key in fisher:
                    ewc_penalty += (fisher[key] * (param - opt_params[key]) ** 2).sum()
            # Accumulate FC parameters
            for name, param in self.fc.named_parameters():
                key = f"fc.{name}"
                if key in opt_params and key in fisher:
                    ewc_penalty += (fisher[key] * (param - opt_params[key]) ** 2).sum()
                    
        total_loss = loss + (self.ewc_lambda / 2.0) * ewc_penalty
        total_loss.backward()
        # Gradient clipping to avoid exploding gradients due to EWC penalty scale
        torch.nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), max_norm=2.0)
        self.optimizer.step()
        self.h = self.h.detach()
        
        return {"error": loss.item(), "ewc_penalty": ewc_penalty.item()}

    def start_task(self, task_id: int):
        if task_id > 0 and self.current_task_sequences:
            self.compute_ewc_task_data()
        self.current_task_sequences = []

    def compute_ewc_task_data(self):
        # Save copy of current parameters
        opt_params = {}
        for name, param in self.rnn.named_parameters():
            opt_params[f"rnn.{name}"] = param.clone().detach()
        for name, param in self.fc.named_parameters():
            opt_params[f"fc.{name}"] = param.clone().detach()
            
        self.task_optimal_params.append(opt_params)
        
        # Estimate diagonal of Fisher
        fisher = {}
        for name, param in self.rnn.named_parameters():
            fisher[f"rnn.{name}"] = torch.zeros_like(param)
        for name, param in self.fc.named_parameters():
            fisher[f"fc.{name}"] = torch.zeros_like(param)
        
        n_samples = 0
        for seq in self.current_task_sequences:
            if len(seq) < 2:
                continue
            n_samples += 1
            
            self.optimizer.zero_grad()
            h = torch.zeros(1, self.d_hidden, device=self.h.device)
            loss = 0.0
            
            for t in range(len(seq) - 1):
                x_t = torch.zeros(self.d_in, device=self.h.device)
                x_t[seq[t]] = 1.0
                x_in = x_t.unsqueeze(0)
                h = self.rnn(x_in, h)
                logits = self.fc(h)
                target = torch.tensor([seq[t+1]], device=self.h.device)
                loss += self.loss_fn(logits, target)
                
            loss = loss / (len(seq) - 1)
            loss.backward()
            # Gradient clipping to avoid exploding gradients during Fisher estimation BPTT
            torch.nn.utils.clip_grad_norm_(list(self.rnn.parameters()) + list(self.fc.parameters()), max_norm=2.0)
            
            for name, param in self.rnn.named_parameters():
                if param.grad is not None:
                    fisher[f"rnn.{name}"] += param.grad.detach() ** 2
            for name, param in self.fc.named_parameters():
                if param.grad is not None:
                    fisher[f"fc.{name}"] += param.grad.detach() ** 2
                    
        for name in fisher:
            fisher[name] = fisher[name] / max(n_samples, 1)
            
        self.fishers.append(fisher)


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


class SPARCSequenceWrapper(SequenceModel):
    """Sequence prediction wrapper for the SPARC v0.2 model."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_columns: int = 4,
        d_latent: int = 32,
        alpha: float = 0.01,
        beta: float = 0.5,
        eta_D: float = 0.01,
        eta_R: float = 0.01,
        eta_Q: float = 0.01,
        step_c: float = 0.5,
        n_settle: int = 15,
        routing_mode: str = "task_id_oracle",
        decay_factor: float = 0.9,
    ):
        from agnis.sparc.sparc_model import SPARCSequenceModel
        self.model = SPARCSequenceModel(
            num_columns=num_columns,
            d_input=d_in,
            d_latent=d_latent,
            d_output=d_out,
            alpha=alpha,
            beta=beta,
            eta_D=eta_D,
            eta_R=eta_R,
            eta_Q=eta_Q,
            step_c=step_c,
            n_settle=n_settle,
            routing_mode=routing_mode,
            decay_factor=decay_factor,
        )
        self.current_task_id = 0
        self.d_out = d_out
        self.use_memory = False
        self.use_replay = False

        # Router optimizer (trains only the router projection projection parameters)
        self.router_optimizer = None
        if routing_mode in [
            "supervised_router",
            "energy_distilled_router",
            "learned_router_no_distill",
            "learned_router_distill",
            "learned_router_mixture",
        ]:
            self.router_optimizer = torch.optim.Adam(self.model.router.parameters(), lr=1e-3)
            # Freeze column and head experts at initialization
            self.model.freeze_experts()

        # Calibration & Route replay distillation structures
        self.energy_calibration_buffers = {j: [] for j in range(num_columns)}
        self.route_replay_buffer = []

    def reset_sequence_state(self):
        self.model.reset_states()

    def start_task(self, task_id: int):
        # Boundary: Run calibration for the task we just completed (task_id - 1)
        if task_id > 0 and (task_id - 1) in self.energy_calibration_buffers:
            buffer_data = self.energy_calibration_buffers[task_id - 1]
            if len(buffer_data) > 0:
                # Compute robust Median / MAD calibration
                tensor_data = torch.tensor(buffer_data)
                med = tensor_data.median().item()
                mad = (tensor_data - med).abs().median().item()
                # Store in teacher and primary MinimumEnergyRouter calibration
                self.model.energy_teacher.set_calibration(
                    column_idx=task_id - 1,
                    median=med,
                    mad=mad,
                    n_samples=len(buffer_data)
                )
                if hasattr(self.model.router, "set_calibration"):
                    self.model.router.set_calibration(
                        column_idx=task_id - 1,
                        median=med,
                        mad=mad,
                        n_samples=len(buffer_data)
                    )
                print(f"[Calibration] Column {task_id - 1} Calibrated: Median={med:.4f}, MAD={mad:.4f}")
                self.energy_calibration_buffers[task_id - 1] = []

        self.current_task_id = task_id

    def train_transition(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        target_idx = y.argmax()

        # 1. Forward step through SPARC Model
        logits, diag = self.model.forward_step(
            z=x,
            target=target_idx,
            task_id=self.current_task_id,
            is_training=True
        )

        col_idx = diag.get("active_column", 0)

        # Collect settling energy for robust calibration later
        with torch.no_grad():
            # Get energy of winner column and add to calibration buffer
            # We decay previous state as prior
            prior_winner = self.model.decay_factor * self.model.h_prev[col_idx].clone()
            raw_energy = self.model.columns[col_idx].energy(x, self.model.h_prev[col_idx], prior_winner).item()
            self.energy_calibration_buffers[self.current_task_id].append(raw_energy)

        # 2. Replay cache update
        if self.model.routing_mode in [
            "supervised_router",
            "energy_distilled_router",
            "learned_router_no_distill",
            "learned_router_distill",
            "learned_router_mixture",
        ]:
            if len(self.route_replay_buffer) < 2000:
                if torch.rand(1).item() < 0.1:
                    with torch.no_grad():
                        raw_logits = self.model.router.contextual_logits(self.model.context_state)
                        self.route_replay_buffer.append({
                            "context": self.model.context_state.clone().detach(),
                            "teacher_raw_logits": raw_logits.clone().detach(),
                            "domain_id": self.current_task_id,
                        })

        # 3. Router optimization step
        router_loss_val = 0.0
        if self.router_optimizer is not None:
            self.router_optimizer.zero_grad()

            # 3.1 Base router objective loss
            if self.model.routing_mode == "supervised_router":
                raw_logits = self.model.router.contextual_logits(self.model.context_state)
                loss_objective = F.cross_entropy(raw_logits.unsqueeze(0), torch.tensor([self.current_task_id], device=x.device))

                # Load balancing: D_KL(p_batch || mean_routes)
                mean_routes = diag["routing_weights"]
                mean_routes = mean_routes / mean_routes.sum().clamp_min(1e-8)
                target_usage = torch.zeros(self.model.num_columns, device=x.device)
                target_usage[self.current_task_id] = 1.0
                balance_loss = F.kl_div(mean_routes.clamp_min(1e-8).log(), target_usage, reduction="sum")

                loss_total = loss_objective + 0.1 * balance_loss

            elif self.model.routing_mode == "energy_distilled_router":
                with torch.no_grad():
                    _, _, calibrated_energies, _ = self.model.energy_teacher.route_step(
                        x, list(self.model.h_prev), self.model.decay_factor
                    )
                    teacher_energies = torch.tensor(calibrated_energies, device=x.device)
                    q = torch.softmax(-teacher_energies / 1.0, dim=-1)

                raw_logits = self.model.router.contextual_logits(self.model.context_state)
                student_log_probs = torch.log_softmax(raw_logits, dim=-1)
                loss_objective = F.kl_div(student_log_probs, q, reduction="sum")

                # Load balancing: D_KL(q || mean_routes)
                mean_routes = diag["routing_weights"]
                mean_routes = mean_routes / mean_routes.sum().clamp_min(1e-8)
                balance_loss = F.kl_div(mean_routes.clamp_min(1e-8).log(), q, reduction="sum")

                loss_total = loss_objective + 0.1 * balance_loss

            elif self.model.routing_mode == "learned_router_mixture":
                # Router C loss: Task cross entropy loss on mixture log probability
                # logits is mixture_log_probs (already log_softmaxed)
                loss_objective = F.nll_loss(logits, target_idx.unsqueeze(0))
                loss_total = loss_objective

            else:  # learned_router_no_distill, learned_router_distill
                raw_logits = self.model.router.contextual_logits(self.model.context_state)
                loss_objective = F.cross_entropy(raw_logits.unsqueeze(0), torch.tensor([col_idx], device=x.device))
                loss_total = loss_objective

            # 3.2 Interleaved Cached Route Distillation
            distill_loss = 0.0
            if len(self.route_replay_buffer) > 0 and self.model.routing_mode in [
                "energy_distilled_router",
                "learned_router_distill",
                "learned_router_mixture",
            ]:
                indices = torch.randint(0, len(self.route_replay_buffer), (min(4, len(self.route_replay_buffer)),))
                replay_entries = [self.route_replay_buffer[idx] for idx in indices]

                dist_temp = 2.0
                for entry in replay_entries:
                    rep_context = entry["context"].to(x.device)
                    rep_teacher_logits = entry["teacher_raw_logits"].to(x.device)
                    rep_student_logits = self.model.router.contextual_logits(rep_context)

                    teacher_probs = torch.softmax(rep_teacher_logits / dist_temp, dim=-1)
                    student_log_probs = torch.log_softmax(rep_student_logits / dist_temp, dim=-1)

                    distill_loss += (dist_temp ** 2) * F.kl_div(
                        student_log_probs.unsqueeze(0),
                        teacher_probs.unsqueeze(0),
                        reduction="batchmean"
                    )
                distill_loss /= len(replay_entries)

            loss_total = loss_total + 1.0 * distill_loss
            loss_total.backward()
            self.router_optimizer.step()
            router_loss_val = loss_total.item()

        return {"error": diag.get("readout_loss", 0.0), "router_loss": router_loss_val, **diag}

    def predict_transition(self, x: torch.Tensor) -> torch.Tensor:
        dummy_target = torch.zeros(self.d_out, device=x.device)
        logits, diag = self.model.forward_step(
            z=x,
            target=dummy_target.argmax(),
            task_id=self.current_task_id,
            is_training=False
        )
        return torch.softmax(logits, dim=-1)

    def predict_no_state_update(self, x: torch.Tensor) -> torch.Tensor:
        h_prev_backup = self.model.h_prev.clone()
        context_state_backup = self.model.context_state.clone()
        smoothed_logits_backup = self.model.smoothed_logits.clone()

        prob = self.predict_transition(x)

        self.model.h_prev.copy_(h_prev_backup)
        self.model.context_state.copy_(context_state_backup)
        self.model.smoothed_logits.copy_(smoothed_logits_backup)
        return prob

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_columns": self.model.num_columns,
            "routing_mode": self.model.routing_mode,
            "decay_factor": self.model.decay_factor,
        }


