from __future__ import annotations
import torch
import math
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional

class Layer:
    def __init__(self, input_size: int, 
                 activation_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 max_weight: float = 1,
                 device: str | torch.device = 'cpu',
                 optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW,
                 optimizer_kwargs: dict[str, Any] = {'lr': 0.001},
                 min_charge: float = -5.0,
                 stateless: bool = False) -> None:
        self.input_size: int = input_size
        self.device: str | torch.device = device
        self.next_layer: Optional[Layer] = None
        self.weights: Optional[torch.Tensor] = None
        # Double check that we shouldn't be doing float16 or something
        self.spikes: torch.Tensor = torch.zeros(self.input_size, dtype=torch.float32, device=device)
        self.charges: torch.Tensor = torch.zeros(self.input_size, dtype=torch.float32, device=device)

        self.activation_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = activation_function
        self.optimizer_kwargs: dict[str, Any] = optimizer_kwargs
        self.post_compute_spikes: Optional[torch.Tensor] = None
        self.max_weight: float = float(max_weight)
        self.threshold: float = -2
        self.min_charge: float = float(min_charge)
        self.stateless: bool = stateless
        self.threshold_initialized: bool = False
        self.iteration: int = 0
        self.optimizer_class: type[torch.optim.Optimizer] = optimizer
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def init_weights(self, next_layer: Layer) -> None:
        self.next_layer = next_layer
        scale = math.sqrt(2.0 / (self.input_size + next_layer.input_size))
        self.weights = (torch.randn((self.input_size, next_layer.input_size), device=self.device) * (scale / 2.0))
        self.weights.requires_grad_(True)  # Enable gradients for PyTorch optimizer
        self.optimizer = self.optimizer_class([self.weights], **self.optimizer_kwargs)

    def _apply_optimizer_update(self, scaled_gradient: torch.Tensor) -> None:
        """Apply PyTorch optimizer update with safety features (no fitness modulation)."""                
        assert self.optimizer is not None, "Optimizer not initialized. Call init_weights first."
        assert self.weights is not None, "Weights not initialized. Call init_weights first."
        self.optimizer.zero_grad()
        self.weights.grad = -scaled_gradient
        self.optimizer.step()
    
    def get_activation(self) -> float:
        return float((self.spikes > 0).float().mean().item())  # Use spikes, not post_compute_spikes
        
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
        x = x.float()  # Convert to float32 for numerical stability

        if self.next_layer:
            # Hidden layers need spike computation for gating
            # Always reallocate if batch size changed (handles both train and eval)
            if self.spikes.shape[0] != x.shape[0]:
                self.charges = torch.zeros_like(x)
                self.spikes = torch.zeros_like(x)
            
            # Zero out all charges if stateless mode enabled
            if self.stateless:
                self.charges.zero_()
            else:
                # Reset charges that fell below minimum threshold
                self.charges[self.charges < self.min_charge] = 0
            
            self.charges += x

            # Efficient spike computation (boolean mask), keep compute dense for GEMM efficiency
            spikes = (self.charges > self.threshold)
            if train:
                self.spikes = spikes.float()
            
            # Dense masked inputs (fast on accelerators) - compute BEFORE zeroing charges
            spike_outputs = self.charges * spikes.float()
            
            # Zero out charges where spikes occurred (for next iteration)
            self.charges[spikes] = 0
            
            assert self.weights is not None
            if self.activation_function:
                outputs = self.activation_function(spike_outputs)
                if train:
                    self.post_compute_spikes = outputs  # Remove redundant clone
                return outputs @ self.weights
            else:
                if train:
                    self.post_compute_spikes = spike_outputs  # Remove redundant clone
                return spike_outputs @ self.weights
        else:
            # Output layer
            if not train:
                return self.activation_function(x) if self.activation_function else x
            
            # Reallocate if batch size changed
            if self.spikes.shape[0] != x.shape[0]:
                self.charges = torch.zeros_like(x)
                self.spikes = torch.zeros_like(x)
            
            # Zero out all charges if stateless mode enabled
            if self.stateless:
                self.charges.zero_()
            else:
                # Reset charges that fell below minimum threshold
                self.charges[self.charges < self.min_charge] = 0
            
            self.charges += x
            spikes = (self.charges > self.threshold).float()
            self.spikes = spikes
            # Compute spike outputs BEFORE zeroing charges
            spike_outputs = self.charges * spikes
            # Zero out charges where spikes occurred (for next iteration)
            self.charges[spikes > 0] = 0
            final_outputs = self.activation_function(spike_outputs) if self.activation_function else spike_outputs
            self.post_compute_spikes = final_outputs  # Remove redundant clone
            return final_outputs


    @torch.no_grad()
    def reinforce(self, signal: torch.Tensor, accuracy: float) -> torch.Tensor:
        signal = signal / signal.norm(dim=0, keepdim=True)
        signal = torch.clamp(signal, -1, 1)
        self.iteration += 1

        if self.next_layer and self.weights is not None and self.post_compute_spikes is not None:
            next_signal = signal @ self.weights.T
            pre = self.post_compute_spikes
            post = self.next_layer.spikes
            pre = pre / (pre.norm(dim=0, keepdim=True) + 1e-8)
            post = post / (post.norm(dim=0, keepdim=True) + 1e-8)

            post = torch.abs(post)
            gradient = (pre.T @ (signal * post))
            self._apply_optimizer_update(gradient)
            self.weights.clamp_(-self.max_weight, self.max_weight)
            return next_signal
        return signal




class Zor:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers: list[Layer] = layers
        self.accuracy_history: list[float] = []
        self.activation_history: list[list[float]] = [[] for _ in range(len(layers))]
        self.last_batch: Optional[torch.Tensor] = None
        self.last_outputs: Optional[torch.Tensor] = None
        for i in range(len(layers) - 1):
            layers[i].init_weights(layers[i + 1])

    @torch.no_grad()
    def forward(self, input_data: torch.Tensor, train: bool = True) -> torch.Tensor:
        x = input_data
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x
    
    def reinforce(self, rewards: torch.Tensor, accuracy: float) -> torch.Tensor:
        for layer in reversed(self.layers):
            rewards = layer.reinforce(rewards, accuracy)
        return rewards
    
    
    @torch.no_grad()
    def train_batch(self, input_data: torch.Tensor, target_data: torch.Tensor) -> torch.Tensor:
        outputs = self.forward(input_data, train=True)
        errors = (target_data - outputs)
        errors = errors
        accuracy = 1.0 - float(torch.mean(torch.abs(errors)))

        accuracy = accuracy
        self.accuracy_history.append(accuracy)
        self.reinforce(errors, accuracy)
        return errors
    
    
    def evaluate(self, validation_data: torch.Tensor, train: bool = False) -> float:
        """Evaluate on validation data without affecting model parameters."""
        with torch.inference_mode():
            # Store current spike states to restore later
            original_spikes = [layer.spikes.clone() for layer in self.layers]
            
            outputs = self.forward(validation_data, train=False)
            errors = validation_data - outputs
            accuracy = 100.0 * (1.0 - float(torch.mean(torch.abs(errors))))
            for i, layer in enumerate(self.layers):
                layer.spikes = original_spikes[i]
            
            return accuracy
    
    def plot_accuracy(self) -> None:
        # Normalize history to floats for plotting to satisfy type checker
        history = [
            float(x.detach().cpu().item()) if isinstance(x, torch.Tensor) else float(x)
        for x in self.accuracy_history]
        plt.plot(history, '-')
        plt.show()
