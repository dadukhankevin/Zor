import torch
import torch.nn as nn
import math
from activation_functions import sigmoid
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, input_size, 
                 activation_function=None,
                 max_weight=1, device='cpu',
                 optimizer=torch.optim.AdamW,
                 optimizer_kwargs={'lr': 0.001}):
        self.input_size = input_size
        self.device = device
        self.next_layer = None
        self.weights = None
        self.spikes = torch.zeros(self.input_size, dtype=torch.float32, device=device)
        self.activation_function = activation_function
        self.optimizer_kwargs = optimizer_kwargs
        self.post_compute_spikes = None
        self.max_weight = max_weight
        self.threshold = -1
        self.threshold_initialized = False
        self.iteration = 0
        self.optimizer = optimizer 

    def init_weights(self, next_layer):
        self.next_layer = next_layer
        scale = torch.sqrt(torch.tensor(2.0 / (self.input_size + next_layer.input_size)))
        self.weights = torch.normal(0, scale, (self.input_size, next_layer.input_size), device=self.device) / 2
        self.weights.requires_grad_(True)  # Enable gradients for PyTorch optimizer
        self.optimizer = self.optimizer([self.weights], **self.optimizer_kwargs)

    def _apply_optimizer_update(self, scaled_gradient):
        """Apply PyTorch optimizer update with safety features (no fitness modulation)."""                
        self.optimizer.zero_grad()
        self.weights.grad = -scaled_gradient
        self.optimizer.step()
    
    def get_activation(self):
        return (self.spikes > 0).float().mean()  # Use spikes, not post_compute_spikes
        
    
    @torch.no_grad()
    def forward(self, x, train=True):
        x = x.float()  # Convert to float32 for numerical stability
        
        if self.next_layer:
            # Hidden layers need spike computation for gating
            if train:
                # Only reallocate if batch size changed (more efficient check)
                if self.spikes.shape[0] != x.shape[0]:
                    self.spikes = torch.zeros_like(x)

            # Efficient spike computation (boolean mask), keep compute dense for GEMM efficiency
            spikes = (x > self.threshold)
            if train:
                self.spikes = spikes.float()

            # Dense masked inputs (fast on accelerators)
            spike_outputs = x * spikes.float()
            
        
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
            if not train:
                return self.activation_function(x) if self.activation_function else x            
            if self.spikes.shape[0] != x.shape[0]:
                self.spikes = torch.zeros_like(x)
            spikes = (x > self.threshold).float()
            self.spikes = spikes
            spike_outputs = x * spikes
            final_outputs = self.activation_function(spike_outputs) if self.activation_function else spike_outputs
            self.post_compute_spikes = final_outputs  # Remove redundant clone
            return final_outputs


    @torch.no_grad()
    def reinforce(self, signal, accuracy):
        signal /= signal.norm(dim=0, keepdim=True)
        batch_size = self.post_compute_spikes.shape[0]
        self.iteration += 1

        if self.next_layer:
            next_signal = signal @ self.weights.T
            elig = (self.spikes.T @ self.next_layer.post_compute_spikes)

            # elig /= elig.norm(dim=0, keepdim=True)
            elig = torch.clamp(elig, 0, 1)
            gradient = (self.post_compute_spikes.T @ signal) * elig
            self._apply_optimizer_update(gradient)
            self.weights.clamp_(-self.max_weight, self.max_weight)
            return next_signal
        return signal




class Zor:
    def __init__(self, layers):
        self.layers = layers
        self.accuracy_history = []
        self.activation_history = [[] for _ in range(len(layers))]
        self.last_batch = None
        self.last_outputs = None
        for i in range(len(layers) - 1):
            layers[i].init_weights(layers[i + 1])

    @torch.no_grad()
    def forward(self, input_data, train=True):
        x = input_data
        for layer in self.layers:
            x = layer.forward(x, train=train)
        return x
    
    def reinforce(self, rewards, accuracy):
        for layer in reversed(self.layers):
            rewards = layer.reinforce(rewards, accuracy)
        return rewards
    
    
    @torch.no_grad()
    def train_batch(self, input_data, target_data):
        outputs = self.forward(input_data, train=True)
        errors = target_data - outputs
        accuracy = 1.0 - float(torch.mean(torch.abs(errors)))
        accuracy = accuracy
        self.accuracy_history.append(accuracy)
        self.reinforce(errors, accuracy)
        return errors
    
    
    def evaluate(self, validation_data, train=False):
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
    
    def plot_accuracy(self):
        # Convert to CPU for matplotlib
        if isinstance(self.accuracy_history[0], torch.Tensor):
            history = [x.cpu() if hasattr(x, 'cpu') else x for x in self.accuracy_history]
        else:
            history = self.accuracy_history
        plt.plot(history)
        plt.show()
