import torch
import torch.nn as nn
import math
from activation_functions import sigmoid
import matplotlib.pyplot as plt

#16 = 300
class Layer:
    def __init__(self, input_size, 
                 activation_function=None, learning_range=0.7,
                 max_weight=100, device='cpu',
                 mutation_scale=0, update_vectors_every=1, # mutation_scale seems to do nothing
                 optimizer_class=None, optimizer_kwargs=None,
                 # New optimizer/modulation knobs
                 neg_boost=0.15,
                 clip_grad_norm=None, clip_grad_value=None, # t
                 reward_baseline_beta=0.3,
                 enable_reward_baseline=True):
        self.input_size = input_size
        self.device = device
        self.next_layer = None
        self.weights = None
        self.spikes = torch.zeros(self.input_size, dtype=torch.float32, device=device)
        self.activation_function = activation_function
        self.post_compute_spikes = None
        self.learning_range = learning_range
        self.max_weight = max_weight
        self.threshold = -math.inf
        self.threshold_initialized = False
        self.iteration = 0
        self.accuracy_ema = None
        self.last_accuracy = None
        self.mutation_scale = mutation_scale
        
        # Track whether user supplied custom optimizer settings
        self.optimizer_is_default = (optimizer_class is None and optimizer_kwargs is None)

        # Store optimizer configuration for later initialization
        self.optimizer_class = optimizer_class or torch.optim.AdamW
        # Avoid unintended regularization from AdamW default weight_decay
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 0.001, 'weight_decay': 0.0}
        self.optimizer = None  # Will be initialized when weights are created 

        # Modulation/regularization configuration
        self.neg_boost = neg_boost
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.reward_baseline_beta = reward_baseline_beta
        self.reward_baseline = None
        self.enable_reward_baseline = enable_reward_baseline

    def init_weights(self, next_layer):
        self.next_layer = next_layer
        scale = torch.sqrt(torch.tensor(2.0 / (self.input_size + next_layer.input_size)))
        self.weights = torch.normal(0, scale, (self.input_size, next_layer.input_size), device=self.device) / 2
        self.weights.requires_grad_(True)  # Enable gradients for PyTorch optimizer
        
        # Initialize PyTorch optimizer
        self.optimizer = self.optimizer_class([self.weights], **self.optimizer_kwargs)

    def _apply_optimizer_update(self, scaled_gradient):
        """Apply PyTorch optimizer update with safety features (no fitness modulation)."""
        # Apply optional negative boost, independent of fitness
        modulated_gradient = scaled_gradient  * (1 + self.neg_boost * (scaled_gradient < 0).float())

        # PyTorch optimizers perform gradient descent: w -= lr * grad
        # Our reinforce logic computes a direction to INCREASE weights, so negate here
        modulated_gradient = -modulated_gradient
        
        # Set gradient and optional clipping, then step
        self.optimizer.zero_grad()
        self.weights.grad = modulated_gradient

        if self.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_([self.weights], self.clip_grad_value)
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_([self.weights], self.clip_grad_norm)

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
            
            # Initialize threshold only once (only during training)
            if train and not self.threshold_initialized and spike_outputs.numel() > 0:
                self.threshold = spike_outputs.min() * 0.8
                self.threshold_initialized = True
                
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
            # Output layer - no spike computation needed during inference!
            if not train:
                # Pure inference: just apply activation function if present
                return self.activation_function(x) if self.activation_function else x
            
            # Training: still need spike computation for learning
            if self.spikes.shape[0] != x.shape[0]:
                self.spikes = torch.zeros_like(x)
                
            spikes = (x > self.threshold).float()
            self.spikes = spikes
            spike_outputs = x * spikes
            
            if not self.threshold_initialized and spike_outputs.numel() > 0:
                self.threshold = spike_outputs.min() * 0.8
                self.threshold_initialized = True
                
            final_outputs = self.activation_function(spike_outputs) if self.activation_function else spike_outputs
            self.post_compute_spikes = final_outputs  # Remove redundant clone
            return final_outputs


    @torch.no_grad()
    def reinforce(self, signal, accuracy):
        batch_size = self.post_compute_spikes.shape[0]
        self.iteration += 1

        if self.next_layer:
            # Cache sparsity calculation for efficiency
            next_layer_sparsity = 1.0 - (self.next_layer.post_compute_spikes > self.next_layer.threshold).float().mean()

            # Compute next signal before updating weights to avoid expensive clone
            next_signal = signal @ self.weights.T

            elig = (self.post_compute_spikes.T @ self.next_layer.post_compute_spikes) / batch_size
            elig = elig * self.weights



            elig = torch.relu(elig)  # More efficient than clamp(0, math.inf)
            # Reward baseline (EMA) to reduce variance
            if self.enable_reward_baseline:
                batch_reward = signal.mean(dim=0, keepdim=True)
                if self.reward_baseline is None:
                    self.reward_baseline = torch.zeros_like(batch_reward)
                self.reward_baseline = self.reward_baseline_beta * self.reward_baseline + (1 - self.reward_baseline_beta) * batch_reward
                centered_signal = signal - self.reward_baseline
            else:
                centered_signal = signal

            gradient = ((self.post_compute_spikes.T @ centered_signal) / batch_size) * elig

            gradient_norm = torch.norm(gradient)
            if gradient_norm > self.learning_range:
                gradient *= (self.learning_range / gradient_norm)

            # Accuracy scaling with gamma
            accuracy_gamma = 0.3
            scale = (1 - accuracy) ** accuracy_gamma
            scaled_gradient = gradient * scale
            
            # Apply optimizer update using the new unified system
            self._apply_optimizer_update(scaled_gradient)
            
            # In-place clamp to preserve the optimizer's Parameter reference
            self.weights.clamp_(-self.max_weight, self.max_weight)
            return next_signal
        return signal




class Zor:
    def __init__(self, layers, optimizer_class=None, optimizer_kwargs=None):
        self.layers = layers
        self.optimizer_class = optimizer_class or torch.optim.AdamW
        self.optimizer_kwargs = optimizer_kwargs or {'lr': 0.001, 'weight_decay': 0.0}
        
        # Apply optimizer settings to all layers that don't already have custom optimizers
        for layer in layers:
            if getattr(layer, 'optimizer_is_default', True):
                # Layer is using defaults, apply Zor-level optimizer settings
                layer.optimizer_class = self.optimizer_class
                layer.optimizer_kwargs = self.optimizer_kwargs.copy()
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
    
    def experimental_reinforce(self, rewards, accuracy):
        for layer in reversed(self.layers):
            rewards = layer.experimental_reinforce(rewards, accuracy)
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
            
            # Restore original spike states (no learning occurred)
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
