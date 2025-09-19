import torch
import torch.nn as nn
import math

class Layer:
    def __init__(self, input_size, 
                 activation_function=None, learning_range=0.1, 
                 max_weight=1, device='cpu',
                 momentum_factor=0.9, min_accuracy_scale=0.01, do_solidity=False):
        self.input_size = input_size
        self.device = device
        self.next_layer = None
        self.weights = None
        self.spikes = torch.zeros(self.input_size, dtype=torch.float32, device=device)
        self.activation_function = activation_function
        self.post_compute_spikes = None
        self.learning_range = learning_range
        self.max_weight = max_weight
        self.zero = torch.tensor(0.0, device=self.device)
        self.momentum_factor = momentum_factor
        self.min_accuracy_scale = min_accuracy_scale
        self.do_solidity = do_solidity
        self.threshold = -math.inf

    def init_weights(self, next_layer):
        self.next_layer = next_layer
        scale = torch.sqrt(torch.tensor(2.0 / (self.input_size + next_layer.input_size)))
        self.weights = torch.normal(0, scale, (self.input_size, next_layer.input_size), device=self.device) / 2
        self.momentum = torch.zeros_like(self.weights)
        self.solidity = torch.zeros_like(self.weights)
    
    def get_activation(self):
        return (self.spikes > 0).float().mean()  # Use spikes, not post_compute_spikes
        
    
    @torch.no_grad()
    def forward(self, x, train=True):
        if self.spikes.shape[0] != x.shape[0]:
            self.spikes = torch.zeros_like(x)

        spikes = (x > self.threshold).to(torch.float32)
        
        if train:
            self.spikes = spikes

        spike_outputs = torch.where(spikes > 0 , x, self.zero)
        if self.threshold == -math.inf:
            self.threshold = spike_outputs.min() * .8
        if self.next_layer:
            outputs = self.activation_function(spike_outputs) if self.activation_function else spike_outputs
            if train:
                self.post_compute_spikes = outputs.clone()
            return outputs @ self.weights
        else:
            final_outputs = self.activation_function(spike_outputs) if self.activation_function else spike_outputs
            if train:
                self.post_compute_spikes = final_outputs.clone()
            return final_outputs


    @torch.no_grad()
    def reinforce(self, vector_reward, accuracy):
        
        batch_size = self.post_compute_spikes.shape[0]

        if self.next_layer:
            next_layer_sparsity = 1.0 - (self.next_layer.post_compute_spikes > self.next_layer.threshold).float().mean()

            old_weights = self.weights.clone()

            elig = (self.post_compute_spikes.T @ self.next_layer.post_compute_spikes) / batch_size

            if self.do_solidity:
                batch_reward = vector_reward.mean(dim=0, keepdim=True) 
                update =  elig * next_layer_sparsity * batch_reward
                update = update - update.mean()

                self.solidity += update #* accuracy
                norm = torch.norm(torch.abs(self.solidity), dim=0, keepdim=True) + 1e-8
                self.solidity = self.solidity / norm




            elig = torch.clamp(elig, 0, math.inf)
            gradient = ((self.post_compute_spikes.T @ vector_reward) / batch_size) * elig

            gradient_norm = torch.norm(gradient)
            if gradient_norm > 0 and gradient_norm > self.learning_range:
                gradient = gradient * (self.learning_range / gradient_norm)

            # scale = max(1 - accuracy, self.min_accuracy_scale)
            scale = max(1.0 / (1 + 2 * accuracy), self.min_accuracy_scale)
            self.momentum = self.momentum_factor * self.momentum + gradient * scale

            solidity = torch.clamp(self.solidity, 0, math.inf)
            self.weights += self.momentum * (1-solidity) * (1 + 0.01 * (self.momentum < 0).float())
            self.weights = torch.clamp(self.weights, -self.max_weight, self.max_weight)
            
            return vector_reward @ old_weights.T
        return vector_reward




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
    
    def experimental_reinforce(self, rewards, accuracy):
        for layer in reversed(self.layers):
            rewards = layer.experimental_reinforce(rewards, accuracy)
        return rewards
    
    @torch.no_grad()
    def train_batch(self, input_data, target_data):
        outputs = self.forward(input_data, train=True)
        errors = target_data - outputs
        accuracy = 1.0 - float(torch.mean(torch.abs(errors)))
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
