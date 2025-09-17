import torch
import torch.nn as nn


class Layer:
    def __init__(self, input_size, target_activation, threshold=0, threshold_inhibition=0, 
                 activation_function=None, learning_range=0.001, activation_rate=0.1, 
                 universal_rolling_factor=0.2, novelty_factor=0.01, max_weight=.2, device='cpu'):
        self.input_size = input_size
        self.device = device
        self.next_layer = None
        self.weights = None
        self.spikes = torch.zeros(self.input_size, dtype=torch.float32, device=device)
        self.target_activation = target_activation
        self.activation_function = activation_function
        self.activation_rolling_average = 0
        self.thresholds = torch.full((self.input_size,), threshold, dtype=torch.float32, device=device)
        self.unit_rate_ema = torch.zeros(self.input_size, dtype=torch.float32, device=device)
        self.threshold_inhibition = threshold_inhibition
        self.post_compute_spikes = None
        self.learning_range = learning_range
        self.activation_rate = activation_rate
        self.universal_rolling_factor = universal_rolling_factor
        self.rolling_coactivation = None
        self.max_weight = max_weight
        self.novelty_factor = novelty_factor
        self.zero = torch.tensor(0.0, device=self.device)

    def init_weights(self, next_layer):
        self.next_layer = next_layer
        scale = torch.sqrt(torch.tensor(2.0 / (self.input_size + next_layer.input_size)))
        self.weights = torch.normal(0, scale, (self.input_size, next_layer.input_size), device=self.device) / 2
        self.rolling_coactivation = torch.zeros((self.input_size, next_layer.input_size), device=self.device)

    def forward(self, x, train=True):
        if self.spikes.shape[0] != x.shape[0]:
            self.spikes = torch.zeros_like(x)

        inhibition = self.spikes * self.threshold_inhibition
        spikes = (x > (self.thresholds - inhibition)).to(torch.float32)
        activation_percentage = float(spikes.mean())

        if self.activation_rolling_average == 0:
            self.activation_rolling_average = activation_percentage

        if train:
            self.spikes = spikes
            self.activation_rolling_average = (
                self.activation_rolling_average * self.universal_rolling_factor +
                activation_percentage * (1 - self.universal_rolling_factor)
            )
            batch_rate = self.spikes.mean(dim=0).to(torch.float32)
            self.unit_rate_ema = (
                self.unit_rate_ema * self.universal_rolling_factor +
                batch_rate * (1 - self.universal_rolling_factor)
            )

        spike_mask = spikes > 0
        spike_outputs = x * spikes

        if self.next_layer:
            outputs = spike_outputs
            if self.activation_function and spike_mask.any():
                outputs = torch.zeros_like(spike_outputs)
                outputs[spike_mask] = self.activation_function(spike_outputs[spike_mask])
            
            if train:
                self.post_compute_spikes = outputs.clone()
            
            if not spike_mask.any():
                return torch.zeros(x.shape[0], self.weights.shape[1], device=self.device)
            
            sparsity = 1.0 - float(spike_mask.sum()) / spike_mask.numel()
            if sparsity > 0.8 and not train:
                return torch.sparse.mm(outputs.to_sparse(), self.weights.T).T
            return outputs @ self.weights
        else:
            final_outputs = self.activation_function(x) if self.activation_function else x
            if train:
                post = spike_outputs
                if self.activation_function and spike_mask.any():
                    post = torch.zeros_like(spike_outputs)
                    post[spike_mask] = self.activation_function(spike_outputs[spike_mask])
                self.post_compute_spikes = post.clone()
            return final_outputs


    def reinforce(self, vector_reward):
        batch_size = self.post_compute_spikes.shape[0]

        per_unit_error = (self.target_activation - self.unit_rate_ema).to(torch.float32)
        self.thresholds -= self.activation_rate * per_unit_error
        self.thresholds = torch.clamp(self.thresholds, 0, 2)

        if self.next_layer:
            old_weights = self.weights.clone()

            elig = (self.post_compute_spikes.T @ self.next_layer.post_compute_spikes) / batch_size
            self.rolling_coactivation = self.rolling_coactivation * self.universal_rolling_factor + torch.abs(elig) * (1 - self.universal_rolling_factor)

            elig = elig - (self.rolling_coactivation * self.novelty_factor)

            elig = torch.clamp(elig, 0, 1)
            gradient = ((self.post_compute_spikes.T @ vector_reward) / batch_size) * elig

            gradient_norm = torch.norm(gradient)
            if gradient_norm > 0 and gradient_norm > self.learning_range:
                gradient = gradient * (self.learning_range / gradient_norm)
            self.weights += gradient

            self.weights = torch.clamp(self.weights, -self.max_weight, self.max_weight)

            return vector_reward @ old_weights.T
        return vector_reward


class Zor:
    def __init__(self, layers):
        self.layers = layers
        self.accuracy_history = []
        self.validation_accuracy_history = []
        self.activation_history = [[] for _ in range(len(layers))]
        self.last_batch = None
        self.last_outputs = None
        for i in range(len(layers) - 1):
            layers[i].init_weights(layers[i + 1])

    def forward(self, input_data, train=True):
        x = input_data
        for layer in self.layers:
            x = layer.forward(x, train=train)
            if not train and torch.all(x == 0):
                break
        return x
    
    def reinforce(self, rewards):
        for layer in reversed(self.layers):
            rewards = layer.reinforce(rewards)
        return rewards
    
    def train_batch(self, input_data, target_data):
        outputs = self.forward(input_data, train=True)
        errors = target_data - outputs
        self.reinforce(errors)
        return errors
    
    def evaluate(self, validation_data, train=False):
        original_spikes = [layer.spikes.clone() for layer in self.layers]
        
        outputs = self.forward(validation_data, train=False)
        errors = validation_data - outputs
        accuracy = 100.0 * (1.0 - float(torch.mean(torch.abs(errors))))
        
        for i, layer in enumerate(self.layers):
            layer.spikes = original_spikes[i]
        
        return accuracy
