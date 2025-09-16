import numpy as np


class Layer:
    def __init__(self, input_size, target_activation, threshold=0, threshold_inhibition=0, 
                 activation_function=None, learning_range=0.001, sparsity_rate=0.1, 
                 universal_rolling_factor=0.2, batch_size=20, novelty_factor=0.01):
        self.input_size = input_size
        self.next_layer = None
        self.weights = None
        self.spikes = np.zeros(self.input_size, dtype=np.float32)
        self.target_activation = target_activation
        self.activation_function = activation_function
        self.sparsity_rolling_average = 0
        self.thresholds = np.full(self.input_size, threshold, dtype=np.float32)
        self.unit_rate_ema = np.zeros(self.input_size, dtype=np.float32)
        self.threshold_inhibition = threshold_inhibition
        self.post_compute_spikes = None
        self.learning_range = learning_range
        self.sparsity_rate = sparsity_rate
        self.universal_rolling_factor = universal_rolling_factor
        self.batch_size = batch_size
        self.rolling_coactivation = None
        self.novelty_factor = novelty_factor

    def init_weights(self, next_layer):
        self.next_layer = next_layer
        scale = np.sqrt(2.0 / (self.input_size + next_layer.input_size))
        self.weights = np.random.normal(0, scale, (self.input_size, next_layer.input_size)) / 2
        self.rolling_coactivation = np.zeros((self.input_size, next_layer.input_size))

    def forward(self, x, train=True):
        if self.spikes.shape[0] != x.shape[0]:
            self.spikes = np.zeros_like(x)
            
        inhibition = self.spikes * self.threshold_inhibition
        self.spikes = (x > self.thresholds - inhibition).astype(np.float32)
        sparsity = float(self.spikes.mean())

        if self.sparsity_rolling_average == 0:
            self.sparsity_rolling_average = sparsity
        if train:
            self.sparsity_rolling_average = (self.sparsity_rolling_average * self.universal_rolling_factor +
                                        sparsity * (1 - self.universal_rolling_factor))
            batch_rate = self.spikes.mean(axis=0).astype(np.float32)

            self.unit_rate_ema = (self.unit_rate_ema * self.universal_rolling_factor +
                              batch_rate * (1 - self.universal_rolling_factor))

        outputs = np.where(self.spikes, x, 0)
        if self.activation_function:
            outputs = self.activation_function(outputs)
        self.post_compute_spikes = outputs.copy()
        return outputs @ self.weights if self.next_layer else x

    def reinforce(self, vector_reward):
        per_unit_error = (self.target_activation - self.unit_rate_ema).astype(np.float32)
        self.thresholds -= np.float32(self.sparsity_rate) * per_unit_error
        self.thresholds = np.clip(self.thresholds, 0, 2)

        if self.next_layer:
            old_weights = self.weights.copy()

            elig = (self.post_compute_spikes.T @ self.next_layer.post_compute_spikes) / self.batch_size
            elig = elig - (self.rolling_coactivation * self.novelty_factor) # 15.04

            
            gradient = ((self.post_compute_spikes.T @ vector_reward) / self.batch_size) * elig
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 0 and gradient_norm > self.learning_range:
                gradient = gradient * (self.learning_range / gradient_norm)
            self.rolling_coactivation = self.rolling_coactivation * self.universal_rolling_factor + np.abs(elig) * (1 - self.universal_rolling_factor)
            self.weights += gradient

            max_val = .1
            self.weights = np.clip(self.weights, -max_val, max_val)

            return vector_reward @ old_weights.T
        return vector_reward


class BrainSpace:
    def __init__(self, layers):
        self.layers = layers
        self.accuracy_history = []
        self.validation_accuracy_history = []
        self.sparsity_history = [[] for _ in range(len(layers))]
        self.last_batch = None
        self.last_outputs = None
        for i in range(len(layers) - 1):
            layers[i].init_weights(layers[i + 1])

    def forward(self, input_data):
        x = input_data
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def reinforce(self, rewards):
        for layer in reversed(self.layers):
            rewards = layer.reinforce(rewards)
        return rewards
    
    def train_batch(self, input_data, target_data):
        outputs = self.forward(input_data)
        errors = target_data - outputs
        self.reinforce(errors)
        return errors
    
    def evaluate(self, validation_data):
        """Evaluate on validation data without affecting model parameters."""
        # Store current spike states to restore later
        original_spikes = [layer.spikes.copy() for layer in self.layers]
        
        outputs = self.forward(validation_data)
        errors = validation_data - outputs
        accuracy = 100.0 * (1.0 - float(np.mean(np.abs(errors))))
        
        # Restore original spike states (no learning occurred)
        for i, layer in enumerate(self.layers):
            layer.spikes = original_spikes[i]
        
        return accuracy
