# Zor ⚡️

A lightweight spiking neural network that learns without backpropagation through activation functions. Zor uses **analog-spike gating**: binary spike decisions control information flow for efficiency, while learning operates on continuous analog values for rich gradient signals.

## Key Results

- **Beats MLPs** on Fashion-MNIST reconstruction using only 10 training samples while being **30x faster**
- **Strong CIFAR-10 reconstruction** with efficient training
- **Learns colorization** on CIFAR-10 from limited data
- **~40% classification accuracy** on CIFAR-10 (work in progress)

## Quick Start

```python
import torch
from zor import Zor, Layer
from keras.datasets.fashion_mnist import load_data

# Define a simple autoencoder
snn = Zor([
    Layer(784),
    Layer(64),
    Layer(784)
])

# Load and prepare data
(X_train, y_train), (X_test, _) = load_data()
X_train = torch.tensor(X_train.reshape(-1, 784) / 255.0, dtype=torch.float32)

# Train
for epoch in range(500):
    batch = X_train[torch.randperm(len(X_train))[:48]]
    errors = snn.train_batch(batch, batch)
    
    if epoch % 50 == 0:
        accuracy = 1.0 - torch.mean(torch.abs(errors)).item()
        print(f"Epoch {epoch}: {accuracy:.1%} accuracy")
```

## How It Works

### Analog-Spike Gating

```python
spikes = (x > threshold)        # Binary decision (efficiency)
output = x * spikes.float()     # Analog values flow (learning signal)
```

Combines computational efficiency of sparse activation with rich learning signals from continuous values.

### Learning Rule

Zor uses a local correlation-based learning mechanism:

1. **Error signals** propagate backward through weight transposes (no derivatives through activations)
2. **Spike gating** provides structured activity patterns
3. **Local correlations** between layer activity and errors drive weight updates
4. **Adam optimizer** handles adaptive learning rates

This approach avoids computing gradients through activation functions while maintaining effective credit assignment.

### Architecture

- **Input/Hidden layers**: Spike when charge exceeds threshold, gate analog values
- **Output layer**: Returns continuous values directly (no spiking)
- **Flexible activation functions**: Optional per-layer (though often unnecessary)
- **GPU accelerated**: Built on PyTorch for efficient computation

## Philosophy

**Digital Plausibility over Biological Plausibility**

Biology and silicon are different substrates. Rather than simulate neurons, Zor adapts the core principles of efficient learning to digital hardware. The goal: spike-based efficiency without the complexity of traditional SNNs.

**Spiking for Efficiency**

Sparse, event-driven computation can be more efficient than dense matrix operations. Zor explores whether this efficiency can be achieved without sacrificing learning quality.

**Simplicity**

Most SNNs are complex because they simulate biological details. Zor keeps only what matters: sparse gating and local learning rules that work.

## What Makes Zor Different

- **Works competitively** with MLPs on reconstruction tasks
- **Learns to spike** rather than requiring it—early in training, most neurons spike on every input
- **Direct error signals** via weight transposes, not BPTT or pure Hebbian rules
- **Stable training** without the instabilities common in SNNs
- **Sample efficient** on reconstruction and colorization tasks

## Current Limitations

- **Classification performance** lags behind backprop-trained networks (~40% on CIFAR-10)
- **Shallow networks** tested so far—scaling to very deep architectures unproven
- **No convolutional layers** yet (critical for competitive image performance)
- **Early stage**: Missing basic features like model saving/loading

## Roadmap

- **Convolutional layers**: Essential for image task performance
- **Deeper architectures**: Test credit assignment through 10+ layers
- **Classification improvements**: Close the gap with backprop on discriminative tasks
- **Model persistence**: Save/load trained networks
- **Low-resource applications**: Efficient inference on edge devices, translation for low-resource languages
- **More examples**: Additional tasks and benchmarks

## Name Origin

Zor combines "Zeus" and "Thor"—and means "strength" in several languages.

## Examples

Check the `/examples` directory for:
- Fashion-MNIST autoencoding
- CIFAR-10 reconstruction
- CIFAR-10 colorization
- Classification examples

## Status

Zor is an active research project exploring alternatives to backpropagation. It shows promising results on reconstruction tasks and sample efficiency, with ongoing work to improve classification performance and architectural flexibility.

The learning mechanism is simpler than traditional SNNs while achieving competitive results—suggesting there may be efficient paths to learning that don't require full backpropagation through activation functions.