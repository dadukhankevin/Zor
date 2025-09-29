# Zor ⚡️

A lightweight spiking neural network that learns without backpropagation through activation functions. Zor uses **analog-spike gating**: binary spike decisions control information flow for efficiency, while learning operates on continuous analog values for rich gradient signals.

<<<<<<< HEAD
## Key Results

- **Beats MLPs** on Fashion-MNIST reconstruction using only 10 training samples while being **30x faster**
- **Strong CIFAR-10 reconstruction** with efficient training
- **Learns colorization** on CIFAR-10 from limited data
- **~40% classification accuracy** on CIFAR-10 (work in progress)
=======
## Performance

Zor seems to outperform traditional MLPs across all data scales while achieving dramatically faster training speeds (twice as fast in controlled tests).

Zor uses a novel learning rule without derivatives, achieving better generalization with smaller train/validation gaps.

## Quick Example

```python
import torch
import torch.optim as optim
import time
from activation_functions import *
from zor import Zor, Layer
from keras.datasets.fashion_mnist import load_data

SAMPLES_PER_CLASS = 2
snn = Zor([
    Layer(784),
    Layer(64),
    Layer(784)
])

(X_train, y_train), (X_test, _) = load_data()

train_indices = []
for class_id in range(10):
    class_indices = torch.where(torch.tensor(y_train) == class_id)[0][:SAMPLES_PER_CLASS]
    train_indices.extend(class_indices.tolist())

X_train_subset = torch.tensor(X_train[train_indices].reshape(-1, 784) / 255.0, dtype=torch.float32)
X_val = torch.tensor(X_train[2000:3000].reshape(-1, 784) / 255.0, dtype=torch.float32)

start_time = time.time()
for epoch in range(500):
    indices = torch.randperm(len(X_train_subset))[:48]
    batch = X_train_subset[indices]
    errors = snn.train_batch(batch, batch)
    accuracy = 1.0 - torch.mean(torch.abs(errors)).item()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: {accuracy:.1f}% accuracy")

training_time = time.time() - start_time
print(f"Training complete! Training time: {training_time:.1f} seconds")

val_outputs = snn.forward(X_val, train=False)
val_errors = X_val - val_outputs
val_accuracy = 100.0 * (1.0 - torch.mean(torch.abs(val_errors)))
print(f"Validation accuracy: {val_accuracy:.1f}%")
```

**Results**: Beats MLPs using only 10 training samples (1 per class) while being 30x faster. This is all without backpropagation, using Adam-optimized analog-spike learning.

## Philosophy

Zor operates on this philosophy:

- **Digital Plausibility** -> Existing SNNs fail because they try to implement "biological plausibility," when we should aim for digital plausibility. Silicon and biology are different substrates, so the abstract principle `learning` will need to be implemented uniquely on each substrate.

- **Spiking** -> The "spiking" aspect of neural networks is vital to efficient learning, and backpropagation is likely orders of magnitude less efficient than an optimal learning rule.

- **Simplicity** -> SNNs are typically ridiculously complicated because they essentially try to simulate biological neurons. Zor takes the best of both traditional _ANNs_ and efficiency lessons from nature (and my imagination).

- **Framing** -> Rather than redesign the network for a problem, we should redesign the problem for the network.

## Design

### Signal Propagation

**Normal**: A typical SNN operates by receiving inputs (duh) and propagating them forward; each "neuron" accumulates a "charge." When it reaches a certain threshold, the neuron "spikes" and sends a binary "on" signal to all connected neurons.

**Zor**: Uses binary spike decisions to gate analog charges. The key insight: spikes control what information flows forward, but learning operates on the continuous analog values that passed through the gates. This combines computational efficiency (sparse activation) with rich learning signals (analog gradients).

### Learning

**Normal**: SNNs often implement either "Hebbian" learning or "Backpropagation Through Time" (BTT). Hebbian learning strengthens connections between coactive neurons, but this often leads to runaway behaviour and requires complex homeostatic pressures that often suppress learning. BTT is terrible for many reasons, including that I generally think that backpropagation may not be needed for learning (nothing in nature implements it, and "digiture" implements it poorly).

**Zor**: Zor does something _vaguely similar_ to three-factor Hebbian learning, but is also quite different. Rather than looking merely at coactivity, Zor looks at _how_ coactive neurons are, and it can do this because we use analog spiking rather than binary. This strengthens the signal enormously. Zor also "backpropagates" errors across the coactivity matrix, weighted by how novel the coactivity is between any two neurons. Unlike many alternatives to backpropagation, Zor can tell to what extent and in what direction each weight should change.

- **Analog-spike gating**: Binary spikes gate continuous values - efficiency of sparsity, richness of analog learning
- **Subtractive novelty gating**: Down-weights familiar patterns rather than boosting novel ones
- **Threshold homeostasis**: Simple target activation rates maintain stable sparse activity
- **Works at any activation level**: Learning quality doesn't depend on sparsity - can run dense or sparse

## Relation to prior work

Zor takes some inspiration from several lines of work (three‑factor rules, local eligibility traces, layer‑local objectives, and feedback‑alignment‑style signals) but combines them differently, and adds new evolutionary concepts like fitness to weights as well as many other things.

How Zor is different:

- It works, and is competitive not only with Spiking Neural Nets but also with MLP.
- Unlike many spiking nets, Zor does not need spiking to learn, rather it learns spiking as a way to increase efficiency. At the start of training all neurons spike every time.
- Vector error per layer via the current forward weights’ transpose (no BPTT, no random feedback)
- Analog‑spike gating: a binary spike gates the analog charge that actually learns
- Subtractive novelty: an EMA co‑activation penalty that down‑weights familiar co‑activity
- Per‑unit threshold homeostasis paired with a simple reconstruction‑based curriculum
- Lots of other things in the code lol.

In short, it overlaps in spirit with prior ideas but the specific mechanics and their integration appear uncommon.

### Outputs

**Normal**: Outputs are tricky for SNNs, sometimes people look at their firing frequencies, since the spikes themselves carry no useful information in the output layer. This requires more computation and bluntly, is bad design. Normal neural networks have conventient outputs that can take any value they need to (especially through activation functions).

**Zor**: Zor directly returns instant _charges_ for the output layer rather than spikes, the spikes are still usefull for learning though since they carry an important signal! You can also optionally add activation functions to any layer in Zor but I haven't found any example where that helps yet. Seems they aren't needed.

**Zor** is also pretty stable, and not prone to seizures like some SNNs (: The homeostatic measures taken also help learning rather than hinder it.

---

If you want to really know how it works look at the code.

_Zor is a mix between "Zeus" and "Thor" and conveniently means strength in several languages._
>>>>>>> a1e0525a13e89288a7d40d6b7a523f43ed6184cf

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
<<<<<<< HEAD

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
=======

Zor is the result of many, many, sleepless nights pulling my hair out over how stupid backpropagation is vs how simple I felt like spike-based learning should be, and there's still a ways to go, so here's a roadmap:

- GPU acceleration (DONE!). I built Zor with the idea that we shouldn't need to use "neuromorphic chips" in order to benifit from spike based efficiency, so this is HIGH on the list.
- Saving/Loading. Save models, load them, etc... Plus efficient layer by layer laoding where we only load the parts of the network we need (benifit of sparsity). I hope this could make it possible to use very large models on very limited hardware.
- Research: I made lots of headway on the learning mechanism (non-hebbian, and not-backprop). But I know there is a significant ways to go still.
- Examples: I'll post more examples and code.
- Language models. Since this works well with low data, I hope to be able to use it to translate into low-resource languages on very low-end hardware. Remains to be seen how well this will work for translation.
- Image models: So far I'm very happy with how well the model can learn image representations from very little data.
>>>>>>> a1e0525a13e89288a7d40d6b7a523f43ed6184cf
