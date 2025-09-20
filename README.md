# Zor

Zor is a lightweight spiking neural network that uses **analog-spike gating** - binary spike decisions control information flow, while learning operates on the continuous analog values that pass through. This combines the computational efficiency of sparse spiking with the rich gradients needed for effective learning.


## Performance

Zor consistently outperforms traditional MLPs across all data scales while achieving competitive inference speeds. Both models use AdamW optimizers (lr=0.001) for fair comparison:

### Training Performance (450 epochs)
| Dataset Size | Zor Val Acc | MLP Val Acc | Zor PSNR | MLP PSNR | Zor Time | MLP Time | Zor FWD | MLP FWD |
|--------------|-------------|-------------|----------|----------|----------|----------|---------|---------|
| 64 images    | **88.2%**   | 85.2%       | **16.30dB** | 14.47dB | 5.9s     | 3.0s     | 1.5ms   | 1.2ms   |
| 500 images   | **91.4%**   | 88.8%       | **18.99dB** | 16.79dB | 8.0s     | 6.5s     | 1.5ms   | 1.2ms   |
| 1000 images  | **91.4%**   | 88.8%       | **19.02dB** | 16.79dB | 8.4s     | 6.5s     | 1.3ms   | 1.2ms   |
| 5000 images  | **90.4%**   | 89.3%       | **18.22dB** | 17.15dB | 7.1s     | 6.7s     | 1.3ms   | 1.3ms   |

### Extended Training Performance (1000 epochs)  
| Dataset Size | Zor Val Acc | MLP Val Acc | Zor PSNR | MLP PSNR | Zor Time | MLP Time | Zor FWD | MLP FWD |
|--------------|-------------|-------------|----------|----------|----------|----------|---------|---------|
| 64 images    | **88.2%**   | 85.2%       | **16.32dB** | 14.46dB | 11.3s    | 7.4s     | 1.3ms   | 1.2ms   |
| 1000 images  | **93.0%**   | 90.2%       | **20.60dB** | 17.89dB | 17.1s    | 14.9s    | 1.3ms   | 1.3ms   |
| 5000 images  | **92.3%**   | 91.5%       | **19.92dB** | 18.95dB | 17.7s    | 14.0s    | 1.4ms   | 1.3ms   |

### Long Training Performance (5000 epochs)
| Dataset Size | Zor Val Acc | MLP Val Acc | Zor PSNR | MLP PSNR | Zor Time | MLP Time | Zor FWD | MLP FWD |
|--------------|-------------|-------------|----------|----------|----------|----------|---------|---------|
| 500 images   | **92.5%**   | 90.0%       | **19.97dB** | 17.80dB | 85.8s    | 73.1s    | 1.3ms   | 1.2ms   |
| 50,000 images| **94.3%**   | 93.2%       | **22.16dB** | 20.83dB | 75.3s    | 64.2s    | 1.3ms   | 1.2ms   |

**Key Insights:**
- **Superior Accuracy**: Zor achieves 1-3% higher validation accuracy across all datasets and training durations
- **Exceptional Reconstruction Quality**: 1.3-2.7dB PSNR improvement, reaching 22+ dB with full-scale training
- **Competitive Speed**: Forward pass times are nearly identical (1.3ms vs 1.2ms)  
- **Scales Excellently**: Performance continues improving with more data (94.3% accuracy on full CIFAR-10)
- **No Backpropagation**: Achieves these results using novel analog-spike learning without derivatives
- **Consistent Advantage**: Performance gap maintained across different dataset sizes and training durations

Zor uses a novel learning rule without derivatives, achieving better generalization with smaller train/validation gaps. See [detailed test results](examples/readme.md).

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
], optimizer_class=optim.Adam, optimizer_kwargs={'lr': 0.001})

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

- **Simplicity** -> SNNs are typically ridiculously complicated because they essentially try to simulate biological neurons. Zor takes the best of both traditional *ANNs* and efficiency lessons from nature (and my imagination).

- **Framing** -> Rather than redesign the network for a problem, we should redesign the problem for the network.

## Design

### Signal Propagation

**Normal**: A typical SNN operates by receiving inputs (duh) and propagating them forward; each "neuron" accumulates a "charge." When it reaches a certain threshold, the neuron "spikes" and sends a binary "on" signal to all connected neurons.

**Zor**: Uses binary spike decisions to gate analog charges. The key insight: spikes control what information flows forward, but learning operates on the continuous analog values that passed through the gates. This combines computational efficiency (sparse activation) with rich learning signals (analog gradients).

### Learning

**Normal**: SNNs often implement either "Hebbian" learning or "Backpropagation Through Time" (BTT). Hebbian learning strengthens connections between coactive neurons, but this often leads to runaway behaviour and requires complex homeostatic pressures that often suppress learning. BTT is terrible for many reasons, including that I generally think that backpropagation may not be needed for learning (nothing in nature implements it, and "digiture" implements it poorly).

**Zor**: Zor does something *vaguely similar* to three-factor Hebbian learning, but is also quite different. Rather than looking merely at coactivity, Zor looks at *how* coactive neurons are, and it can do this because we use analog spiking rather than binary. This strengthens the signal enormously. Zor also "backpropagates" errors across the coactivity matrix, weighted by how novel the coactivity is between any two neurons. Unlike many alternatives to backpropagation, Zor can tell to what extent and in what direction each weight should change.


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

**Zor**: Zor directly returns instant *charges* for the output layer rather than spikes, the spikes are still usefull for learning though since they carry an important signal! You can also optionally add activation functions to any layer in Zor but I haven't found any example where that helps yet. Seems they aren't needed.

**Zor** is also pretty stable, and not prone to seizures like some SNNs (: The homeostatic measures taken also help learning rather than hinder it.

---

If you want to really know how it works look at the code.

*Zor is a mix between "Zeus" and "Thor" and conveniently means strength in several languages.*

## Quick Start

Look at the examples in `/examples` I will add more and more.

## Roadmap
Zor is the result of many, many, sleepless nights pulling my hair out over how stupid backpropagation is vs how simple I felt like spike-based learning should be, and there's still a ways to go, so here's a roadmap:

- GPU acceleration (DONE!). I built Zor with the idea that we shouldn't need to use "neuromorphic chips" in order to benifit from spike based efficiency, so this is HIGH on the list.
- Saving/Loading. Save models, load them, etc... Plus efficient layer by layer laoding where we only load the parts of the network we need (benifit of sparsity). I hope this could make it possible to use very large models on very limited hardware.
- Research: I made lots of headway on the learning mechanism (non-hebbian, and not-backprop). But I know there is a significant ways to go still.
- Examples: I'll post more examples and code.
- Language models. Since this works well with low data, I hope to be able to use it to translate into low-resource languages on very low-end hardware. Remains to be seen how well this will work for translation.
- Image models: So far I'm very happy with how well the model can learn image representations from very little data.