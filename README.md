# Zor

Zor is a lightweight spiking neural network that uses **analog-spike gating** - binary spike decisions control information flow, while learning operates on the continuous analog values that pass through. This combines the computational efficiency of sparse spiking with the rich gradients needed for effective learning.

I think this leads to competitive performance to backpropagation in low-data settings, often with dramatically faster training times.

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

Zor takes some inspiration from several lines of work (three‑factor rules, local eligibility traces, layer‑local objectives, and feedback‑alignment‑style signals) but combines them differently, simply.

What’s different here:

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

### Fashion-MNIST Autoencoder Example

Run the Fashion-MNIST autoencoder example to see Zor in action:

```bash
python fashon.py
```

This example demonstrates:
1. **Extreme data efficiency** - 83% validation accuracy using only 10 Fashion-MNIST samples (1 per class)
2. **Fast training** - Complete training in 0.2 seconds on CPU
3. **One-shot learning** - Generalizes from seeing just one example of each clothing type
4. **Curriculum learning** - Automatically focuses on the hardest samples during training

*Note: I measure reconstruction accuracy as 100% × (1 - mean absolute error), which gives higher numbers than typical metrics but consistently tracks learning progress.*

A working version in just 15 lines, should get similar results.

```python
from zor import Zor, Layer
from activation_functions import sigmoid

# Create network
snn = Zor([
    Layer(784, target_activation=0.9, learning_range=0.5, activation_rate=0.2),
    Layer(128, target_activation=0.5, learning_range=0.5, activation_rate=0.2), 
    Layer(784, target_activation=0.5, activation_function=sigmoid, learning_range=0.5, activation_rate=0.2)
])

# Train
for epoch in range(200):
    batch = X[torch.randint(0, len(X), (500,))]
    outputs = snn.forward(batch)
    errors = batch - outputs
    snn.reinforce(errors)
```

In the example file I've gotten up to 83% validation accuracy on 1000 unseen samples after training on just 10 examples in 0.2 seconds. Again, this is all without any normal backpropagation, and is much better than any SNN I've ever seen.

## Roadmap
Zor is the result of many, many, sleepless nights pulling my hair out over how stupid backpropagation is vs how simple I felt like spike-based learning should be, and there's still a ways to go, so here's a roadmap:

- GPU acceleration (DONE!). I built Zor with the idea that we shouldn't need to use "neuromorphic chips" in order to benifit from spike based efficiency, so this is HIGH on the list.
- Saving/Loading. Save models, load them, etc... Plus efficient layer by layer laoding where we only load the parts of the network we need (benifit of sparsity). I hope this could make it possible to use very large models on very limited hardware.
- Research: I made lots of headway on the learning mechanism (non-hebbian, and not-backprop). But I know there is a significant ways to go still.
- Examples: I'll post more examples and code.
- Language models. Since this works well with low data, I hope to be able to use it to translate into low-resource languages on very low-end hardware. Remains to be seen how well this will work for translation.
- Image models: So far I'm very happy with how well the model can learn image representations from very little data.