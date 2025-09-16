# Zor
Zor is a lightweight, novel spiking-neural-network (SNN) architecture that shows competitive performance to `backpropagation` in low-data settings.

Zor operates on this philosophy:
- **Digital Plausibility** Existing SNNs fail because they try to implement "biological plausibility," when we should aim for digital plausibility. Silicon and biology are different substrates, so the abstract principle `learning` will need to be implemented uniquely on each substrate.

- **spiking** The "spiking" aspect of neural networks is vital to efficient learning, and backpropagation is likely orders of magnitude less efficient than an optimal learning rule.
* **Simplicity** SNNs are typically ridiculously complicated because they essentially try to simulate biological neurons. Zor takes the best of both traditional *ANNs* and efficiency lessons from nature (and my imagination).
* **Framing** Rather than redesign the network for a problem, we should redesign the problem for the network.

## Design

**Normal**: A typical SNN operates by receiving inputs (duh) and propagating them forward; each "neuron" accumulates a "charge." When it reaches a certain threshold, the neuron "spikes" and sends a binary "on" signal to all connected neurons.

**Zor**: Charges are not accumulated; after a neuron spikes, it sends an analogue signal to all connected neurons. This means we don't need to encode information in spiking frequencies, but we still get the benefits of sparse activity and clear eligibility throughout the network.

**Normal**: SNNs often implement either "Hebbian" learning or "Backpropagation Through Time" (BTT). Hebbian learning strengthens connections between coactive neurons, but this often leads to runaway behaviour and requires complex homeostatic pressures that often suppress learning. BTT is terrible for many reasons, including that I generally think that backpropagation is not needed for learning (nothing in nature implements it, and digital nature implements it poorly).

**Zor**: Zor does something vaguely similar to three-factor Hebbian learning, but is also quite different. Rather than looking merely at coactivity, Zor looks at *how* coactive neurons are, and it can do this because we use analog spiking rather than binary. This strengthens the signal enormously. Zor also "backpropagates" errors across the coactivity matrix, weighted by how novel the coactivity is between any two neurons. Unlike many alternatives to backpropagation, Zor can tell to what extent and in what direction each weight should change.

**Normal** Outputs are tricky for SNNs, sometimes people look at their firing frequencies, since the spikes themselves carry no useful information in the output layer. This requires more computation and bluntly, is bad design. Normal neural networks have conventient outputs that can take any value they need to (especially through activation functions). Zor directly returns *charges* for the output layer rather than spikes, the spikes are still usefull for learning though since they carry an important signal! You can also optionally add activation functions to any layer in Zor but I haven't found any example where that helps yet. 

If you want to really know how it works look at the code.

Zor is a mix between "Zeus" and "Thor" and conveniently means strength in several languages.