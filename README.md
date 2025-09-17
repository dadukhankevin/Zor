# Zor

Zor is a lightweight, novel spiking-neural-network (SNN) architecture that shows competitive performance to `backpropagation` in low-data settings.

## Philosophy

Zor operates on this philosophy:

- **Digital Plausibility** -> Existing SNNs fail because they try to implement "biological plausibility," when we should aim for digital plausibility. Silicon and biology are different substrates, so the abstract principle `learning` will need to be implemented uniquely on each substrate.

- **Spiking** -> The "spiking" aspect of neural networks is vital to efficient learning, and backpropagation is likely orders of magnitude less efficient than an optimal learning rule.

- **Simplicity** -> SNNs are typically ridiculously complicated because they essentially try to simulate biological neurons. Zor takes the best of both traditional *ANNs* and efficiency lessons from nature (and my imagination).

- **Framing** -> Rather than redesign the network for a problem, we should redesign the problem for the network.

## Design

### Signal Propagation

**Normal**: A typical SNN operates by receiving inputs (duh) and propagating them forward; each "neuron" accumulates a "charge." When it reaches a certain threshold, the neuron "spikes" and sends a binary "on" signal to all connected neurons.

**Zor**: Charges are not accumulated; after a neuron spikes, it sends an analogue signal to all connected neurons. This means we don't need to encode information in spiking frequencies, but we still get the benefits of sparse activity and clear eligibility throughout the network.

### Learning

**Normal**: SNNs often implement either "Hebbian" learning or "Backpropagation Through Time" (BTT). Hebbian learning strengthens connections between coactive neurons, but this often leads to runaway behaviour and requires complex homeostatic pressures that often suppress learning. BTT is terrible for many reasons, including that I generally think that backpropagation may not be needed for learning (nothing in nature implements it, and "digiture" implements it poorly).

**Zor**: Zor does something *vaguely similar* to three-factor Hebbian learning, but is also quite different. Rather than looking merely at coactivity, Zor looks at *how* coactive neurons are, and it can do this because we use analog spiking rather than binary. This strengthens the signal enormously. Zor also "backpropagates" errors across the coactivity matrix, weighted by how novel the coactivity is between any two neurons. Unlike many alternatives to backpropagation, Zor can tell to what extent and in what direction each weight should change.

## What's New Here

While Zor builds on established concepts like three-factor learning rules and homeostatic plasticity, it combines them in a novel way:

- **Analog-spike gating**: Binary spike decisions gate analog charges, which are then used directly in learning (not typical in SNNs)
- **Subtractive novelty gating**: Familiarity penalties reduce eligibility rather than adding novelty bonuses
- **Threshold homeostasis**: Direct target-rate control of sparsity during learning
- **Reconstruction-based curriculum**: Sample replacement based on per-item reconstruction quality

## Relation to prior work

Zor builds on known ingredients but combines them differently:

- Three‑factor rules and reward‑modulated STDP typically use a global scalar modulatory signal (e.g., dopamine) multiplying pre/post activity; they do not propagate vector errors layer‑by‑layer. See e.g. Frémaux & Gerstner (review) [Frontiers](https://www.frontiersin.org/articles/10.3389/fncir.2015.00085/full).
- e‑prop uses local eligibility traces with neuron‑wise learning signals to approximate BPTT in SNNs; it still separates eligibility from a broadcast learning signal. See Bellec et al. 2020 [Nature](https://www.nature.com/articles/s41586-020-2019-3).
- DECOLLE applies continuous local losses with surrogate gradients at each layer (no explicit error propagation through weights). See Kaiser et al. 2019 [arXiv](https://arxiv.org/abs/1901.09049).
- Feedback alignment propagates vector errors with fixed/random feedback matrices rather than exact transposes. See Lillicrap et al. 2016 [Nat. Comms](https://www.nature.com/articles/ncomms13276).

How Zor differs:

- Uses a vector error per layer obtained by multiplying by the current forward weights’ transpose (no BPTT, no random feedback).
- Learns with analog‑spike gating (binary spike decides passage of the analog charge) in both forward and eligibility terms.
- Applies a subtractive novelty gate (EMA co‑activation penalty) rather than additive novelty bonuses.
- Couples per‑unit threshold homeostasis with a simple reconstruction‑based curriculum.

Net: overlaps in spirit with three‑factor/e‑prop/FA, but the specific mechanics and their integration here are, to our knowledge, uncommon.

### Outputs

**Normal**: Outputs are tricky for SNNs, sometimes people look at their firing frequencies, since the spikes themselves carry no useful information in the output layer. This requires more computation and bluntly, is bad design. Normal neural networks have conventient outputs that can take any value they need to (especially through activation functions).

**Zor**: Zor directly returns instant *charges* for the output layer rather than spikes, the spikes are still usefull for learning though since they carry an important signal! You can also optionally add activation functions to any layer in Zor but I haven't found any example where that helps yet. Seems they aren't needed.

**Zor** is also pretty stable, and not prone to seizures like some SNNs (: The homeostatic measures taken also help learning rather than hinder it.

---

If you want to really know how it works look at the code.

*Zor is a mix between "Zeus" and "Thor" and conveniently means strength in several languages.*

## Roadmap
Zor is the result of many, many, sleepless nights pulling my hair out over how stupid backpropagation is vs how simple I felt like spike-based learning should be, and there's still a ways to go, so here's a roadmap:

- GPU acceleration. I built Zor with the idea that we shouldn't need to use "neuromorphic chips" in order to benifit from spike based efficiency, so this is HIGH on the list.
- Saving/Loading. Save models, load them, etc... Plus efficient layer by layer laoding where we only load the parts of the network we need (benifit of sparsity). I hope this could make it possible to use very large models on very limited hardware.
- Research: I made lots of headway on the learning mechanism (non-hebbian, and not-backprop). But I know there is a significant ways to go still.
- Examples: I'll post more examples and code.
- Language models. Since this works well with low data, I hope to be able to use it to translate into low-resource languages on very low-end hardware. Remains to be seen how well this will work for translation.
- Image models: So far I'm very happy with how well the model can learn image representations from very little data.