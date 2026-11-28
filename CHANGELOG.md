# Kaspersky Neuromorphic Platform changelog

## 2026

### Core Framework

### Backends

- CUDABackend (NVIDIA Pascal+ GPU support)
- CUDA SNN training pipeline.
- CUDA backend training stability & performance improved.
- Fixed dynamic backend loading by name.
- Decoupled training vs. inference backend selection in examples.

### Motiv-NT Neuromorphic Hardware Support (External Packages)

- AltAI NPU backend & software emulator.
- ANN-to-SNN conversion toolkit via backpropagation.
- Standardized KNP 2.0 pipeline targeting AltAI-2 NPU & emulator.
- Updated AltAI MNIST example (WTA removal).

### Python API, Tooling & Build System

- C++ / Python SDK (Debian 12.5+, Ubuntu 22.04+, Windows 7/10)
- Encapsulated internal network graph structure in examples/tests
- Improved install_knp.sh installer (#209) & Docker build image (+Graphviz, #219)
- macOS CI workflow Boost fixes (#225)

### Research Tracks (Conference 2026 Talks — NOT yet in KasperskyLab/knp)

- Neuro-astrocytic & neurosemantic networks, Neuromorphic transformers.

## Release for Kaspersky Neuromorphic Platform 2.0 - 2025-12

### Changes

- Distribution kit for developing Linux applications: added packages containing a backend library for GPUs with CUDA support technology.

- Backends:

  - `CUDABackend`: added an interface for the GPU backend.
  - `SingleThreadedCPUBackend` backend: added neuron support based on the AltAILIF model.
  - `MultiThreadedCPUBackend` backend: added neuron support based on the SynapticResourceSTDPBLIFATNeuron model and synapse based on the SynapticResourceSTDPDeltaSynapse model.
  - Optimized CPU backend operation.

- C++ components:

  - `core` library: extended the functionality for the `Backend`, `MessageBus`, `MessageEndpoint`, `Population`, `Projection`, `Subscription`, and `TagMap` classes.

  - Framework for C ++: extended the functionality of the C++ framework:

    - `data_processing` namespace: added a namespace that provides functionality for processing classification model training and testing data.
    - `inference_evaluation` namespace: added a namespace that provides functionality for evaluating model quality.
    - `modifier` namespace: added a namespace that provides message handlers.
    - `normalization` namespace: added a namespace that provides functions and functors for normalizing the parameters of neurons and synapses in neural networks.
    - `monitoring` namespace: added the model namespace with model monitoring functions.
    - `projection` namespace: added functions to generate synapses and synapse projections.
    - `ModelExecutor` class: added methods.

  - `neuron_traits` library: extended the neuron interfaces for the `BLIFATNeuron` and `SynapticResourceSTDPBLIFATNeuron` models.

- Components for use in Python: added methods for the `ModelExecutor` and `UID` classes.

- `synapse_traits` library (for use in C++ and Python): extended the interface of the synapse based on the `SynapticResourceSTDPDeltaSynapse` model.

- Example `mnist-learn` is an added example of training a neural network on images of handwritten numbers and their labels from the MNIST database and its execution.

© 2026 AO Kaspersky Lab
