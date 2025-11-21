# 📖 Consciousness Research Glossary

This glossary defines key terms used throughout the Apple MLX Consciousness project. It bridges the gap between cognitive science, neuroscience, and computer science.

## 🧠 Cognitive Science & Neuroscience

### Bicameralism
A hypothesis by Julian Jaynes suggesting that the early human mind was divided into two parts: one that "spoke" (hallucinated commands) and one that "obeyed." In this project, we model this as the interaction between **System 1** (proposing actions) and **System 2** (evaluating/critiquing them).

### Dual-Process Theory
A framework proposing two distinct modes of thought:
*   **System 1:** Fast, automatic, unconscious, and emotional (e.g., recognizing a face).
*   **System 2:** Slow, deliberate, conscious, and logical (e.g., solving a math problem).

### Global Workspace Theory (GWT)
A theory by Bernard Baars likening consciousness to a theater stage. Unconscious processes compete for access to the "stage" (Global Workspace). When information wins and is broadcast to the rest of the brain, it becomes conscious.

### Integrated Information Theory (IIT) / Phi ($\Phi$)
A mathematical theory by Giulio Tononi defining consciousness as the capacity of a system to integrate information. $\Phi$ is the measure of this integration. A system has high $\Phi$ if it has many possible states (differentiation) that cannot be reduced to independent parts (integration).

### Metacognition
"Thinking about thinking." The ability of a system to monitor its own cognitive processes, such as estimating confidence in a decision or recognizing when it needs more information.

### Sparse Distributed Representations (SDR)
A theory by Jeff Hawkins (Numenta) stating that the brain represents information using massive vectors where very few neurons are active (sparse) at any given time. This allows for noise robustness and semantic generalization.

---

## 💻 Artificial Intelligence & Architecture

### Collective Consciousness
In this project, the emergent intelligence arising from a swarm of agents sharing a single **Collective Workspace** and **Unified Memory Pool**. It allows individual agents to access the "group mind."

### Heterogeneous Compute
Using different types of processors for different tasks. We use the **Neural Engine (NPU)** for System 1 (fast, efficient) and the **GPU** for System 2 (complex, flexible).

### Latent Space / State Vector
A mathematical representation of the agent's internal state. In our system, the "stream of consciousness" is a 128-dimensional vector evolving over time.

### Quantization
The process of reducing the precision of a neural network's weights (e.g., from 16-bit floating point to 8-bit integers). We use this to make System 1 ultra-fast and memory-efficient on the NPU.

### Unified Memory
Apple Silicon's architecture where the CPU, GPU, and NPU share the same physical memory pool. This allows for **Zero-Copy** data sharing—agents can read each other's memories without expensive data transfers.

### World Model
An internal simulation of the environment. It allows the agent to "imagine" the consequences of its actions without actually performing them.

---

## 🧪 Project-Specific Terms

### Collective Workspace
The shared 512-dimensional vector that aggregates the states of all agents in a swarm. It acts as a top-down attention signal.

### Dreaming / Sleep Consolidation
An offline process where the agent replays past experiences from its episodic memory to train its World Model and System 1 policy. This mimics biological REM sleep.

### Phase Transition
A sudden change in the system's behavior (e.g., the emergence of consciousness) as a parameter (e.g., memory size or integration) crosses a critical threshold.
