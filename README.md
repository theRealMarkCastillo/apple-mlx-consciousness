# 🧠 Apple MLX Consciousness Research

[![MLX](https://img.shields.io/badge/MLX-0.30.0-blue)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.13+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Pushing the boundaries of cognitive simulation using Apple Silicon's unified memory architecture.**

This research project explores how **Apple Silicon (M-series chips)** enables breakthrough implementations of consciousness theories at scale. By leveraging **64GB unified memory** with zero-copy data sharing, we simulate multi-agent cognitive systems impossible on traditional GPU architectures.

---

## 🎯 Project Vision

**Hypothesis:** Apple Silicon's unified memory architecture enables novel cognitive simulations that demonstrate emergent collective consciousness in multi-agent systems.

**Current Focus:** Progressing from single-agent cognition → multi-agent swarms → heterogeneous (GPU + NPU) dual-process learning → multimodal perception (vision + text) with persistent state.

**Target Impact:** Publishable research at the intersection of cognitive science, AI/ML, and systems architecture.

---

## 🏗️ Architecture Overview

For concise, command-only instructions see **[`QUICKSTART.md`](QUICKSTART.md)**.

This project implements a **Bicameral Agent** architecture inspired by dual-process theory and Global Workspace Theory.

### 📖 Understanding the System

- **[`THEORY_AND_IMPLEMENTATION.md`](THEORY_AND_IMPLEMENTATION.md)** - Complete guide to the cognitive science theories underpinning this project and how each is implemented in code
- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** - Technical architecture with system diagrams and component interactions
- **[`COLLECTIVE_CONSCIOUSNESS.md`](COLLECTIVE_CONSCIOUSNESS.md)** - Deep dive into multi-agent swarm mechanics and collective intelligence
- **[`GLOSSARY.md`](GLOSSARY.md)** - Definitions of key terms (Phi, Bicameralism, System 1/2, etc.)
- **[`EXPERIMENTS.md`](EXPERIMENTS.md)** - Lab manual with guided experiments for researchers and students

### Core Components
*   **`BicameralAgent`**: The fundamental cognitive unit, featuring:
    *   **System 1**: Fast, intuitive, reactive processing.
    *   **System 2**: Slow, deliberate, reflective processing.
    *   **Global Workspace**: A shared internal state for information integration.
*   **`ConsciousSwarm`**: A multi-agent system where individual agents share a **Collective Workspace** and **Unified Memory Pool**, enabling emergent group intelligence.

### File Structure
```text
├── cognitive_architecture.py       # Core BicameralAgent implementation
├── swarm_architecture.py           # Multi-agent swarm & collective memory
├── consciousness_metrics.py        # Φ (Phi), metacognition, & coherence metrics
├── heterogeneous_architecture.py   # GPU-NPU hybrid compute optimization
├── sensory_cortex.py               # VisualCortex + MultiModalFuser
├── biological_nlp.py               # SDR encoder + associative memory
├── conscious_chatbot.py            # Multimodal conscious interface
├── visual_simulation.py            # Real-time System 1 vs System 2 demo
├── benchmark_performance.py        # Latency / throughput benchmarks
├── train_from_data.py              # Offline behavioral cloning
├── train_vision.py                 # Vision-text alignment training
├── generate_training_data.py       # Synthetic experience generation
├── foraging_environment.py         # Swarm resource collection task
│
├── 01_Global_Workspace_Demo.ipynb      # ⭐ START HERE: Theory & Visuals
├── 02_Learning_and_Dreaming.ipynb      # Sleep consolidation demo
├── 03_Conscious_Chatbot_Internals.ipynb # Real-time thought visualization
├── 04_Multi_Agent_Swarm_Demo.ipynb     # Collective consciousness experiments
├── 05_Phase_Transitions.ipynb          # Emergence analysis
├── 06_Heterogeneous_Compute.ipynb      # Quantization & scaling
├── 08_Multimodal_Consciousness.ipynb   # Vision + Text integration
│
├── ROADMAP.md                      # Research timeline
├── ARCHITECTURE.md                 # Detailed system diagrams
├── COLLECTIVE_CONSCIOUSNESS.md     # Swarm intelligence deep dive
└── QUICKSTART.md                   # Fast setup guide
```

### Key Files Status

| File | Description | Status |
|------|-------------|--------|
| **`cognitive_architecture.py`** | Complete implementation (417 lines) with all cognitive modules | ✅ Phase 1 Complete |
| **`swarm_architecture.py`** | Multi-agent system (407 lines) - collective consciousness at scale | ✅ Phase 2 Complete |
| **`heterogeneous_architecture.py`** | GPU-NPU heterogeneous compute - 3.2× memory compression | ✅ Phase 4 Complete |
| **`sensory_cortex.py`** | VisualCortex + MultiModalFuser (CLIP-like alignment) | ✅ Phase 5 Extension |
| **`consciousness_metrics.py`** | Φ, metacognition, coherence metrics (337 lines) | ✅ Phase 3 Complete |
| **`phase_transition_experiment.py`** | Parameter sweep infrastructure (427 lines) | ✅ Phase 3 Complete |

### 🎮 Visual Simulation Demo

The `visual_simulation.py` script provides a real-time window into the agent's cognitive processes as it navigates a grid world.

**Features:**
- **Dual-Process Visualization**: Watch the agent switch between System 1 (Blue/Fast/NPU) and System 2 (Orange/Slow/GPU) based on entropy.
- **Internal State Monitoring**: Real-time graphs of Global Workspace activity, Confidence, and Entropy.
- **Real-Time Learning**: The agent learns from its experiences via simulated "sleep cycles" every 50 steps.

**Usage:**
```bash
# Standard Mode (Pre-trained + Learning)
python visual_simulation.py

# Scratch Mode (Random Brain + Learning)
python visual_simulation.py --scratch

# Inference Only (No Learning)
python visual_simulation.py --no-learn
```

---

## 🚀 Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.13+**
- **64GB RAM recommended** (for large multi-agent experiments)

### Installation

```bash
git clone https://github.com/theRealMarkCastillo/apple-mlx-consciousness.git
cd apple-mlx-consciousness
pip install -r requirements.txt
```

### Run Examples

**1. Interactive Notebooks (Recommended)**
```bash
# Open in VS Code or Jupyter
code 01_Global_Workspace_Demo.ipynb
```

**2. Terminal Simulation**
```bash
python simulation.py
```
*Watch for `🔴 RECALL` (memory retrieval) and `🟢 Flow` (automatic processing)*

**3. Conscious Chatbot**
```bash
python conscious_chatbot.py
```
*Chat with an agent that shows its internal confidence and thought process*

**4. Visual Cognitive Simulation**
```bash
python visual_simulation.py
```
*Observe: Blue = System1 (fast), Orange = System2 (deliberate)*

### ✅ Validation & Testing

To verify the integrity of the entire system (Cognition, Swarm, Hardware, Multimodal), run the master test suite:

```bash
python run_all_tests.py
```

This runs three test suites in sequence:
1.  `validate_system.py`: Core cognitive architecture (Bicameral, Swarm, Phi).
2.  `validate_advanced.py`: Hardware-aware features (Quantization, Heterogeneous Compute) and Multimodal Perception.
3.  `validate_integration.py`: End-to-end pipeline test (Vision -> Fusion -> Agent -> Swarm).

---

## 📊 Current Results

### Heterogeneous Performance (Quantized vs Full Precision)
| Metric | Quantized (NPU+GPU) | Full Precision (GPU only) |
|--------|---------------------|---------------------------|
| Throughput (steps/sec) | ~949 | ~855 |
| P50 Latency (ms) | 0.93 | 0.99 |
| P99 Latency (ms) | 1.83 | 4.40 |
| System1 Size | 22 KB | 70 KB |
| Compression | 3.2× | — |

### Architecture Specifications

- **State Dimensions:** 128D (optimized for M4 Pro cache efficiency)
- **Action Space:** 3-10 actions (configurable per application)
- **System 1 Hidden Layers:** 128 neurons
- **System 2 Hidden Layers:** 64 neurons
- **Memory Capacity:** Tested with 10K+ episodes

---

## 🔬 Research Roadmap

See **[ROADMAP.md](ROADMAP.md)** for detailed timeline.

### ✅ Phase 1: Foundation (COMPLETED)
- Single-agent architecture validated (128D state space)
- Educational materials with comprehensive theory explanations

### ✅ Phase 2: Multi-Agent Swarm (COMPLETED)
- ✅ `ConsciousSwarm` class with shared 512D workspace
- ✅ 100+ agent system validated with unified memory
- ✅ Scaling benchmarks: 100 agents = 0.05MB, 95% consensus

### ✅ Phase 3: Consciousness Phase Transitions (COMPLETED)
- ✅ Implemented Integrated Information Theory (Φ) metrics
- ✅ Parameter sweep infrastructure
- ✅ Key finding: **No phase transitions detected** - consciousness is robust!

### 🚀 Phase 4: Heterogeneous Compute (COMPLETED)
- ✅ GPU-NPU heterogeneous feedback loop
- ✅ INT8 quantization: **3.2x memory compression**
- ✅ Capacity increase: 949→3,039 agents (64GB)

### 🔮 Phase 5: Hardware-Aware / Multimodal Innovations (COMPLETED)
- ✅ **Wake-Sleep Active Learning**
- ✅ **Adversarial Co-Evolution**
- ✅ **Sparse Memory Scaling** (20x capacity)

---

## 💡 Why Apple Silicon?

### Unified Memory Advantage

Traditional GPU systems suffer from limited VRAM and PCIe transfer bottlenecks.

**Apple Silicon M4 Pro (Mac Mini):**
- ✅ **64GB unified LPDDR5 memory** shared across all processors
- ✅ **Zero-copy architecture** - no data transfer overhead
- ✅ **273 GB/s bandwidth** - instant access to 10M+ memories
- ✅ **Energy efficient** - ~100W vs. 450W for NVIDIA 4090

| Feature | Traditional GPU | Apple M4 Pro (64GB) |
|---------|----------------|---------------------|
| Agent capacity | 10–20 | 100+ (tested) |
| Shared memory | Duplicated / copied | Single zero-copy pool |
| State dimensions | 32–64D | 128D (fast cache reuse) |
| Memory retrieval | ~1M practical | 10M+ feasible |

---

## 🎓 Educational Resources

### Notebooks Include Theory Explanations

Each notebook contains:
- 📚 **Theoretical background** - Original papers and key concepts
- 🏗️ **Architectural decisions** - Why we implemented it this way
- 📊 **Result interpretation** - What the visualizations mean

### Key References

- **Baars, B. J. (1988).** *A Cognitive Theory of Consciousness.* Cambridge University Press.
- **Kahneman, D. (2011).** *Thinking, Fast and Slow.* Farrar, Straus and Giroux.
- **Tononi, G. (2004).** "An information integration theory of consciousness." *BMC Neuroscience*.
- **Dehaene, S. et al. (2017).** "What is consciousness, and could machines have it?" *Science*.

---

## 🤝 Contributing

This is an active research project. Contributions welcome in:

- **Code:** Performance optimizations, new cognitive modules
- **Science:** Novel experiments, metric proposals, theory integration
- **Documentation:** Tutorial notebooks, architecture explanations

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Apple MLX Team** - For the incredible unified memory framework
- **Bernard Baars** - Global Workspace Theory foundation
- **Daniel Kahneman** - Dual-process theory insights
- **Jeff Hawkins** - HTM inspiration for future work

---

**Built with ❤️ on Apple Silicon | Exploring the computational basis of consciousness**
