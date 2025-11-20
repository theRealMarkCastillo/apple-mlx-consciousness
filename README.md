# 🧠 Apple MLX Consciousness Research

[![MLX](https://img.shields.io/badge/MLX-0.30.0-blue)](https://github.com/ml-explore/mlx)
[![Python](https://img.shields.io/badge/Python-3.13+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Pushing the boundaries of cognitive simulation using Apple Silicon's unified memory architecture.**

This research project explores how **Apple Silicon (M-series chips)** enables breakthrough implementations of consciousness theories at scale. By leveraging **64GB unified memory** with zero-copy data sharing, we simulate multi-agent cognitive systems impossible on traditional GPU architectures.

---

## 🎯 Project Vision

**Hypothesis:** Apple Silicon's unified memory architecture enables novel cognitive simulations that demonstrate emergent collective consciousness in multi-agent systems.

**Current Focus:** Building from single-agent demonstrations to 100+ agent swarms with shared episodic memory and collective decision-making.

**Target Impact:** Publishable research at the intersection of cognitive science, AI/ML, and systems architecture.

---

## 🏗️ Architecture: "Bicameral Global Workspace"

The core `BicameralAgent` integrates multiple consciousness theories:

### 1. **Global Workspace Theory (Baars, 1988)**
- Shared "stream of consciousness" broadcasts information to all cognitive modules
- Implemented as a persistent state vector in unified memory
- **Apple Silicon Advantage:** Zero-copy access from CPU, GPU, and Neural Engine

### 2. **Dual-Process Theory (Kahneman, 2011)**
- **System 1:** Fast, automatic, intuitive processing (shallow neural network)
- **System 2:** Slow, deliberate, meta-cognitive monitoring (evaluates confidence)
- Detects uncertainty via entropy and triggers intervention (recall or imagination)

### 3. **Episodic Memory (Hippocampal Model)**
- Content-addressable storage: (State, Action, Reward, NextState) tuples
- GPU-accelerated cosine similarity retrieval (10M+ memories possible)
- **Apple Silicon Advantage:** All agents share one memory pool with instant access

### 4. **World Model (Prefrontal Cortex)**
- Predictive neural network: `f(state, action) → next_state`
- Enables "imagination" - mental simulation before acting
- Trained during "sleep" cycles via batch consolidation

### 5. **Meta-Cognition & Top-Down Attention**
- System 2 sets goal vectors that bias perception
- Implements selective attention and confirmation bias
- Monitors self-confidence to trigger deliberate reasoning

---

## 📂 Repository Structure

```
apple-mlx-consciousness/
├── cognitive_architecture.py      # Core: BicameralAgent, GlobalWorkspace, Memory, WorldModel
├── simulation.py                  # CLI demo: Stream of consciousness with recall/imagination
├── conscious_chatbot.py           # Application: Chatbot with internal meta-cognition
│
├── 01_Global_Workspace_Demo.ipynb     # 📊 Educational: Theory + visualizations
├── 02_Learning_and_Dreaming.ipynb     # 🌙 Experiment: Sleep consolidation benefits
├── 03_Conscious_Chatbot_Internals.ipynb  # 💬 Analysis: Real-time thought visualization
│
├── requirements.txt               # Python dependencies (pinned versions)
├── README.md                      # This file
├── ROADMAP.md                     # Research plan & milestones
└── .gitignore                     # Version control settings
```

### Key Files

| File | Description |
|------|-------------|
| **`cognitive_architecture.py`** | Complete implementation of the Bicameral architecture with all cognitive modules |
| **`01_Global_Workspace_Demo.ipynb`** | ⭐ **START HERE** - Educational notebook with theory explanations and visualizations |
| **`ROADMAP.md`** | Research timeline: single-agent → multi-agent swarm → phase transitions |
| **`simulation.py`** | Terminal-based demo showing flow states, recall events, and dreaming cycles |

---

## 🚀 Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.13+**
- **64GB RAM recommended** (for multi-agent experiments)
  - **Tested on:** Mac Mini M4 Pro (Model: Mac16,11)
  - **CPU:** 14 cores (10 performance + 4 efficiency)
  - **GPU:** 20 cores with Metal 4
  - **Memory:** 64GB LPDDR5 unified

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/apple-mlx-consciousness.git
cd apple-mlx-consciousness

# Install dependencies
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

---

## 📊 Current Results (Phase 1: Single Agent)

### Key Metrics from `01_Global_Workspace_Demo.ipynb`

- **Simulation:** 50 cognitive cycles with random sensory input
- **Memory:** 89 episodic experiences stored during run
- **Dreaming:** World Model MSE improved to 0.4438 after sleep consolidation
- **Meta-Cognition:** Correlation between entropy and confidence: -0.0124 (needs improvement)
- **Cognitive Strategies:**
  - **Flow (blue):** Automatic processing - 17 instances
  - **Imagination (green):** World Model simulation - 33 instances  
  - **Recall (red):** Memory retrieval - sparse (early training phase)

### Visualizations Generated

1. **Global Workspace Heatmap** - "Stream of consciousness" evolving over time
2. **Entropy vs. Confidence** - Meta-cognitive monitoring dynamics
3. **Goal Vector Heatmap** - Top-down attention patterns
4. **Action Probability Stream** - Decision-making dynamics
5. **Cognitive Phase Space** - State-of-mind scatter plot (flow/recall/imagination zones)

---

## 🔬 Research Roadmap

See **[ROADMAP.md](ROADMAP.md)** for detailed timeline. Summary:

### ✅ Phase 1: Foundation (Completed)
- Single-agent architecture validated
- Educational materials with theory explanations
- Baseline performance metrics established

### 🎯 Phase 2: Multi-Agent Swarm (Current - Weeks 1-6)
- Design shared consciousness architecture
- Implement 100-agent system leveraging unified memory
- Study emergent collective behavior (culture, communication, roles)

### 🌙 Phase 3: Consciousness Phase Transitions (Weeks 7-16)
- Parameter sweeps to identify consciousness "tipping points"
- Implement Integrated Information Theory (Φ) metrics
- Publish findings: emergent consciousness at scale

### 🚀 Phase 4: Advanced Extensions (Future)
- Hierarchical Temporal Memory (HTM) integration
- Neural Engine optimization (hybrid CPU/GPU/ANE)
- Lifelong learning without catastrophic forgetting

---

## 💡 Why Apple Silicon?

### Unified Memory Advantage

Traditional GPU systems suffer from:
- ❌ Limited VRAM (16-24GB typical)
- ❌ PCIe transfer bottleneck (CPU ↔ GPU copying)
- ❌ Separate memory pools for multi-agent systems

**Apple Silicon M4 Pro (Mac Mini):**
- ✅ **14-core CPU** (10 performance + 4 efficiency cores)
- ✅ **20-core GPU** with Metal 4 support
- ✅ **16-core Neural Engine** for ML acceleration
- ✅ **64GB unified LPDDR5 memory** shared across all processors
- ✅ **Zero-copy architecture** - no data transfer overhead
- ✅ **273 GB/s bandwidth** - instant access to 10M+ memories
- ✅ **Energy efficient** - ~100W vs. 450W for NVIDIA 4090

### Concrete Benefits for Cognitive Simulation

| Feature | Traditional GPU | Apple M4 Pro (Mac Mini, 64GB) |
|---------|----------------|-------------------------------|
| **Agent capacity** | 10-20 agents | 100+ agents |
| **Shared memory** | Must duplicate data | Single shared pool |
| **Memory retrieval** | 1M memories max | 10M+ memories |
| **Transfer overhead** | 10-100ms per copy | Zero (unified) |
| **Multi-model execution** | Sequential | Parallel (CPU+GPU+ANE) |

---

## 🎓 Educational Resources

### Notebooks Include Theory Explanations

Each notebook contains:
- 📚 **Theoretical background** - Original papers and key concepts
- 🏗️ **Architectural decisions** - Why we implemented it this way
- 📊 **Result interpretation** - What the visualizations mean
- 🔧 **Suggested experiments** - How to extend the research

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
- **Benchmarking:** Comparisons with other frameworks/hardware

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-idea`)
3. Commit changes with clear messages
4. Push and open a Pull Request

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

## 📬 Contact & Updates

- **GitHub Issues:** [Report bugs or request features](https://github.com/yourusername/apple-mlx-consciousness/issues)
- **Discussions:** [Share ideas and results](https://github.com/yourusername/apple-mlx-consciousness/discussions)
- **Twitter/X:** [@yourusername](https://twitter.com/yourusername) - Follow for updates

---

## 🌟 Star History

If you find this research useful, please ⭐ star the repository to help others discover it!

---

**Built with ❤️ on Apple Silicon | Exploring the computational basis of consciousness**
