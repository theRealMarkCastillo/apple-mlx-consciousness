# 🚀 Quickstart: Apple MLX Consciousness

Concise setup & run instructions. For full theory see `README.md` and notebooks.

---
## 1. Prerequisites
- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.13+
- Recommended: 32GB+ RAM (64GB for large swarms)

```bash
python --version
```

---
## 2. Install
```bash
git clone https://github.com/theRealMarkCastillo/apple-mlx-consciousness.git
cd apple-mlx-consciousness
pip install -r requirements.txt
```

Optional: Create venv
```bash
python -m venv .venv
source .venv/bin/activate
```

---
## 3. Core Demos
### Global Workspace (Educational)
```bash
code 01_Global_Workspace_Demo.ipynb
```
### Conscious Chatbot (Stateful)
```bash
python conscious_chatbot.py
```
Type `quit` to auto-save brain + memory.

### Simulation (Text-Only Cognitive Cycle)
```bash
python simulation.py
```

---
## 4. Vision & Multimodal Grounding
Train synthetic shape/color alignment (creates `visual_cortex.npz`, `multimodal_fuser.npz`):
```bash
python train_vision.py
```
Then run chatbot again to load them.

---
## 5. Offline Behavioral Cloning (System 1 Intuition)
Generate supervised experiences and train System 1; produces `agent_brain.npz`:
```bash
python generate_training_data.py      # creates synthetic_experiences.json
python train_from_data.py             # trains + saves brain
```

---
## 6. Performance & Visualization
Latency / throughput benchmark:
```bash
python benchmark_performance.py
```
Real-time System1 vs System2 demo (blue fast; orange deliberative):
```bash
python visual_simulation.py
```

---
## 7. Swarm & Advanced Experiments
```bash
python 04_Multi_Agent_Swarm_Demo.ipynb     # Notebook swarm demo
python foraging_environment.py             # Resource collection task
python heterogeneous_training.py           # Wake-sleep active learning
python adversarial_coevolution.py          # Robustness evolution
python sparse_memory.py                    # High-capacity memory variant
```

---
## 8. Persistence Files (auto-managed)
Ignored by git:
- `agent_brain.npz` (System 1 weights)
- `visual_cortex.npz`, `multimodal_fuser.npz` (vision grounding)
- `long_term_memory.json` (associative memory)
- Generated datasets & synthetic experiences

Delete any to retrain from scratch.

---
## 9. Troubleshooting
| Issue | Fix |
|-------|-----|
| Slow startup | Ensure no large rogue `.npz` files; reinstall deps |
| Vision load fails | Run `python train_vision.py` before chatbot |
| Memory not saving | Use `quit` to trigger `save_state()` |
| Missing MLX ops | Upgrade MLX: `pip install --upgrade mlx` |

---
## 10. Next Steps
- Measure energy (`sudo powermetrics --samplers tasks --show-all -n 5`)
- Extend vision dataset beyond synthetic primitives
- Add reinforcement loop to integrate reward-driven adaptation

---
**Happy experimenting.** See `ROADMAP.md` for research directions.
