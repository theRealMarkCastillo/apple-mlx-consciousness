# 🧪 Experiments Lab Manual

This document is a guide for curious researchers to run specific experiments using the provided codebase. It is designed to help you explore the properties of consciousness, memory, and swarm intelligence.

---

## Experiment 1: The "Confusion" Threshold
**Hypothesis:** As the environment becomes more chaotic (random), System 2 intervention (deliberation) should increase.

### Setup
1. Open `01_Global_Workspace_Demo.ipynb`.
2. Locate the simulation loop.
3. Modify the `sensory_input` noise level.

### Procedure
Run the simulation twice with different noise levels:

**Run A (Low Noise):**
```python
# Low variance input
sensory_input = mx.random.normal((32,)) * 0.1 
```

**Run B (High Noise):**
```python
# High variance input
sensory_input = mx.random.normal((32,)) * 2.0
```

### What to Observe
*   Check the **Intervention Rate** (how often `decision['memory_retrieved']` is True).
*   **Expected Result:** Run B should trigger significantly more memory retrievals and imagination steps as System 1's confidence drops.

---

## Experiment 2: The Benefit of Dreaming (Prioritized Replay)
**Hypothesis:** "Dreaming" (offline training) significantly reduces the agent's surprise (prediction error) when facing the world. Prioritized Replay should make this process faster by focusing on hard examples.

### Setup
1. Open `02_Learning_and_Dreaming.ipynb`.
2. This notebook is already set up for this experiment.

### Procedure
1. Run the "Pre-Sleep" simulation and note the **World Model MSE** (Mean Squared Error).
2. Run the "Dreaming" cell. The agent will automatically prioritize high-surprise memories.
3. Run the "Post-Sleep" simulation.

### What to Observe
*   Compare the MSE before and after sleep.
*   **Expected Result:** The agent should be much better at predicting the outcome of its actions after dreaming. This demonstrates **consolidation**.

---

## Experiment 3: Curiosity vs. Boredom
**Hypothesis:** An agent with Intrinsic Motivation (Curiosity) will explore the environment more thoroughly than a random agent.

### Setup
1. Open `simulation.py`.
2. Observe the `SURPRISE` column in the output.

### Procedure
1. Run the simulation.
2. Watch how the `SURPRISE` value changes over time.

### What to Observe
*   Initially, `SURPRISE` should be high (everything is new).
*   As the agent learns, `SURPRISE` should drop.
*   **Expected Result:** If the agent gets "bored" (low surprise), it might change its behavior to find new patterns (if the environment allows).

---

## Experiment 4: Swarm Consensus Speed
**Hypothesis:** Larger swarms take longer to reach consensus, but the consensus is more stable.

### Setup
1. Open `04_Multi_Agent_Swarm_Demo.ipynb`.
2. Locate the `ConsciousSwarm` initialization.

### Procedure
Run the simulation with different agent counts:

**Run A (Small Swarm):**
```python
swarm = ConsciousSwarm(num_agents=10, ...)
```

**Run B (Large Swarm):**
```python
swarm = ConsciousSwarm(num_agents=100, ...)
```

### What to Observe
*   Look at the `consensus` metric in the output logs.
*   Count how many steps it takes for consensus to reach > 0.9.
*   **Expected Result:** The small swarm aligns quickly but may fluctuate. The large swarm has "inertia"—it takes longer to align but resists noise better.

---

## Experiment 4: The "Zombie" Agent (Ablation Study)
**Hypothesis:** An agent without System 2 (pure System 1) will fail in novel situations but perform fine in familiar ones.

### Setup
1. Open `simulation.py` or create a new script.
2. Modify `BicameralAgent.step()` in `cognitive_architecture.py` (temporarily) to force `confidence = 1.0`.

### Procedure
```python
# In cognitive_architecture.py -> step()
# Force high confidence to bypass System 2
confidence = mx.array([1.0]) 
```

### What to Observe
*   Run the agent in a changing environment.
*   Observe the **Entropy** of its actions.
*   **Expected Result:** The "Zombie" agent will act confidently even when it is wrong (high entropy in action probabilities but no intervention). It lacks **Metacognition**.

---

## Experiment 5: Memory Capacity & Intelligence
**Hypothesis:** Increasing the memory capacity allows the agent to solve more complex temporal patterns.

### Setup
1. Use `01_Global_Workspace_Demo.ipynb`.
2. Increase `steps` to 200 to generate a longer history.

### Procedure
1. Run with `k=1` in memory retrieval (access only the single most relevant memory).
2. Run with `k=10` (access broad context).

### What to Observe
*   Does the agent stabilize its entropy faster with access to more memories?
*   **Interpretation:** Access to a richer past allows for better grounding of the present.
