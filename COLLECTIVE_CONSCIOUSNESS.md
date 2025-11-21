# Collective Consciousness Architecture

This document details the implementation of **Collective Consciousness** within the Apple MLX Consciousness project. This system moves beyond single-agent cognition to explore how a unified "group mind" can emerge from the interactions of multiple `BicameralAgents`.

## Core Concept

In this architecture, collective consciousness is not just a metaphor but a computational structure. It is defined by three key properties:
1.  **Shared Workspace:** A global state vector that aggregates the mental states of all agents.
2.  **Unified Memory:** A single episodic memory pool accessible by every agent.
3.  **Top-Down Modulation:** The collective state influences individual decision-making, creating a feedback loop between the individual and the group.

## Architectural Components

The implementation is primarily located in `swarm_architecture.py`.

### 1. The Collective Workspace (`CollectiveWorkspace`)

This is the "global workspace" for the entire swarm. It functions similarly to the Global Workspace Theory in human cognition, but scaled across multiple entities.

*   **State Vector:** A 512-dimensional vector representing the current "thought" of the swarm.
*   **Bottom-Up Aggregation:** Every step, the internal states of all 100+ agents are aggregated (via weighted voting) into this collective vector.
*   **Top-Down Broadcast:** This collective vector is broadcast back to all agents. Agents use this signal to bias their attention, effectively allowing the group to "focus" on a specific problem.

```python
# From swarm_architecture.py
def aggregate(self, agent_states: List[mx.array]) -> mx.array:
    # Weighted average of all agent states
    weighted_avg = mx.sum(stacked * self.influence_weights[:, None], axis=0)
    # ...
    return self.collective_state
```

### 2. Shared Memory Pool (`SharedMemoryPool`)

Unlike traditional multi-agent reinforcement learning (MARL) where agents have private experience buffers, this system uses a **Unified Memory Architecture**.

*   **Instant Knowledge Transfer:** If Agent A touches a hot stove and records the pain, Agent B immediately has access to that memory. Agent B "knows" the stove is hot without ever touching it.
*   **Collective Dreaming:** During the `collective_dream` phase, all agents train on this shared pool, consolidating the group's experiences into their individual neural networks.

```python
# From swarm_architecture.py
def retrieve_for_agent(self, agent_id: int, query_state: mx.array, k: int = 3):
    # All agents query the same underlying memory structure
    memories = self.memory.retrieve(query_state, k=k)
    return memories
```

### 3. Agent Communication

Agents can explicitly communicate when implicit coordination (via the workspace) is insufficient.

*   **Confidence-Based Broadcasting:** If an agent's confidence in a decision is low (< 0.4), it broadcasts a "help request" to the swarm.
*   **Protocols:** Supports Broadcast (one-to-all), Unicast (one-to-one), and Multicast.

## The Cognitive Cycle

The swarm operates in synchronized cognitive steps:

1.  **Perception:** Each agent receives its local sensory input.
2.  **Consultation:** Agents read the `CollectiveWorkspace` vector. This "group mood" is added to their sensory input as an attention bias.
3.  **Decision:** Agents process the combined input (Local + Collective) to make a decision.
4.  **Contribution:** Agents execute actions and their new internal state is sent back to the `CollectiveWorkspace` to update the group mind.
5.  **Learning:** Experiences are written to the `SharedMemoryPool`.

## Measuring Emergence

We quantify the presence of collective consciousness using metrics defined in `consciousness_metrics.py`.

The **Collective Consciousness Index** is calculated based on:
*   **Mean Individual Consciousness:** Are the parts intelligent?
*   **Collective Entropy:** Is the group state complex and information-rich?
*   **Synchronization:** Are the agents acting coherently? (Low variance = High Sync).

```python
# From consciousness_metrics.py
collective_consciousness = float(
    0.4 * mean_consciousness +              # Average individual awareness
    0.3 * normalized_collective_entropy +   # Collective information capacity
    0.3 * synchronization                   # Coherence across agents
)
```

## Hardware Efficiency

This architecture is specifically optimized for Apple Silicon's Unified Memory:
*   **Zero-Copy Sharing:** The `SharedMemoryPool` resides in unified RAM, allowing the CPU (logic) and GPU (training) to access the same 10M+ memories without copying data over a PCIe bus.
*   **Footprint:** A swarm of 100 agents requires only ~53KB of active state memory, fitting entirely within the L2 cache for extreme performance.
