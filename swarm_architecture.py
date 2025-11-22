"""
Swarm Architecture: Multi-Agent Collective Consciousness

This module implements Phase 2 of the roadmap:
- ConsciousSwarm: Manages 100+ agents with shared workspace
- Collective memory pool leveraging unified memory
- Agent-to-agent communication protocols
- Emergent collective behavior experiments

Hardware Target: Mac Mini M4 Pro (64GB unified memory)
Memory Budget: 100 agents × 128D states = ~50KB active memory
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Any, Optional
import numpy as np
from cognitive_architecture import BicameralAgent, EpisodicMemory, GlobalWorkspace


class CollectiveWorkspace:
    """
    Shared consciousness space for multi-agent swarm.
    
    Architecture:
    - 512D collective state (aggregation of all agent states)
    - Broadcast mechanism: workspace → all agents
    - Voting mechanism: agents → workspace
    """
    def __init__(self, collective_dim: int = 512, num_agents: int = 100):
        self.collective_dim = collective_dim
        self.num_agents = num_agents
        
        # Shared global state (the "collective consciousness")
        self.collective_state = mx.zeros((collective_dim,))
        
        # History for analysis
        self.history = []
        
        # Influence weights (which agents contribute most to collective)
        self.influence_weights = mx.ones((num_agents,)) / num_agents
        
    def aggregate(self, agent_states: List[mx.array]) -> mx.array:
        """
        Aggregate individual agent states into collective consciousness.
        
        Uses weighted voting where influential agents have more say.
        
        Args:
            agent_states: List of 128D agent state vectors
            
        Returns:
            512D collective state vector
        """
        # Stack all agent states: (num_agents, 128)
        stacked = mx.stack(agent_states)
        
        # Weighted average with influence: (num_agents, 128) → (128,)
        weighted_avg = mx.sum(stacked * self.influence_weights[:, None], axis=0)
        
        # Project to collective dimension (128 → 512) via learned mapping
        # For now, use simple tile + noise for exploration
        collective_raw = mx.tile(weighted_avg, (4,))[:self.collective_dim]
        
        # Soft update (momentum-based integration)
        alpha = 0.3
        self.collective_state = (alpha * collective_raw + 
                                 (1 - alpha) * self.collective_state)
        
        self.history.append(self.collective_state.tolist())
        return self.collective_state
    
    def broadcast(self) -> mx.array:
        """
        Broadcast collective state to all agents (top-down modulation).
        
        Returns:
            512D collective state that agents can attend to
        """
        return self.collective_state
    
    def update_influence(self, agent_idx: int, delta: float):
        """
        Adjust an agent's influence on the collective (reputation system).
        
        Args:
            agent_idx: Which agent to update
            delta: +1 for good contribution, -0.1 for bad
        """
        self.influence_weights[agent_idx] = mx.clip(
            self.influence_weights[agent_idx] + delta * 0.01,
            0.01,  # Minimum influence
            1.0    # Maximum influence
        )
        # Re-normalize
        self.influence_weights = self.influence_weights / mx.sum(self.influence_weights)


class SharedMemoryPool:
    """
    Collective episodic memory accessible by all agents.
    
    Leverages unified memory - all agents read/write to same pool.
    10M memories × 128D states = ~5GB (easily fits in 64GB RAM)
    """
    def __init__(self, memory_file: str = "swarm_collective_memory.json", state_dim: int = 128):
        self.memory = EpisodicMemory(
            memory_file=memory_file, 
            expected_state_dim=state_dim
        )
        self.access_log = []  # Track which agents access which memories
        
    def add_experience(self, agent_id: int, state: mx.array, action: mx.array, 
                      reward: float, next_state: mx.array):
        """
        Store an experience with agent attribution.
        
        Args:
            agent_id: Which agent created this memory
            state, action, reward, next_state: Standard RL tuple
        """
        # Add to shared pool
        self.memory.add_episode(state, action, reward, next_state)
        
        # Log access for analysis
        self.access_log.append({
            'agent_id': agent_id,
            'timestamp': len(self.memory.memories),
            'action': 'write'
        })
        
    def retrieve_for_agent(self, agent_id: int, query_state: mx.array, k: int = 3) -> List[Dict]:
        """
        Retrieve memories relevant to an agent's current state.
        
        All agents share the same memory pool (collective knowledge).
        
        Args:
            agent_id: Which agent is querying
            query_state: Agent's current state (128D)
            k: Number of memories to retrieve
            
        Returns:
            List of most relevant memories
        """
        memories = self.memory.retrieve(query_state, k=k)
        
        # Log retrieval
        self.access_log.append({
            'agent_id': agent_id,
            'timestamp': len(self.memory.memories),
            'action': 'read',
            'num_retrieved': len(memories)
        })
        
        return memories


class AgentCommunication:
    """
    Message-passing system for agent-to-agent communication.
    
    Protocols:
    1. Broadcast: One agent sends to all
    2. Unicast: One agent sends to specific other
    3. Multicast: One agent sends to subset
    """
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.message_buffer: Dict[int, List[Dict]] = {i: [] for i in range(num_agents)}
        self.message_history = []
        
    def broadcast(self, sender_id: int, message: mx.array, metadata: Dict = None):
        """
        Send message to all other agents.
        
        Args:
            sender_id: Who is sending
            message: 128D vector encoding the message
            metadata: Optional dict with message type, urgency, etc.
        """
        msg = {
            'from': sender_id,
            'to': 'all',
            'content': message,
            'metadata': metadata or {},
            'timestamp': len(self.message_history)
        }
        
        # Deliver to all except sender
        for agent_id in range(self.num_agents):
            if agent_id != sender_id:
                self.message_buffer[agent_id].append(msg)
        
        self.message_history.append(msg)
        
    def send(self, sender_id: int, receiver_id: int, message: mx.array, metadata: Dict = None):
        """
        Send direct message to one agent.
        
        Args:
            sender_id: Who is sending
            receiver_id: Who receives
            message: 128D vector
            metadata: Optional message info
        """
        msg = {
            'from': sender_id,
            'to': receiver_id,
            'content': message,
            'metadata': metadata or {},
            'timestamp': len(self.message_history)
        }
        
        self.message_buffer[receiver_id].append(msg)
        self.message_history.append(msg)
        
    def receive(self, agent_id: int, max_messages: int = 5) -> List[Dict]:
        """
        Get pending messages for an agent.
        
        Args:
            agent_id: Which agent is checking messages
            max_messages: Limit to prevent overflow
            
        Returns:
            List of message dicts
        """
        messages = self.message_buffer[agent_id][:max_messages]
        self.message_buffer[agent_id] = self.message_buffer[agent_id][max_messages:]
        return messages


class ConsciousSwarm:
    """
    Multi-agent system with collective consciousness.
    
    Phase 2.1 Implementation:
    - 100 BicameralAgents (each with 128D state)
    - Shared 512D CollectiveWorkspace
    - Unified memory pool (10M+ capacity)
    - Agent communication protocols
    
    Memory Footprint:
    - 100 agents × 128D × 4 bytes = 51.2 KB
    - Collective state 512D × 4 bytes = 2 KB
    - Total active memory: ~53 KB (fits in L2 cache!)
    """
    def __init__(self, num_agents: int = 100, agent_state_dim: int = 128, 
                 collective_dim: int = 512, action_dim: int = 10,
                 memory_file: str = "swarm_collective_memory.json"):
        self.num_agents = num_agents
        self.agent_state_dim = agent_state_dim
        self.collective_dim = collective_dim
        
        print(f"🌐 Initializing Conscious Swarm with {num_agents} agents...")
        
        # Individual agents
        self.agents = [
            BicameralAgent(state_dim=agent_state_dim, action_dim=action_dim)
            for _ in range(num_agents)
        ]
        
        # Shared infrastructure
        self.collective_workspace = CollectiveWorkspace(collective_dim, num_agents)
        self.shared_memory = SharedMemoryPool(memory_file=memory_file, state_dim=agent_state_dim)
        self.communication = AgentCommunication(num_agents)
        
        # Swarm-level metrics
        self.step_count = 0
        self.history = {
            'collective_states': [],
            'agent_actions': [],
            'communication_volume': [],
            'consensus_scores': []
        }
        
        print(f"✅ Swarm initialized!")
        print(f"   Memory footprint: ~{self._estimate_memory_mb():.2f} MB")
        
    def _estimate_memory_mb(self) -> float:
        """Estimate active memory usage in MB."""
        agent_memory = self.num_agents * self.agent_state_dim * 4 / 1e6
        collective_memory = self.collective_dim * 4 / 1e6
        return agent_memory + collective_memory
    
    def step(self, environment_signals: List[mx.array], rewards: Optional[List[float]] = None):
        """
        One cognitive cycle for the entire swarm.
        
        Process:
        1. Each agent perceives its local signal
        2. Agents consult collective workspace (top-down)
        3. Agents make decisions
        4. Actions → collective workspace (bottom-up)
        5. Update shared memory pool
        6. Optional: agents communicate
        
        Args:
            environment_signals: List of 128D sensory inputs (one per agent)
            rewards: Optional list of rewards (one per agent)
        """
        if rewards is None:
            rewards = [0.0] * self.num_agents
            
        agent_states = []
        agent_actions = []
        agent_confidences = []
        
        # === PHASE 1: Individual Processing ===
        for i, agent in enumerate(self.agents):
            # Agent perceives local environment
            sensory_input = environment_signals[i]
            
            # Agent accesses collective consciousness (top-down modulation)
            collective_broadcast = self.collective_workspace.broadcast()
            # Agents use first 128D of collective as attention bias
            attention_signal = collective_broadcast[:self.agent_state_dim]
            
            # Combine local + collective signals
            combined_input = sensory_input + 0.3 * attention_signal
            
            # Agent thinks and acts
            decision = agent.step(combined_input, reward=rewards[i])
            
            # Store agent's experience in shared memory
            if agent.last_state is not None:
                self.shared_memory.add_experience(
                    agent_id=i,
                    state=agent.last_state,
                    action=agent.last_action,
                    reward=rewards[i],
                    next_state=decision['state']
                )
            
            agent_states.append(decision['state'])
            agent_actions.append(decision['action'])
            
            # Store confidence if available (default to 1.0 if not)
            conf = decision.get('confidence', mx.array([1.0])).item()
            agent_confidences.append(conf)
        
        # === PHASE 2: Collective Aggregation ===
        # Bottom-up: aggregate all agent states into collective consciousness
        collective_state = self.collective_workspace.aggregate(agent_states)
        
        # === PHASE 3: Communication (Optional, based on uncertainty) ===
        # Agents with low confidence broadcast for help
        for i, agent in enumerate(self.agents):
            # Check decision confidence
            if agent_confidences[i] < 0.4:
                # Broadcast uncertainty signal
                self.communication.broadcast(
                    sender_id=i,
                    message=agent_states[i],
                    metadata={'type': 'help_request', 'confidence': agent_confidences[i]}
                )
        
        # === PHASE 4: Logging ===
        self.history['collective_states'].append(collective_state.tolist())
        self.history['agent_actions'].append(agent_actions)
        self.history['communication_volume'].append(len(self.communication.message_history))
        
        # Consensus score: how similar are agent actions?
        action_variety = len(set(agent_actions))
        consensus = 1.0 - (action_variety / self.num_agents)
        self.history['consensus_scores'].append(consensus)
        
        self.step_count += 1
        
        return {
            'collective_state': collective_state,
            'actions': agent_actions,  # Added for compatibility
            'agent_actions': agent_actions,
            'consensus': consensus,
            'messages_sent': len(self.communication.message_history) - (self.history['communication_volume'][-2] if len(self.history['communication_volume']) > 1 else 0),
            'step': self.step_count
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Compute swarm-level statistics.
        
        Returns:
            Dict with collective behavior metrics
        """
        return {
            'total_steps': self.step_count,
            'shared_memories': len(self.shared_memory.memory.memories),
            'total_messages': len(self.communication.message_history),
            'average_consensus': np.mean(self.history['consensus_scores']) if self.history['consensus_scores'] else 0.0,
            'memory_access_reads': sum(1 for log in self.shared_memory.access_log if log['action'] == 'read'),
            'memory_access_writes': sum(1 for log in self.shared_memory.access_log if log['action'] == 'write'),
        }
    
    def collective_dream(self, batch_size: int = 32, epochs: int = 5):
        """
        Swarm-wide sleep consolidation.
        
        All agents dream using the shared memory pool.
        This is the key advantage of unified memory!
        """
        print(f"\n💤 Swarm entering collective REM sleep...")
        print(f"   Consolidating {len(self.shared_memory.memory.memories)} shared memories...")
        
        for i, agent in enumerate(self.agents):
            # Each agent dreams on the shared memory pool
            agent.dream(batch_size=batch_size, epochs=1)  # Reduced epochs to avoid overfitting
            
            if (i + 1) % 20 == 0:
                print(f"   {i+1}/{self.num_agents} agents dreamed...")
        
        print(f"✨ Collective dreaming complete!")
