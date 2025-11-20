import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import json
import os
import time
from typing import List, Dict, Any

class EpisodicMemory:
    """
    Represents the Hippocampus / Long-term storage.
    Stores 'episodes' (State, Action, Reward, NextState) and allows retrieval
    based on vector similarity (content-addressable memory).
    """
    def __init__(self, memory_file: str = "episodic_memory.json", expected_state_dim: int = None):
        self.memories: List[Dict[str, Any]] = []
        self.memory_file = memory_file
        self.memory_matrix = None
        self.expected_state_dim = expected_state_dim
        self.load_memories()

    def add_episode(self, state: mx.array, action: mx.array, reward: float, next_state: mx.array):
        """
        Encodes and stores an episode.
        """
        episode = {
            "timestamp": time.time(),
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": reward,
            "next_state": next_state.tolist() if next_state is not None else None
        }
        self.memories.append(episode)
        
        # Update GPU memory matrix
        if self.memory_matrix is None:
            self.memory_matrix = state[None, :]
        else:
            # Dimension mismatch check - clear old memories if incompatible
            if self.memory_matrix.shape[1] != state.shape[0]:
                print(f"⚠️  Memory dimension mismatch ({self.memory_matrix.shape[1]} -> {state.shape[0]}). Clearing old memories.")
                self.memories = [episode]
                self.memory_matrix = state[None, :]
            else:
                self.memory_matrix = mx.concatenate([self.memory_matrix, state[None, :]], axis=0)

        # Auto-save every 10 memories for persistence during debug
        if len(self.memories) % 10 == 0:
            self.save_memories()

    def retrieve(self, query_state: mx.array, k: int = 3):
        """
        Retrieves the top-k most similar memories to the query_state.
        Uses vectorized GPU operations for speed.
        """
        if not self.memories or self.memory_matrix is None:
            return []

        # Vectorized Cosine Similarity on GPU
        # query_state: (D,)
        # memory_matrix: (N, D)
        
        # Normalize query
        q_norm = query_state / (mx.linalg.norm(query_state) + 1e-9)
        
        # Normalize memories (N, D)
        m_norms = mx.linalg.norm(self.memory_matrix, axis=1, keepdims=True)
        m_normalized = self.memory_matrix / (m_norms + 1e-9)
        
        # Dot product: (N, D) @ (D,) -> (N,)
        similarities = m_normalized @ q_norm
        
        # Get top k indices
        # argsort sorts in ascending order, so we negate similarities to get descending
        indices = mx.argsort(-similarities)
        top_k_indices = indices[:k].tolist()
        
        return [self.memories[i] for i in top_k_indices]

    def save_memories(self):
        """Persist to JSON for debugging."""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, indent=2)

    def load_memories(self):
        """Load from JSON if exists and rebuild GPU matrix."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.memories = json.load(f)
                
                # Rebuild GPU matrix from loaded memories
                if self.memories:
                    # Validate dimension compatibility
                    first_state_dim = len(self.memories[0]["state"])
                    if self.expected_state_dim is not None and first_state_dim != self.expected_state_dim:
                        print(f"⚠️  Memory dimension mismatch on load ({first_state_dim}D → {self.expected_state_dim}D). Clearing old memories.")
                        self.memories = []
                        self.memory_matrix = None
                    else:
                        states = [mx.array(m["state"]) for m in self.memories]
                        self.memory_matrix = mx.stack(states)
                else:
                    self.memory_matrix = None
                    
            except json.JSONDecodeError:
                self.memories = []
                self.memory_matrix = None
        else:
            self.memories = []
            self.memory_matrix = None

class GlobalWorkspace:
    """
    Represents the 'Global Workspace' - a shared memory space that 
    broadcasts information to all cognitive modules.
    In MLX terms, this is a persistent state vector in Unified Memory.
    """
    def __init__(self, state_dim: int = 128):
        self.state_dim = state_dim
        # The "Stream of Consciousness" - current global state
        self.current_state = mx.zeros((state_dim,))
        # Top-Down Attention Goal Vector (set by System 2)
        self.goal_vector = mx.zeros((state_dim,))
        self.history = []

    def update(self, new_info: mx.array, attention_weight: float = 0.5):
        """
        Update the global workspace with new information.
        This mimics 'broadcasting' new content into consciousness.
        
        Includes Top-Down Attention:
        If a goal is active, we bias the update towards information that aligns with the goal.
        """
        # 1. Apply Top-Down Attention
        # If new_info aligns with goal_vector, boost its weight
        if mx.max(mx.abs(self.goal_vector)) > 0:
            # Simple dot product attention
            relevance = mx.sum(new_info * self.goal_vector)
            # Sigmoid to scale relevance between 0 and 1, then shift to boost
            attention_boost = mx.sigmoid(relevance) 
            effective_weight = attention_weight * (1.0 + attention_boost)
            effective_weight = mx.minimum(effective_weight, 1.0) # Clamp at 1.0
        else:
            effective_weight = attention_weight

        # 2. Soft update (moving average) representing integration of new thought
        self.current_state = (1 - effective_weight) * self.current_state + effective_weight * new_info
        self.history.append(self.current_state)
        return self.current_state

    def set_goal(self, goal: mx.array):
        """System 2 sets a goal to bias future perception."""
        self.goal_vector = goal

    def get_state(self):
        return self.current_state

class System1(nn.Module):
    """
    Fast, intuitive, unconscious processor.
    Proposes actions/thoughts rapidly based on current state.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def __call__(self, x):
        x = self.activation(self.l1(x))
        logits = self.l2(x)
        return logits

class System2(nn.Module):
    """
    Slow, deliberate, conscious monitor.
    Evaluates the 'entropy' or 'confidence' of System 1's proposal.
    If confidence is low, it intervenes (simulated here as a modulation).
    Now also capable of Goal-Setting (Top-Down Attention).
    """
    def __init__(self, input_dim: int, hidden_dim: int, goal_dim: int):
        super().__init__()
        # Input is concatenation of [GlobalState, System1_Logits]
        self.l1 = nn.Linear(input_dim, hidden_dim)
        
        # Head 1: Confidence/Approval Score
        self.l_score = nn.Linear(hidden_dim, 1) 
        
        # Head 2: Goal Vector (for Top-Down Attention)
        self.l_goal = nn.Linear(hidden_dim, goal_dim)
        
        self.activation = nn.Tanh() 

    def __call__(self, x):
        x = self.activation(self.l1(x))
        
        # Confidence: 0 to 1
        score = mx.sigmoid(self.l_score(x))
        
        # Goal: -1 to 1 vector
        goal = mx.tanh(self.l_goal(x))
        
        return score, goal

class WorldModel(nn.Module):
    """
    The 'Imagination Engine'.
    Predicts the future state given the current state and an action.
    Allows the agent to simulate outcomes without acting.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, state_dim)
        self.activation = nn.ReLU()

    def __call__(self, state, action):
        # Concatenate State + Action
        x = mx.concatenate([state, action], axis=-1)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        next_state_pred = self.l3(x)
        return next_state_pred

class BicameralAgent:
    """
    An agent that combines System 1 and System 2 via a Global Workspace.
    Enhanced with:
    - Episodic Memory (Hippocampus)
    - Plasticity (Learning)
    - Goal-Directed Planning (via Memory Retrieval)
    - World Model (Imagination/Dreaming)
    """
    def __init__(self, state_dim=128, action_dim=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.workspace = GlobalWorkspace(state_dim)
        self.memory = EpisodicMemory(expected_state_dim=state_dim)
        
        self.system1 = System1(state_dim, 128, action_dim)
        # System 2 sees state + proposed action logits
        self.system2 = System2(state_dim + action_dim, 64, goal_dim=state_dim)
        
        # World Model for Imagination
        self.world_model = WorldModel(state_dim, action_dim)
        
        # Plasticity: Optimizer for System 1 and World Model
        self.optimizer = optim.Adam(learning_rate=0.01)
        self.wm_optimizer = optim.Adam(learning_rate=0.01)

        # State tracking for transition storage (s_t, a_t, r_t, s_{t+1})
        self.last_state = None
        self.last_action = None

    def step(self, sensory_input: mx.array, reward: float = 0.0):
        """
        One cognitive cycle.
        1. Update Workspace with sensory input (modulated by Attention).
        2. Store previous experience (if any).
        3. System 1 proposes action.
        4. System 2 evaluates proposal & sets goals.
        5. Planning: If uncertain, query Memory OR Imagine outcomes.
        6. Action Selection.
        7. Learning (Plasticity).
        """
        # 1. Perception: Update Global Workspace
        current_state = self.workspace.update(sensory_input)

        # 2. Store Experience (S_prev, A_prev, R, S_curr)
        if self.last_state is not None and self.last_action is not None:
            self.memory.add_episode(self.last_state, self.last_action, reward, current_state)

        # 3. Intuition: System 1 proposes action
        logits = self.system1(current_state)
        
        # 4. Reflection: System 2 evaluates (State + Proposal)
        sys2_input = mx.concatenate([current_state, logits], axis=0)
        confidence, new_goal = self.system2(sys2_input)
        
        # Update the goal for the NEXT cycle
        self.workspace.set_goal(new_goal)
        
        # 5. Planning / Deliberation (System 2 Intervention)
        is_confident = confidence.item() > 0.5
        imagination_used = False
        
        if not is_confident:
            # Strategy A: Recall (Fast, based on past)
            memories = self.memory.retrieve(current_state, k=3)
            # ... (Memory bias logic same as before) ...
            if memories:
                memory_bias = mx.zeros_like(logits)
                for mem in memories:
                    past_action = mx.array(mem['action'])
                    past_reward = mem['reward']
                    if past_reward > 0:
                        memory_bias = memory_bias + (past_action * past_reward)
                    else:
                        memory_bias = memory_bias - (past_action * abs(past_reward))
                logits = logits + (memory_bias * 0.5)

            # Strategy B: Imagination (Slow, uses World Model)
            # If still uncertain or if no memories, try to simulate future
            # We simulate the outcome of the top action candidate
            top_action_idx = mx.argmax(logits).item()
            top_action_vec = mx.zeros((self.action_dim,))
            top_action_vec[top_action_idx] = 1.0
            
            predicted_next_state = self.world_model(current_state, top_action_vec)
            # If predicted state is "bad" (e.g. high entropy in future?), we might suppress.
            # For this demo, we just flag that we used imagination.
            imagination_used = True

        # 6. Decision
        probs = mx.softmax(logits)
        action_idx = mx.argmax(logits).item()
        
        # Create a one-hot action vector
        action_vector = mx.zeros((self.action_dim,))
        action_vector[action_idx] = 1.0
        
        entropy = -mx.sum(probs * mx.log(probs + 1e-9))
        
        decision = {
            "state": current_state,
            "logits": logits,
            "probs": probs,
            "action": action_idx,
            "action_vector": action_vector,
            "confidence": confidence,
            "entropy": entropy,
            "goal": new_goal,
            "memory_retrieved": not is_confident,
            "imagination_used": imagination_used
        }

        # 7. Plasticity (Online Learning)
        if reward != 0.0:
            self.learn(current_state, action_vector, reward)

        # Update internal state for next step
        self.last_state = current_state
        self.last_action = action_vector

        return decision

    def learn(self, state: mx.array, action_taken: mx.array, reward: float):
        """
        Plasticity: Update System 1 weights based on reward.
        """
        def loss_fn(model, x, target_action, r):
            logits = model(x)
            ce_loss = nn.losses.cross_entropy(logits, target_action)
            return ce_loss * r

        loss, grads = mx.value_and_grad(loss_fn)(self.system1, state, action_taken, reward)
        self.optimizer.update(self.system1, grads)

    def dream(self, batch_size: int = 32, epochs: int = 5):
        """
        Offline Consolidation ('Dreaming').
        Uses the GPU to batch-train the World Model and System 1 on memories.
        """
        if len(self.memory.memories) < batch_size:
            print("Not enough memories to dream yet.")
            return

        print(f"💤 Dreaming (Consolidating {len(self.memory.memories)} memories)...")
        
        import random
        
        # Convert memories to batch tensors (Unified Memory / GPU)
        # We sample a batch
        batch = random.sample(self.memory.memories, min(len(self.memory.memories), batch_size * 10))
        
        # Filter valid transitions
        valid_batch = [m for m in batch if m['next_state'] is not None]
        if not valid_batch:
            return

        states = mx.array([m['state'] for m in valid_batch])
        actions = mx.array([m['action'] for m in valid_batch])
        next_states = mx.array([m['next_state'] for m in valid_batch])
        rewards = mx.array([m['reward'] for m in valid_batch])

        # 1. Train World Model: Minimize MSE(Predicted_Next_State, Actual_Next_State)
        def wm_loss_fn(model, s, a, s_next):
            pred_s_next = model(s, a)
            return mx.mean((pred_s_next - s_next) ** 2)

        wm_loss_and_grad = mx.value_and_grad(wm_loss_fn)

        for _ in range(epochs):
            loss, grads = wm_loss_and_grad(self.world_model, states, actions, next_states)
            self.wm_optimizer.update(self.world_model, grads)
            
        print(f"✨ Dream Cycle Complete. World Model MSE: {loss.item():.4f}")

