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

    def add_episode(self, state: mx.array, action: mx.array, reward: float, next_state: mx.array, surprise: float = 0.0):
        """
        Encodes and stores an episode.
        """
        episode = {
            "timestamp": time.time(),
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": reward,
            "next_state": next_state.tolist() if next_state is not None else None,
            "surprise": surprise  # Store surprise for Prioritized Replay
        }
        
        # Check for dimension mismatch BEFORE appending
        if self.memory_matrix is not None and self.memory_matrix.shape[1] != state.shape[0]:
            print(f"⚠️  Memory dimension mismatch ({self.memory_matrix.shape[1]} -> {state.shape[0]}). Clearing old memories.")
            self.memories = []
            self.memory_matrix = None
        
        # Now append consistently
        self.memories.append(episode)
        
        # Update GPU memory matrix efficiently
        if self.memory_matrix is None:
            self.memory_matrix = state[None, :]
        else:
            self.memory_matrix = mx.concatenate([self.memory_matrix, state[None, :]], axis=0)

        # Auto-save every 1000 memories to avoid disk thrashing
        if len(self.memories) % 1000 == 0:
            self.save_memories()

    def retrieve(self, query_state: mx.array, k: int = 3):
        """
        Retrieves the top-k most similar memories to the query_state.
        Uses vectorized GPU operations for speed.
        """
        if not self.memories or self.memory_matrix is None:
            return []
        
        # Ensure consistency between memories list and matrix
        actual_memory_count = len(self.memories)
        matrix_rows = self.memory_matrix.shape[0]
        
        if actual_memory_count != matrix_rows:
            print(f"⚠️  Memory sync issue: {actual_memory_count} memories but {matrix_rows} matrix rows. Rebuilding matrix...")
            self.memory_matrix = mx.stack([mx.array(m['state']) for m in self.memories])
        
        # Limit k to available memories
        k = min(k, len(self.memories))
        if k == 0:
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
    Enhanced with Hebbian Fast Weights for one-shot learning.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)
        
        # Hebbian Fast Weights (Hidden -> Output)
        # These run in parallel to l2 and update instantly
        self.hebbian_weights = mx.zeros((output_dim, hidden_dim))
        self.hebb_decay = 0.99  # Decay factor per step (Standard)
        self.hebb_lr = 0.1      # Lower LR to prevent "obsession" with one action

    def __call__(self, x):
        h = nn.relu(self.l1(x))
        
        # Slow path (Standard Weights)
        logits_slow = self.l2(h)
        
        # Fast path (Hebbian Weights)
        # (output, hidden) @ (hidden,) -> (output,)
        logits_fast = self.hebbian_weights @ h
        
        logits = logits_slow + logits_fast
        
        # Clamp logits to prevent exploding gradients/confidence
        # This keeps the softmax distribution from becoming a hard one-hot vector
        logits = mx.clip(logits, -10.0, 10.0)
        
        return logits, h

    def update_hebbian(self, h, action_vector, reward):
        """
        Apply Modulated Hebbian Update:
        Delta W = lr * reward * (action_vector * h.T)
        """
        # Normalize hidden state to keep weights stable
        h_norm = h / (mx.linalg.norm(h) + 1e-9)
        
        # Outer product: (Action) * (Context)
        # We reinforce the connection between this context and this action
        # scaled by the reward.
        delta = self.hebb_lr * reward * (mx.expand_dims(action_vector, -1) @ mx.expand_dims(h_norm, 0))
        
        # Dynamic Decay: Pain-Induced Forgetting
        # If we are punished significantly (reward < -1.0), we should "flush" our short-term memory
        # to allow new rules to take over immediately.
        # We use a threshold of -1.0 to avoid triggering on small penalties like Living Cost (-0.01) or Walls (-0.1).
        current_decay = self.hebb_decay
        if reward < -1.0:
            current_decay = 0.5 # Drastic forgetfulness on error
            
        # Apply update with decay
        self.hebbian_weights = (self.hebbian_weights * current_decay) + delta

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

    def __call__(self, x):
        x = mx.tanh(self.l1(x))
        
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

    def __call__(self, state, action):
        # Concatenate State + Action
        x = mx.concatenate([state, action], axis=-1)
        x = nn.relu(self.l1(x))
        x = nn.relu(self.l2(x))
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
        self.optimizer = optim.Adam(learning_rate=0.05) 
        self.sys2_optimizer = optim.Adam(learning_rate=0.05) # Fast adaptation for confidence too
        self.wm_optimizer = optim.Adam(learning_rate=0.01)

        # State tracking for transition storage (s_t, a_t, r_t, s_{t+1})
        self.last_state = None
        self.last_action = None
        self.last_hidden = None
        
        # Reality Check: Track average reward to detect failure
        self.avg_reward = 0.0

    def step(self, sensory_input: mx.array, reward: float = 0.0):
        """
        One cognitive cycle.
        """
        # Update moving average of reward (Reality Check)
        # If this stays negative, we are failing and should panic/explore
        self.avg_reward = 0.95 * self.avg_reward + 0.05 * reward

        # 1. Perception: Update Global Workspace
        current_state = self.workspace.update(sensory_input)

        # --- Intrinsic Motivation (Curiosity) ---
        intrinsic_reward = 0.0
        surprise = 0.0
        total_reward = reward

        if self.last_state is not None and self.last_action is not None:
            # Predict what we expected to see (World Model)
            predicted_next_state = self.world_model(self.last_state, self.last_action)
            
            # Calculate Surprise (MSE between expectation and reality)
            surprise = mx.mean((predicted_next_state - current_state) ** 2).item()
            
            # Intrinsic Reward: The agent gets a dopamine hit for learning something new
            # Scale factor 5.0 makes curiosity a strong driver
            intrinsic_reward = surprise * 5.0 
            total_reward = reward + intrinsic_reward

            # 2. Store Experience (S_prev, A_prev, R_total, S_curr)
            # Pass 'surprise' to memory for Prioritized Replay
            self.memory.add_episode(self.last_state, self.last_action, total_reward, current_state, surprise=surprise)

            # 7. Plasticity (Online Learning) - Moved here to reinforce the TRANSITION
            # We learn from the action that *caused* this state/reward
            # ALWAYS learn if there is a reward OR if there is high surprise (intrinsic reward)
            if abs(total_reward) > 0.01:
                self.learn(self.last_state, self.last_action, total_reward)
                
                # --- Hebbian Update (Fast Weights) ---
                # If we have the hidden state from the previous step, update the fast weights
                if self.last_hidden is not None:
                    self.system1.update_hebbian(self.last_hidden, self.last_action, total_reward)

        # 3. Intuition: System 1 proposes action
        logits, hidden_state = self.system1(current_state)
        
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
            if memories:
                memory_boost = 0.0
                for mem in memories:
                    past_action = mem['action']
                    past_reward = mem['reward']
                    
                    # Extract action index from list
                    if isinstance(past_action, list):
                        action_idx = past_action[0] if len(past_action) > 0 else 0
                    else:
                        action_idx = int(past_action)
                    
                    # Only boost current top action if it matches past success
                    if past_reward > 0:
                        memory_boost += past_reward * 0.1
                
                # Apply uniform boost to encourage consistency
                logits = logits + memory_boost

            # Strategy B: Imagination (Slow, uses World Model)
            # If still uncertain or if no memories, try to simulate future
            # We simulate the outcome of the top action candidate
            top_action_idx = mx.argmax(logits).item()
            top_action_vec = mx.zeros((self.action_dim,))
            top_action_vec[top_action_idx] = 1.0
            
            _ = self.world_model(current_state, top_action_vec)
            # If predicted state is "bad" (e.g. high entropy in future?), we might suppress.
            # For this demo, we just flag that we used imagination.
            imagination_used = True

        # 6. Decision
        # Apply temperature to softmax to encourage exploration if confidence is low
        # OR if we are consistently failing (Reality Check)
        temperature = 1.0
        if not is_confident or self.avg_reward < 0.0:
            temperature = 2.0 # Higher temperature = flatter distribution = more exploration
            
        probs = mx.softmax(logits / temperature)
        
        # Use categorical sampling for exploration
        # Note: mx.random.categorical expects logits, not probs
        action_idx = mx.random.categorical(logits / temperature).item()
        
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
            "imagination_used": imagination_used,
            "surprise": surprise,
            "intrinsic_reward": intrinsic_reward
        }

        # Update internal state for next step
        self.last_state = current_state
        self.last_action = action_vector
        self.last_hidden = hidden_state

        return decision

    def learn(self, state: mx.array, action_taken: mx.array, reward: float):
        """
        Plasticity: Update System 1 weights based on reward.
        Also updates System 2 confidence prediction.
        """
        # 1. Train System 1 (Policy Gradient)
        def loss_fn(model, x, target_action, r):
            logits, _ = model(x)
            ce_loss = nn.losses.cross_entropy(logits, target_action)
            return ce_loss * r
            
        _, grads = mx.value_and_grad(loss_fn)(self.system1, state, action_taken, reward)
        self.optimizer.update(self.system1, grads)

        # 2. Train System 2 (Confidence Prediction)
        # Target: 1.0 if reward > 0.5 (Success), 0.0 otherwise (Failure/Neutral)
        # We ignore small shaping rewards (0.2) for confidence to prevent arrogance
        target_confidence = mx.array([1.0 if reward > 0.5 else 0.0])
        
        def sys2_loss_fn(model, s, a, target):
            # Reconstruct input: State + Logits (we need logits from Sys1)
            # Note: We detach Sys1 logits here to not backprop into Sys1 from Sys2 loss
            logits, _ = self.system1(s)
            logits = mx.stop_gradient(logits)
            
            sys2_input = mx.concatenate([s, logits], axis=0)
            confidence, _ = model(sys2_input)
            
            # MSE Loss for confidence
            return mx.mean((confidence - target) ** 2)

        _, sys2_grads = mx.value_and_grad(sys2_loss_fn)(self.system2, state, action_taken, target_confidence)
        self.sys2_optimizer.update(self.system2, sys2_grads)

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
        import numpy as np
        
        # --- Prioritized Experience Replay ---
        # Calculate sampling probabilities based on 'surprise'
        # We add a small epsilon to ensure even low-surprise memories have a chance
        surprises = [m.get('surprise', 0.0) + 1e-5 for m in self.memory.memories]
        total_surprise = sum(surprises)
        probs = [s / total_surprise for s in surprises]
        
        # Sample indices based on probability distribution
        indices = np.random.choice(len(self.memory.memories), size=min(len(self.memory.memories), batch_size * 10), p=probs, replace=True)
        batch = [self.memory.memories[i] for i in indices]
        
        # Filter valid transitions
        valid_batch = [m for m in batch if m['next_state'] is not None]
        if not valid_batch:
            return

        states = mx.array([m['state'] for m in valid_batch])
        
        # Normalize actions - convert to one-hot vectors
        action_list = []
        for m in valid_batch:
            action = m['action']
            # Extract scalar action index
            if isinstance(action, list):
                action_idx = int(action[0]) if action else 0
            else:
                action_idx = int(action)
            
            # Create one-hot vector
            one_hot = [0.0] * self.action_dim
            one_hot[action_idx] = 1.0
            action_list.append(one_hot)
        
        actions = mx.array(action_list)  # Shape: (N, action_dim)
        next_states = mx.array([m['next_state'] for m in valid_batch])
        # rewards = mx.array([m['reward'] for m in valid_batch]) # Unused

        # 1. Train World Model: Minimize MSE(Predicted_Next_State, Actual_Next_State)
        def wm_loss_fn(model, s, a, s_next):
            pred_s_next = model(s, a)
            return mx.mean((pred_s_next - s_next) ** 2)

        wm_loss_and_grad = mx.value_and_grad(wm_loss_fn)

        loss = mx.array(0.0) # Initialize loss
        for _ in range(epochs):
            loss, grads = wm_loss_and_grad(self.world_model, states, actions, next_states)
            self.wm_optimizer.update(self.world_model, grads)
            
        print(f"✨ Dream Cycle Complete. World Model MSE: {loss.item():.4f}")

    def save_brain(self, path: str):
        """Save agent weights (System 1, System 2, World Model) to disk."""
        flat_params = {}
        
        def flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    flat_params[key] = v
                    
        # Save all components
        flatten(self.system1.parameters(), "system1")
        flatten(self.system2.parameters(), "system2")
        flatten(self.world_model.parameters(), "world_model")
        
        mx.savez(path, **flat_params)
        print(f"💾 Saved Bicameral Brain to {path}")

    def load_brain(self, path: str):
        """Load agent weights from disk."""
        try:
            weights = mx.load(path)
            
            # Helper to reconstruct nested dict for a specific prefix
            def unflatten(prefix):
                nested = {}
                for k, v in weights.items():
                    if k.startswith(prefix):
                        # Remove prefix
                        key_parts = k[len(prefix)+1:].split('.')
                        d = nested
                        for part in key_parts[:-1]:
                            if part not in d:
                                d[part] = {}
                            d = d[part]
                        d[key_parts[-1]] = v
                return nested

            # Load components
            self.system1.update(unflatten("system1"))
            self.system2.update(unflatten("system2"))
            self.world_model.update(unflatten("world_model"))
            
            print(f"🧠 Loaded Bicameral Brain from {path}")
            return True
        except Exception as e:
            print(f"⚠️ Could not load brain: {e}")
            return False

