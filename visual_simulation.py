"""
Visual Simulation of Heterogeneous Agent (System 1 vs System 2)

PURPOSE: Demonstrate cognitive architecture mechanisms
- Visualizes System 1 (fast) vs System 2 (slow) switching
- Shows entropy/uncertainty dynamics in real-time
- Demonstrates workspace integration and decision-making process
- VISUALIZES MEMORY CONSOLIDATION (Hippocampus -> Neocortex)

Features:
- Dual-process visualization (Blue/Orange agent)
- Real-time cognitive state graphs
- Memory Trail (Hippocampus visualization)
- Sleep Replay (Consolidation visualization)
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import numpy as np
import mlx.core as mx
import time
import argparse
import os

# Import the architecture
from heterogeneous_architecture import HeterogeneousAgent

class SimpleGridWorld:
    def __init__(self, size=15, num_resources=8, num_traps=5):
        self.size = size
        self.agent_pos = [size//2, size//2]
        self.resources = []
        self.traps = []
        self.num_resources = num_resources
        self.num_traps = num_traps
        self.reset_objects()
        
    def reset_objects(self):
        """Place objects randomly, avoiding center."""
        self.resources = []
        self.traps = []
        
        # Helper to get random pos
        def get_pos():
            while True:
                pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
                if pos == [self.size//2, self.size//2]: continue
                if pos in self.resources or pos in self.traps: continue
                return pos

        for _ in range(self.num_resources):
            self.resources.append(get_pos())
        for _ in range(self.num_traps):
            self.traps.append(get_pos())

    def get_observation(self, noise_level=0.3):
        """
        Generate noisy sensory input (128-dim vector).
        Includes:
        - Normalized position (2 dims)
        - Local view 5x5 flattened (25 dims)
        - Gradient sensors (4 dims)
        - Random noise (rest)
        """
        obs = np.zeros(128)
        
        # 1. Proprioception (Position)
        obs[0] = self.agent_pos[0] / self.size
        obs[1] = self.agent_pos[1] / self.size
        
        # 2. Local View (5x5 grid around agent)
        view_idx = 2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = self.agent_pos[0] + dx, self.agent_pos[1] + dy
                val = 0.0
                if not (0 <= px < self.size and 0 <= py < self.size):
                    val = -0.5 # Wall
                elif [px, py] in self.resources:
                    val = 1.0  # Resource
                elif [px, py] in self.traps:
                    val = -1.0 # Trap
                
                # Add sensor noise
                if np.random.random() < noise_level:
                    val += np.random.normal(0, 0.2)
                    
                obs[view_idx] = val
                view_idx += 1
        
        # 3. Gradient Sensors (Smell/Heat)
        # Find nearest resource
        min_dist = float('inf')
        nearest_res = None
        for r in self.resources:
            d = abs(r[0] - self.agent_pos[0]) + abs(r[1] - self.agent_pos[1])
            if d < min_dist:
                min_dist = d
                nearest_res = r
        
        if nearest_res:
            # Direction vector to resource
            dx = nearest_res[0] - self.agent_pos[0]
            dy = nearest_res[1] - self.agent_pos[1]
            obs[27] = np.sign(dx)
            obs[28] = np.sign(dy)
            obs[29] = 1.0 / (min_dist + 1) # Proximity intensity
        
        # 4. Fill rest with noise (simulating brain background activity)
        obs[30:] = np.random.normal(0, 0.1, 128-30)
        
        return mx.array(obs, dtype=mx.float32)

    def step(self, action):
        """
        Action map: 0=Up, 1=Right, 2=Down, 3=Left
        Returns: reward, done
        """
        # Map 10 outputs to 4 directions
        move = action % 4
        
        dx, dy = 0, 0
        if move == 0: dy = 1   # Up
        elif move == 1: dx = 1 # Right
        elif move == 2: dy = -1 # Down
        elif move == 3: dx = -1 # Left
        
        new_x = max(0, min(self.size-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size-1, self.agent_pos[1] + dy))
        
        self.agent_pos = [new_x, new_y]
        
        reward = -0.01 # Step cost
        
        if self.agent_pos in self.resources:
            reward = 1.0
            self.resources.remove(self.agent_pos)
            # Respawn resource elsewhere
            while True:
                pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
                if pos not in self.resources and pos not in self.traps and pos != self.agent_pos:
                    self.resources.append(pos)
                    break
        
        elif self.agent_pos in self.traps:
            reward = -1.0
            # Traps stay
            
        return reward

def run_visual_simulation(use_pretrained=True, enable_learning=True):
    print("🚀 Starting Visual Simulation")
    print(f"   Mode: {'Pretrained' if use_pretrained else 'Scratch'}")
    print(f"   Learning: {'ENABLED' if enable_learning else 'DISABLED'}")
    
    # Initialize Environment
    env = SimpleGridWorld()
    
    # Initialize Agent
    agent = HeterogeneousAgent(use_quantization=True)
    
    # Load brain if requested
    if use_pretrained and os.path.exists("agent_brain.npz"):
        agent.load_brain("agent_brain.npz")
    
    # Setup Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])
    
    # Panel 1: Grid World
    ax_grid = fig.add_subplot(gs[:, 0])
    ax_grid.set_title("Heterogeneous Agent (System 1 vs System 2)")
    ax_grid.set_xlim(-1, env.size)
    ax_grid.set_ylim(-1, env.size)
    ax_grid.grid(True, alpha=0.2)
    
    # Grid Elements
    res_scatter = ax_grid.scatter([], [], c='lime', marker='*', s=200, label='Resource', zorder=2)
    trap_scatter = ax_grid.scatter([], [], c='red', marker='x', s=150, label='Trap', zorder=2)
    agent_dot = ax_grid.scatter([], [], c='blue', s=300, label='System 1 (NPU)', zorder=5)
    
    # 🧠 MEMORY VISUALIZATION ELEMENTS
    # Hippocampal memory trace (fading dots)
    memory_scatter = ax_grid.scatter([], [], c=[], s=100, alpha=0.8, marker='o', zorder=1)
    # Sleep replay lines (connections during consolidation)
    dream_lines, = ax_grid.plot([], [], c='gold', alpha=0.6, linewidth=2, linestyle='--', zorder=1)
    
    # Legend (Moved outside)
    ax_grid.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    # Panel 2: Cognitive State (Entropy)
    ax_cog = fig.add_subplot(gs[0, 1])
    ax_cog.set_title("Cognitive State (Uncertainty)")
    ax_cog.set_ylim(0, 2.5)
    ax_cog.set_xlim(0, 50)
    ax_cog.grid(True, alpha=0.2)
    
    line_entropy, = ax_cog.plot([], [], 'purple', linewidth=2, label='Entropy')
    fill_sys2 = ax_cog.fill_between([], [], color='orange', alpha=0.3, label='System 2 Active')
    ax_cog.set_ylabel("Entropy (bits)")
    ax_cog.legend(loc='upper right')
    
    # Panel 3: Internal Dynamics
    ax_internal = fig.add_subplot(gs[1, 1])
    ax_internal.set_title("Internal Dynamics (Workspace & Confidence)")
    ax_internal.set_ylim(0, 1.1)
    ax_internal.set_xlim(0, 50)
    ax_internal.grid(True, alpha=0.2)
    
    line_workspace, = ax_internal.plot([], [], 'cyan', linewidth=1.5, label='Workspace Activity')
    line_confidence, = ax_internal.plot([], [], 'lime', linewidth=1.5, label='Confidence')
    fill_confidence = ax_internal.fill_between([], [], color='lime', alpha=0.1)
    ax_internal.set_xlabel("Time (Recent Steps)")
    ax_internal.set_ylabel("Activation Level")
    ax_internal.legend(loc='upper right')
    
    # Text Info
    info_text = ax_internal.text(0.02, 0.95, "", transform=ax_internal.transAxes, 
                                color='white', fontsize=10, verticalalignment='top',
                                bbox=dict(facecolor='black', alpha=0.5))

    # Data Storage
    history = {
        'entropy': [],
        'sys2': [],
        'workspace': [],
        'confidence': []
    }
    
    # 🧠 MEMORY STORAGE
    # List of dicts: {'pos': [x,y], 'type': 'good'/'bad', 'alpha': 1.0}
    memories = [] 
    
    state = {'step': 0, 'total_reward': 0.0, 'last_pos': None, 'stuck_count': 0}

    def init():
        return agent_dot, res_scatter, trap_scatter, memory_scatter, dream_lines, line_entropy, fill_sys2, line_workspace, line_confidence, fill_confidence, info_text

    def update(frame):
        # 1. Get Observation
        obs = env.get_observation(noise_level=0.3) # Add noise to make it interesting
        
        # Check if stuck (Agent hasn't moved for 10 frames)
        current_pos = list(env.agent_pos)
        if state['last_pos'] == current_pos:
            state['stuck_count'] += 1
        else:
            state['stuck_count'] = 0
        state['last_pos'] = current_pos
        
        force_random = False
        if state['stuck_count'] > 10:
            force_random = True
        
        # 2. Agent Step
        result = agent.step(obs)
        action = result['action']
        entropy = result['entropy']
        used_sys2 = result['used_system2']
        probs = result['action_probs']
        
        # Safety: Check for NaNs
        if np.isnan(probs).any():
            force_random = True
            
        # Force exploration if stuck
        if force_random:
            action = np.random.randint(0, 4)
            # Update the buffer so we learn from the forced move (correct off-policy data)
            if enable_learning and agent.online_buffer:
                agent.online_buffer[-1]['action'] = action
        
        # 3. Environment Step
        reward = env.step(action)
        
        # 4. Learning (Online)
        if enable_learning:
            # Add reward to last buffer entry
            if agent.online_buffer:
                agent.online_buffer[-1]['reward'] = reward
        
        state['step'] += 1
        state['total_reward'] += reward
        
        # 🧠 UPDATE MEMORY TRAIL (Hippocampus)
        # If significant event, add to memory
        if reward == 1.0:
            memories.append({'pos': list(env.agent_pos), 'color': [0, 1, 0], 'alpha': 1.0}) # Green
        elif reward == -1.0:
            memories.append({'pos': list(env.agent_pos), 'color': [1, 0, 0], 'alpha': 1.0}) # Red
            
        # Decay memories (forgetting curve)
        active_memories = []
        colors = []
        offsets = []
        
        for m in memories:
            m['alpha'] -= 0.005 # Slow decay
            if m['alpha'] > 0.05:
                active_memories.append(m)
                # Matplotlib expects RGBA
                c = m['color'] + [m['alpha']]
                colors.append(c)
                offsets.append(m['pos'])
        
        memories[:] = active_memories # Update list in place
        
        # Update Memory Scatter Plot
        if offsets:
            memory_scatter.set_offsets(offsets)
            memory_scatter.set_facecolors(colors)
        else:
            memory_scatter.set_offsets(np.zeros((0, 2)))

        # 💤 SLEEP & CONSOLIDATION LOGIC
        is_sleeping = False
        if enable_learning and state['step'] % 50 == 0:
            is_sleeping = True
            ax_grid.set_title(f"💤 SLEEPING... Replaying {len(memories)} Memories...")
            
            # VISUALIZE REPLAY: Connect memories with gold line
            if len(offsets) > 1:
                pts = np.array(offsets)
                dream_lines.set_data(pts[:, 0], pts[:, 1])
            
            # Force draw to show the dream lines
            plt.draw()
            fig.canvas.flush_events()
            
            # Perform Sleep (Consolidation)
            stats = agent.sleep(epochs=15) # Moderate sleep for demo
            
            # Clear memories (transferred to cortex)
            memories.clear()
            memory_scatter.set_offsets(np.zeros((0, 2)))
            dream_lines.set_data([], [])
            
        else:
            # Normal Title
            mode = "🧠 SYS2 (GPU)" if used_sys2 else "⚡ SYS1 (NPU)"
            ax_grid.set_title(f"Step: {state['step']} | Reward: {reward:.2f} | Total: {state['total_reward']:.1f} | {mode}")

        # Update Visuals
        agent_dot.set_offsets([env.agent_pos])
        agent_dot.set_color('orange' if used_sys2 else 'blue')
        agent_dot.set_label('System 2 (GPU)' if used_sys2 else 'System 1 (NPU)')
        
        res_scatter.set_offsets(env.resources)
        trap_scatter.set_offsets(env.traps)
        
        # Update Graphs (Rolling window)
        history['entropy'].append(entropy)
        history['sys2'].append(1 if used_sys2 else 0)
        history['workspace'].append(mx.mean(mx.abs(agent.workspace)).item())
        history['confidence'].append(mx.max(probs).item())
        
        if len(history['entropy']) > 50:
            for k in history: history[k].pop(0)
            
        x_data = range(len(history['entropy']))
        
        # Panel 2
        line_entropy.set_data(x_data, history['entropy'])
        
        # Fill System 2 regions
        for c in ax_cog.collections: c.remove() # Clear old fills
        fill_sys2 = ax_cog.fill_between(x_data, 0, 2.5, where=[s==1 for s in history['sys2']], 
                                       color='orange', alpha=0.3)
        
        # Panel 3
        line_workspace.set_data(x_data, history['workspace'])
        line_confidence.set_data(x_data, history['confidence'])
        
        for c in ax_internal.collections: c.remove()
        fill_confidence = ax_internal.fill_between(x_data, 0, history['confidence'], color='lime', alpha=0.1)
        
        # Update Text
        p = probs.tolist()
        # Handle case where model has fewer outputs than 4 (e.g. binary classification model)
        if len(p) < 4:
            p = p + [0.0] * (4 - len(p))
            
        txt = f"Buffer: {len(agent.online_buffer)}/{agent.max_online_buffer}\n"
        txt += f"Sleep Cycles: {agent.sleep_cycles}\n"
        txt += "Action Probs:\n"
        txt += f" ↑ Up:    {p[0]:.2f}\n"
        txt += f" → Right: {p[1]:.2f}\n"
        txt += f" ↓ Down:  {p[2]:.2f}\n"
        txt += f" ← Left:  {p[3]:.2f}"
        info_text.set_text(txt)
        
        return agent_dot, res_scatter, trap_scatter, memory_scatter, dream_lines, line_entropy, fill_sys2, line_workspace, line_confidence, fill_confidence, info_text

    # Run Animation
    anim = FuncAnimation(fig, update, init_func=init, frames=None, interval=20, blit=False)
    
    plt.tight_layout()
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if enable_learning:
            agent.save_brain("agent_brain.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", action="store_true", help="Start from scratch (random weights)")
    parser.add_argument("--no-learn", action="store_true", help="Disable learning (inference only)")
    args = parser.parse_args()
    
    run_visual_simulation(
        use_pretrained=not args.scratch,
        enable_learning=not args.no_learn
    )
