"""
Visual Simulation of Conscious Agent (Bicameral Architecture)

PURPOSE: Demonstrate advanced cognitive mechanisms
- Visualizes System 1 (Fast) vs System 2 (Slow) switching
- Demonstrates "Hot Stove" Effect (Hebbian Fast Weights)
- Shows Intrinsic Motivation (Curiosity/Surprise)
- Visualizes Memory Consolidation (Hippocampus -> Neocortex)

Features:
- Dual-process visualization (Blue/Orange agent)
- Real-time cognitive state graphs (Entropy, Surprise)
- Rule Switching (Hot Stove Demo)
- Memory Trail & Sleep Replay
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
from cognitive_architecture import BicameralAgent

class SimpleGridWorld:
    def __init__(self, size=15, num_resources=8, num_traps=5):
        self.size = size
        self.agent_pos = [size//2, size//2]
        self.resources = []
        self.traps = []
        self.num_resources = num_resources
        self.num_traps = num_traps
        self.rule_phase = 0 # 0: Normal, 1: Inverted (Hot Stove)
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

    def switch_rules(self):
        """Flip the rules of the world!"""
        self.rule_phase = 1 - self.rule_phase
        return self.rule_phase

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
        
        # Determine values based on rule phase
        resource_val = 1.0 if self.rule_phase == 0 else -1.0
        trap_val = -1.0 if self.rule_phase == 0 else 1.0
        
        if self.agent_pos in self.resources:
            reward = resource_val
            if self.rule_phase == 0: # Only consume good things
                self.resources.remove(self.agent_pos)
                # Respawn resource elsewhere
                while True:
                    pos = [np.random.randint(0, self.size), np.random.randint(0, self.size)]
                    if pos not in self.resources and pos not in self.traps and pos != self.agent_pos:
                        self.resources.append(pos)
                        break
        
        elif self.agent_pos in self.traps:
            reward = trap_val
            # Traps stay
            
        return reward

def run_visual_simulation(use_pretrained=True, enable_learning=True):
    print("🚀 Starting Visual Simulation")
    print(f"   Mode: {'Pretrained' if use_pretrained else 'Scratch'}")
    print(f"   Learning: {'ENABLED' if enable_learning else 'DISABLED'}")
    
    # Initialize Environment
    env = SimpleGridWorld()
    
    # Initialize Agent (Bicameral)
    agent = BicameralAgent()
    
    # Load brain if requested
    if use_pretrained and os.path.exists("agent_brain.npz"):
        agent.load_brain("agent_brain.npz")
    
    # Setup Plot
    plt.style.use('default') # Switch to white background
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.5], height_ratios=[1, 1])
    
    # Panel 1: Grid World
    ax_grid = fig.add_subplot(gs[:, 0])
    ax_grid.set_title("Conscious Agent (System 1 vs System 2)")
    ax_grid.set_xlim(-1, env.size)
    ax_grid.set_ylim(-1, env.size)
    ax_grid.grid(True, alpha=0.2, color='black')
    
    # Grid Elements
    res_scatter = ax_grid.scatter([], [], c='green', marker='*', s=200, label='Resource', zorder=2)
    trap_scatter = ax_grid.scatter([], [], c='red', marker='x', s=150, label='Trap', zorder=2)
    agent_dot = ax_grid.scatter([], [], c='blue', s=300, label='System 1 (NPU)', zorder=5)
    
    # Dummy handles for legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='System 1 (Fast)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='System 2 (Slow)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=15, label='Resource'),
        Line2D([0], [0], marker='x', color='red', markersize=10, linestyle='None', markeredgewidth=2, label='Trap')
    ]
    
    # 🧠 MEMORY VISUALIZATION ELEMENTS
    # Hippocampal memory trace (fading dots)
    memory_scatter = ax_grid.scatter([], [], c=[], s=100, alpha=0.8, marker='o', zorder=1)
    # Sleep replay lines (connections during consolidation)
    dream_lines, = ax_grid.plot([], [], c='orange', alpha=0.6, linewidth=2, linestyle='--', zorder=1)
    
    # Legend (Moved outside)
    ax_grid.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    # Panel 2: Cognitive State (Entropy & Surprise)
    ax_cog = fig.add_subplot(gs[0, 1])
    ax_cog.set_title("Cognitive State: Entropy & Surprise")
    ax_cog.set_ylim(0, 3.0)
    ax_cog.set_xlim(0, 50)
    ax_cog.grid(True, alpha=0.2, color='black')
    
    # Add threshold line and background
    ax_cog.axhline(y=1.5, color='salmon', linestyle='--', alpha=0.8, label='Threshold (1.5)')
    
    line_entropy, = ax_cog.plot([], [], 'purple', linewidth=2, marker='.', label='Entropy')
    line_surprise, = ax_cog.plot([], [], 'red', linewidth=2, linestyle=':', label='Surprise')
    
    fill_sys2 = ax_cog.fill_between([], [], color='orange', alpha=0.0) # Hidden, using static background instead
    ax_cog.set_ylabel("Magnitude")
    ax_cog.set_xlabel("Steps (last 50)")
    ax_cog.legend(loc='upper left')
    
    # Panel 3: Internal Dynamics
    ax_internal = fig.add_subplot(gs[1, 1])
    ax_internal.set_title("Internal State Dynamics")
    ax_internal.set_ylim(0, 1.1)
    ax_internal.set_xlim(0, 50)
    ax_internal.grid(True, alpha=0.2, color='black')
    
    line_workspace, = ax_internal.plot([], [], 'cyan', linewidth=2, label='Workspace Activity')
    line_confidence, = ax_internal.plot([], [], 'lime', linewidth=2, label='Confidence')
    fill_confidence = ax_internal.fill_between([], [], color='lime', alpha=0.0) # Removed fill to match clean look
    ax_internal.set_xlabel("Time Steps (last 50)")
    ax_internal.set_ylabel("Activity / Confidence")
    ax_internal.legend(loc='upper right')
    
    # Text Info
    info_text = ax_internal.text(0.02, 0.98, "", transform=ax_internal.transAxes, 
                                color='black', fontsize=10, verticalalignment='top',
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Data Storage
    history = {
        'entropy': [],
        'surprise': [],
        'sys2': [],
        'workspace': [],
        'confidence': []
    }
    
    # 🧠 MEMORY STORAGE
    # List of dicts: {'pos': [x,y], 'type': 'good'/'bad', 'alpha': 1.0}
    memories = [] 
    
    # State tracking
    state = {
        'step': 0, 
        'total_reward': 0.0, 
        'last_pos': [-1, -1], # Initialize with impossible pos
        'stuck_count': 0,
        'pos_history': [] # For loop detection
    }

    def init():
        return agent_dot, res_scatter, trap_scatter, memory_scatter, dream_lines, line_entropy, line_surprise, fill_sys2, line_workspace, line_confidence, fill_confidence, info_text

    def update(frame):
        try:
            # 0. Rule Switching Logic (Hot Stove Demo)
            if state['step'] > 0 and state['step'] % 100 == 0:
                new_phase = env.switch_rules()
                # Visual cue for rule switch
                bg_color = '#ffebee' if new_phase == 1 else 'white' # Light red for danger phase
                ax_grid.set_facecolor(bg_color)
                print(f"Step {state['step']}: Rules Switched! Phase: {new_phase}")
                
            # 1. Get Observation
            obs = env.get_observation(noise_level=0.3) # Add noise to make it interesting
            
            # 2. Agent Step (Bicameral)
            # Note: BicameralAgent.step takes (sensory_input, reward)
            # We pass the reward from the PREVIOUS step (or 0 initially)
            
            # To make the loop work correctly with BicameralAgent's internal learning:
            # We need to pass the reward resulting from the *previous* action.
            # But for the very first step, reward is 0.
            # Let's store the last reward in state.
            last_reward = state.get('last_reward', 0.0)
            
            result = agent.step(obs, reward=last_reward)
            
            action = result['action']
            entropy = result['entropy'].item()
            confidence = result['confidence'].item()
            used_sys2 = not result['confidence'].item() > 0.5 # Low confidence = System 2
            probs = result['probs']
            surprise = result.get('surprise', 0.0)
            intrinsic_reward = result.get('intrinsic_reward', 0.0)
            
            # Safety: Check for NaNs
            # Convert to numpy first to be safe
            probs_np = np.array(probs)
            if np.isnan(probs_np).any():
                print("⚠️ NaN detected in action probs! Forcing random action.")
                action = np.random.randint(0, 4)

            # --- STUCK DETECTION & FORCED EXPLORATION ---
            # If agent hasn't moved for 5 steps, force a random move
            current_pos_list = list(env.agent_pos)
            if state['last_pos'] == current_pos_list:
                state['stuck_count'] += 1
            else:
                state['stuck_count'] = 0
                state['last_pos'] = current_pos_list
                
            if state['stuck_count'] > 5:
                action = np.random.randint(0, 4)
                # Visual indication of boredom/forcing
                agent_dot.set_edgecolor('red')
                agent_dot.set_linewidth(2)
            else:
                agent_dot.set_edgecolor('none')
            
            # 3. Environment Step
            reward = env.step(action)
            state['last_reward'] = reward # Store for next agent step
            
            state['step'] += 1
            state['total_reward'] += reward
            
            # 🧠 UPDATE MEMORY TRAIL (Hippocampus)
            # If significant event, add to memory
            if reward >= 1.0:
                memories.append({'pos': list(env.agent_pos), 'color': [0, 1, 0], 'alpha': 1.0}) # Green
            elif reward <= -1.0:
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
            if enable_learning and state['step'] % 150 == 0: # Sleep less often
                is_sleeping = True
                ax_grid.set_title(f"💤 SLEEPING... Dreaming of {len(agent.memory.memories)} memories...")
                
                # VISUALIZE REPLAY: Connect memories with gold line
                if len(offsets) > 1:
                    pts = np.array(offsets)
                    dream_lines.set_data(pts[:, 0], pts[:, 1])
                
                # Force draw to show the dream lines
                plt.draw()
                fig.canvas.flush_events()
                
                # Perform Sleep (Consolidation)
                agent.dream(epochs=5)
                
                # Clear visual memories (transferred to cortex)
                memories.clear()
                memory_scatter.set_offsets(np.zeros((0, 2)))
                dream_lines.set_data([], [])
                
            else:
                # Normal Title
                mode = "🧠 SYS2 (Deliberate)" if used_sys2 else "⚡ SYS1 (Intuitive)"
                phase_name = "NORMAL" if env.rule_phase == 0 else "INVERTED (HOT STOVE)"
                title_text = f"Step: {state['step']} | R: {reward:.1f} | {mode} | {phase_name}"
                if surprise > 0.1:
                    title_text += f" | 😲 SURPRISE: {surprise:.2f}"
                ax_grid.set_title(title_text)

            # Update Visuals
            agent_dot.set_offsets([env.agent_pos])
            agent_dot.set_color('orange' if used_sys2 else 'blue')
            # No label update needed as we use static legend
            
            res_scatter.set_offsets(env.resources)
            trap_scatter.set_offsets(env.traps)
            
            # Update Graphs (Rolling window)
            history['entropy'].append(entropy)
            history['surprise'].append(surprise)
            history['sys2'].append(1 if used_sys2 else 0)
            history['workspace'].append(mx.mean(mx.abs(agent.workspace.current_state)).item())
            history['confidence'].append(confidence)
            
            if len(history['entropy']) > 50:
                for k in history: history[k].pop(0)
                
            x_data = range(len(history['entropy']))
            
            # Panel 2
            line_entropy.set_data(x_data, history['entropy'])
            line_surprise.set_data(x_data, history['surprise'])
            
            # Fill System 2 regions
            for c in ax_cog.collections: c.remove() # Clear old fills
            fill_sys2 = ax_cog.fill_between(x_data, 0, 3.0, where=[s==1 for s in history['sys2']], 
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
                
            txt = f"Memories: {len(agent.memory.memories)}\n"
            txt += f"Intrinsic Reward: {intrinsic_reward:.2f}\n"
            txt += "Action Probs:\n"
            txt += f" ↑ Up:    {p[0]:.2f}\n"
            txt += f" → Right: {p[1]:.2f}\n"
            txt += f" ↓ Down:  {p[2]:.2f}\n"
            txt += f" ← Left:  {p[3]:.2f}"
            info_text.set_text(txt)
            
            return agent_dot, res_scatter, trap_scatter, memory_scatter, dream_lines, line_entropy, line_surprise, fill_sys2, line_workspace, line_confidence, fill_confidence, info_text
        except Exception as e:
            print(f"❌ Error in animation update: {e}")
            import traceback
            traceback.print_exc()
            return agent_dot,

    # Run Animation
    anim = FuncAnimation(fig, update, init_func=init, frames=None, interval=50, blit=False)
    
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
