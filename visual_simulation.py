"""
Visual Simulation of Heterogeneous Agent (System 1 vs System 2)

This script creates a 2D visual environment to demonstrate the "Fast vs Slow" thinking
architecture of the HeterogeneousAgent.

Visualization:
- Agent moves in a grid world collecting resources (Green).
- Avoids traps (Red).
- System 1 (NPU): Agent moves instantly (Blue).
- System 2 (GPU): Agent pauses to "think" (Orange) when entropy is high.
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from heterogeneous_architecture import HeterogeneousAgent

class SimpleGridWorld:
    def __init__(self, size=20, num_resources=5, num_traps=3):
        self.size = size
        self.agent_pos = np.array([size // 2, size // 2])
        self.resources = []
        self.traps = []
        self.num_resources = num_resources
        self.num_traps = num_traps
        self.reset_objects()
        
    def reset_objects(self):
        # Spawn resources
        self.resources = []
        for _ in range(self.num_resources):
            while True:
                pos = np.random.randint(0, self.size, 2)
                if not np.array_equal(pos, self.agent_pos) and \
                   not any(np.array_equal(pos, r) for r in self.resources):
                    self.resources.append(pos)
                    break
                    
        # Spawn traps
        self.traps = []
        for _ in range(self.num_traps):
            while True:
                pos = np.random.randint(0, self.size, 2)
                if not np.array_equal(pos, self.agent_pos) and \
                   not any(np.array_equal(pos, r) for r in self.resources) and \
                   not any(np.array_equal(pos, t) for t in self.traps):
                    self.traps.append(pos)
                    break

    def get_observation(self):
        # Create a 128D vector representing the local view
        # This is a simplified "retina"
        obs = np.zeros(128)
        
        # Encode position (normalized)
        obs[0] = self.agent_pos[0] / self.size
        obs[1] = self.agent_pos[1] / self.size
        
        # Scan 5x5 area around agent
        view_radius = 2
        idx = 2
        for dy in range(-view_radius, view_radius + 1):
            for dx in range(-view_radius, view_radius + 1):
                if idx >= 128: break
                
                check_pos = self.agent_pos + np.array([dx, dy])
                
                # Check bounds
                if 0 <= check_pos[0] < self.size and 0 <= check_pos[1] < self.size:
                    # Check for objects
                    is_resource = any(np.array_equal(check_pos, r) for r in self.resources)
                    is_trap = any(np.array_equal(check_pos, t) for t in self.traps)
                    
                    if is_resource:
                        obs[idx] = 1.0 # Positive signal
                    elif is_trap:
                        obs[idx] = -1.0 # Negative signal
                else:
                    obs[idx] = -0.5 # Wall
                
                idx += 1
                
        # Add some noise to simulate sensor imperfection
        obs[idx:] = np.random.normal(0, 0.1, 128 - idx)
        return mx.array(obs.astype(np.float32))

    def step(self, action):
        # Actions: 0=Up, 1=Right, 2=Down, 3=Left
        move = np.array([0, 0])
        if action == 0: move = np.array([0, 1])
        elif action == 1: move = np.array([1, 0])
        elif action == 2: move = np.array([0, -1])
        elif action == 3: move = np.array([-1, 0])
        
        new_pos = self.agent_pos + move
        new_pos = np.clip(new_pos, 0, self.size - 1)
        
        reward = -0.01 # Step cost
        
        # Check collisions
        hit_resource = -1
        for i, r in enumerate(self.resources):
            if np.array_equal(new_pos, r):
                reward = 1.0
                hit_resource = i
                break
                
        hit_trap = False
        for t in self.traps:
            if np.array_equal(new_pos, t):
                reward = -1.0
                hit_trap = True
                break
        
        self.agent_pos = new_pos
        
        if hit_resource != -1:
            # Respawn resource
            self.resources.pop(hit_resource)
            while True:
                pos = np.random.randint(0, self.size, 2)
                if not np.array_equal(pos, self.agent_pos) and \
                   not any(np.array_equal(pos, r) for r in self.resources) and \
                   not any(np.array_equal(pos, t) for t in self.traps):
                    self.resources.append(pos)
                    break
                    
        return reward

def run_visual_simulation():
    print("Initializing Visual Simulation...")
    
    # Initialize Environment and Agent
    env = SimpleGridWorld(size=15, num_resources=8, num_traps=5)
    agent = HeterogeneousAgent(use_quantization=True)
    
    # Setup Plot
    plt.ion() # Interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Metrics history
    entropy_history = []
    system2_usage_history = []
    
    print("\nStarting Simulation Loop...")
    print("Close the window to stop.")
    
    step = 0
    try:
        while plt.fignum_exists(fig.number):
            step += 1
            
            # 1. Observe
            obs = env.get_observation()
            
            # 2. Act (Agent decides)
            # We map the agent's 10 outputs to 4 directions
            decision = agent.step(obs)
            raw_action = decision['action']
            action = raw_action % 4 # Modulo to map to 4 directions
            
            used_system2 = decision['used_system2']
            entropy = decision['entropy']
            
            # 3. Step Environment
            reward = env.step(action)
            
            # 4. Learn (Online Buffer)
            # In a real loop, we'd call sleep() occasionally
            if step % 50 == 0:
                print(f"   💤 Auto-Sleep triggered at step {step}...")
                agent.sleep(epochs=2)
            
            # --- Visualization ---
            ax1.clear()
            ax2.clear()
            
            # Draw Grid World
            ax1.set_title(f"Step: {step} | Reward: {reward:.2f}")
            ax1.set_xlim(-1, env.size)
            ax1.set_ylim(-1, env.size)
            ax1.grid(True, alpha=0.3)
            
            # Draw Resources (Green)
            for r in env.resources:
                ax1.plot(r[0], r[1], 'g*', markersize=15, label='Resource')
                
            # Draw Traps (Red)
            for t in env.traps:
                ax1.plot(t[0], t[1], 'rx', markersize=15, label='Trap')
                
            # Draw Agent
            # Color depends on System used
            agent_color = 'orange' if used_system2 else 'blue'
            agent_label = 'System 2 (GPU)' if used_system2 else 'System 1 (NPU)'
            
            ax1.plot(env.agent_pos[0], env.agent_pos[1], 'o', color=agent_color, markersize=12, label=agent_label)
            
            # Add legend (deduplicated)
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            # Draw Metrics
            entropy_history.append(entropy)
            system2_usage_history.append(1.0 if used_system2 else 0.0)
            if len(entropy_history) > 50:
                entropy_history.pop(0)
                system2_usage_history.pop(0)
                
            ax2.set_title("Cognitive State")
            ax2.set_ylim(0, 2.5) # Entropy scale
            ax2.plot(entropy_history, label='Entropy (Uncertainty)', color='purple')
            ax2.fill_between(range(len(system2_usage_history)), 0, 2.5, 
                             where=[u > 0 for u in system2_usage_history], 
                             color='orange', alpha=0.3, label='System 2 Active')
            ax2.legend(loc='upper left')
            
            plt.draw()
            plt.pause(0.1) # Animation speed
            
            # Simulate "Thinking Time" if System 2 was used
            if used_system2:
                time.sleep(0.5) # Artificial delay to visualize "slow thinking"
                
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_visual_simulation()
