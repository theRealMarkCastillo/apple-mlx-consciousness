"""
Phase 2.3: Structured Environment - Foraging Task

Collective foraging experiment where agents must:
1. Explore 2D environment to find resources
2. Communicate discoveries to the swarm
3. Collectively optimize resource gathering
4. Develop emergent specialization (scouts vs. gatherers)

Tests:
- Information propagation through swarm
- Consensus formation on exploration strategies
- Role differentiation (do specialists emerge?)
- Collective intelligence vs. individual intelligence
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from typing import List, Tuple, Dict
from swarm_architecture import ConsciousSwarm


class ForagingEnvironment:
    """
    2D grid world with resources to collect.
    
    Environment features:
    - Resources spawn randomly and regenerate over time
    - Agents have limited vision radius
    - Resources give reward when collected
    - Environment state encoded as 128D vector for agents
    """
    
    def __init__(self, 
                 grid_size: int = 50,
                 num_resources: int = 20,
                 vision_radius: int = 5,
                 regeneration_rate: float = 0.02):
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.vision_radius = vision_radius
        self.regeneration_rate = regeneration_rate
        
        # Agent positions (x, y)
        self.agent_positions = []
        
        # Resource locations and availability
        self.resources = []  # List of (x, y, available) tuples
        self._spawn_resources()
        
        # History
        self.total_collected = 0
        self.collection_history = []
        
        print(f"🌲 Foraging Environment Initialized")
        print(f"   Grid: {grid_size}×{grid_size}")
        print(f"   Resources: {num_resources}")
        print(f"   Vision radius: {vision_radius}")
    
    def _spawn_resources(self):
        """Spawn resources at random locations."""
        self.resources = [
            (np.random.randint(0, self.grid_size),
             np.random.randint(0, self.grid_size),
             True)  # available
            for _ in range(self.num_resources)
        ]
    
    def reset(self, num_agents: int):
        """Reset environment and randomly place agents."""
        self.agent_positions = [
            (np.random.randint(0, self.grid_size),
             np.random.randint(0, self.grid_size))
            for _ in range(num_agents)
        ]
        self._spawn_resources()
        self.total_collected = 0
        self.collection_history = []
    
    def get_observation(self, agent_idx: int) -> mx.array:
        """
        Generate 128D observation vector for an agent.
        
        Encoding:
        - [0:2]: Normalized agent position (x, y)
        - [2:4]: Normalized velocity hint (random for now)
        - [4:64]: Local resource map (8×8 grid around agent)
        - [64:128]: Global statistics (resource density, swarm centroid, etc.)
        """
        obs = np.zeros(128, dtype=np.float32)
        
        # Agent position
        x, y = self.agent_positions[agent_idx]
        obs[0] = x / self.grid_size
        obs[1] = y / self.grid_size
        
        # Velocity (random for now - agents don't have physics yet)
        obs[2:4] = np.random.randn(2) * 0.1
        
        # Local resource map (8×8 grid centered on agent)
        local_map = np.zeros((8, 8), dtype=np.float32)
        for rx, ry, available in self.resources:
            if not available:
                continue
            # Relative position
            dx = rx - x
            dy = ry - y
            # Check if in vision radius
            if abs(dx) <= self.vision_radius and abs(dy) <= self.vision_radius:
                # Map to 8×8 grid
                grid_x = int((dx + self.vision_radius) / (2 * self.vision_radius + 1) * 8)
                grid_y = int((dy + self.vision_radius) / (2 * self.vision_radius + 1) * 8)
                grid_x = np.clip(grid_x, 0, 7)
                grid_y = np.clip(grid_y, 0, 7)
                local_map[grid_y, grid_x] = 1.0
        
        obs[4:68] = local_map.flatten()
        
        # Global statistics
        # Resource density
        available_resources = sum(1 for _, _, avail in self.resources if avail)
        obs[68] = available_resources / self.num_resources
        
        # Swarm centroid (for social coordination)
        if self.agent_positions:
            swarm_x = np.mean([pos[0] for pos in self.agent_positions]) / self.grid_size
            swarm_y = np.mean([pos[1] for pos in self.agent_positions]) / self.grid_size
            obs[69] = swarm_x
            obs[70] = swarm_y
        
        # Distance to nearest resource
        min_dist = float('inf')
        for rx, ry, available in self.resources:
            if available:
                dist = np.sqrt((rx - x)**2 + (ry - y)**2)
                min_dist = min(min_dist, dist)
        obs[71] = min_dist / (self.grid_size * np.sqrt(2))  # Normalized
        
        # Fill rest with noise (simulates other environmental factors)
        obs[72:128] = np.random.randn(56) * 0.1
        
        return mx.array(obs)
    
    def step(self, actions: List[int]) -> Tuple[List[mx.array], List[float], Dict]:
        """
        Execute agent actions and update environment.
        
        Actions:
        - 0: Move North
        - 1: Move East
        - 2: Move South
        - 3: Move West
        - 4: Collect resource (if at resource location)
        
        Returns:
            observations: 128D observation for each agent
            rewards: Individual reward for each agent (list)
            info: Additional metrics
        """
        collected_this_step = 0
        individual_rewards = [0.0] * len(actions)
        
        # Track distances before movement for reward shaping
        prev_distances = []
        for agent_idx in range(len(actions)):
            x, y = self.agent_positions[agent_idx]
            min_dist = float('inf')
            for rx, ry, available in self.resources:
                if available:
                    dist = np.sqrt((rx - x)**2 + (ry - y)**2)
                    min_dist = min(min_dist, dist)
            prev_distances.append(min_dist)
        
        # Execute actions
        for agent_idx, action in enumerate(actions):
            x, y = self.agent_positions[agent_idx]
            
            # Movement
            if action == 0:  # North
                y = max(0, y - 1)
            elif action == 1:  # East
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # South
                y = min(self.grid_size - 1, y + 1)
            elif action == 3:  # West
                x = max(0, x - 1)
            elif action == 4:  # Collect
                # Check if at resource location
                for i, (rx, ry, available) in enumerate(self.resources):
                    if available and rx == x and ry == y:
                        self.resources[i] = (rx, ry, False)  # Collect
                        collected_this_step += 1
                        self.total_collected += 1
                        individual_rewards[agent_idx] += 10.0  # Big reward for collection!
            
            # Update position
            self.agent_positions[agent_idx] = (x, y)
            
            # Reward shaping: Small reward for moving closer to resources
            min_dist_after = float('inf')
            for rx, ry, available in self.resources:
                if available:
                    dist = np.sqrt((rx - x)**2 + (ry - y)**2)
                    min_dist_after = min(min_dist_after, dist)
            
            if min_dist_after < prev_distances[agent_idx]:
                individual_rewards[agent_idx] += 0.1  # Small reward for getting closer
            else:
                individual_rewards[agent_idx] -= 0.05  # Small penalty for moving away
        
        # Resource regeneration
        for i, (rx, ry, available) in enumerate(self.resources):
            if not available and np.random.rand() < self.regeneration_rate:
                self.resources[i] = (rx, ry, True)
        
        # Generate new observations
        observations = [self.get_observation(i) for i in range(len(actions))]
        
        # Info
        info = {
            'collected_this_step': collected_this_step,
            'total_collected': self.total_collected,
            'available_resources': sum(1 for _, _, avail in self.resources if avail)
        }
        
        self.collection_history.append(collected_this_step)
        
        return observations, individual_rewards, info
    
    def render(self, ax=None, show_vision: bool = True):
        """Visualize current environment state."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.clear()
        ax.set_xlim(-1, self.grid_size)
        ax.set_ylim(-1, self.grid_size)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        
        # Resources
        for rx, ry, available in self.resources:
            color = 'green' if available else 'lightgray'
            ax.scatter(rx, ry, s=200, c=color, marker='*', alpha=0.8, edgecolors='black')
        
        # Agents
        for idx, (x, y) in enumerate(self.agent_positions):
            ax.scatter(x, y, s=300, c='blue', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
            
            # Vision radius
            if show_vision:
                circle = Circle((x, y), self.vision_radius, color='blue', fill=False, alpha=0.2, linestyle='--')
                ax.add_patch(circle)
        
        ax.set_title(f'Foraging Environment - Collected: {self.total_collected}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')


class ForagingExperiment:
    """
    Run foraging experiments with conscious swarm.
    """
    
    def __init__(self, 
                 num_agents: int = 20,
                 grid_size: int = 50,
                 num_resources: int = 20):
        self.swarm = ConsciousSwarm(
            num_agents=num_agents,
            agent_state_dim=128,
            collective_dim=512,
            action_dim=5  # 4 movements + collect
        )
        
        self.env = ForagingEnvironment(
            grid_size=grid_size,
            num_resources=num_resources,
            vision_radius=5
        )
        
        self.history = {
            'rewards': [],
            'consensus': [],
            'messages': [],
            'collected': []
        }
    
    def run(self, num_steps: int = 100, render_interval: int = 20):
        """
        Run foraging experiment.
        """
        print(f"\n{'='*70}")
        print(f"🐜 FORAGING EXPERIMENT: {self.swarm.num_agents} Agents")
        print(f"{'='*70}")
        
        # Initialize
        self.env.reset(self.swarm.num_agents)
        
        for step in range(num_steps):
            # Get observations
            observations = [self.env.get_observation(i) for i in range(self.swarm.num_agents)]
            
            # Swarm cognitive cycle with previous rewards
            prev_rewards = self.history['rewards'][-1] if self.history['rewards'] else [0.0] * self.swarm.num_agents
            if isinstance(prev_rewards, (int, float)):
                prev_rewards = [prev_rewards / self.swarm.num_agents] * self.swarm.num_agents
            
            result = self.swarm.step(observations, rewards=prev_rewards)
            
            # Extract actions with epsilon-greedy exploration
            actions = result['actions']
            if step < 50:  # More exploration early on
                epsilon = 0.3
            else:
                epsilon = 0.1
            
            # Epsilon-greedy: randomly override some actions for exploration
            for i in range(len(actions)):
                if np.random.rand() < epsilon:
                    actions[i] = np.random.randint(0, 5)
            
            # Environment step
            new_observations, individual_rewards, info = self.env.step(actions)
            
            # Log metrics
            self.history['rewards'].append(individual_rewards)
            self.history['consensus'].append(result['consensus'])
            self.history['messages'].append(result['messages_sent'])
            self.history['collected'].append(info['total_collected'])
            
            # Periodic learning: let agents learn from experiences
            if step > 0 and step % 20 == 0 and len(self.swarm.shared_memory.memory.memories) > 32:
                self.swarm.collective_dream(batch_size=32, epochs=3)
            
            # Print progress
            if (step + 1) % 10 == 0:
                # Calculate average total reward over last 10 steps
                recent_rewards = self.history['rewards'][-10:]
                if recent_rewards:
                    avg_total_reward = np.mean([sum(r) if isinstance(r, list) else r for r in recent_rewards])
                else:
                    avg_total_reward = 0.0
                
                print(f"   Step {step+1:3d} | "
                      f"Avg Reward: {avg_total_reward:5.1f} | "
                      f"Collected: {info['total_collected']:3d} | "
                      f"Consensus: {result['consensus']:.3f} | "
                      f"Messages: {result['messages_sent']:2d}")
        
        print(f"\n✅ Experiment Complete!")
        print(f"   Total resources collected: {self.env.total_collected}")
        # Calculate total rewards
        total_rewards = [sum(r) if isinstance(r, list) else r for r in self.history['rewards']]
        print(f"   Average total reward per step: {np.mean(total_rewards):.2f}")
        print(f"   Average consensus: {np.mean(self.history['consensus']):.3f}")
        print(f"   Shared memories: {len(self.swarm.shared_memory.memory.memories)}")
        
        return self.history
    
    def visualize_results(self):
        """
        Generate comprehensive visualizations.
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Cumulative resources collected
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.history['collected'], linewidth=2, color='green')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Total Collected')
        ax1.set_title('Resource Collection Over Time', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Plot 2: Reward per step
        ax2 = fig.add_subplot(gs[0, 1])
        # Sum individual rewards to get total reward per step
        total_rewards = [sum(r) if isinstance(r, list) else r for r in self.history['rewards']]
        ax2.plot(total_rewards, linewidth=2, color='orange', alpha=0.6)
        # Moving average
        window = 10
        if len(total_rewards) >= window:
            moving_avg = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(total_rewards)), moving_avg, 
                    linewidth=2, color='red', label='Moving Avg (10 steps)')
            ax2.legend()
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Reward Signal (All Agents)', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Consensus evolution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.history['consensus'], linewidth=2, color='blue')
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Consensus')
        ax3.set_title('Swarm Consensus', fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # Plot 4: Communication activity
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.history['messages'], linewidth=2, color='purple')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Messages Sent')
        ax4.set_title('Swarm Communication', fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Plot 5: Environment snapshot
        ax5 = fig.add_subplot(gs[1, 1])
        self.env.render(ax=ax5, show_vision=False)
        
        # Plot 6: Efficiency metric
        ax6 = fig.add_subplot(gs[1, 2])
        # Calculate collection rate (resources per 10 steps)
        window = 10
        if len(self.history['collected']) >= window:
            collection_rates = []
            for i in range(window, len(self.history['collected'])):
                rate = self.history['collected'][i] - self.history['collected'][i-window]
                collection_rates.append(rate)
            ax6.plot(collection_rates, linewidth=2, color='teal')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Collection Rate (per 10 steps)')
        ax6.set_title('Efficiency Trend', fontweight='bold')
        ax6.grid(alpha=0.3)
        
        plt.suptitle('Phase 2.3: Collective Foraging Analysis', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig('foraging_experiment.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved: foraging_experiment.png")
        plt.show()


def main():
    """
    Run Phase 2.3 foraging experiments.
    """
    print("\n" + "="*70)
    print("🐜 PHASE 2.3: STRUCTURED ENVIRONMENT - FORAGING")
    print("="*70)
    print("\nResearch Questions:")
    print("1. Can swarm collectively optimize resource gathering?")
    print("2. Do agents develop specialized roles (scouts vs. gatherers)?")
    print("3. How does information propagate through the swarm?")
    print("4. Does consensus improve collective performance?")
    print()
    
    # Experiment 1: Small swarm (20 agents)
    print("\n--- Experiment 1: 20 Agents ---")
    exp1 = ForagingExperiment(num_agents=20, grid_size=50, num_resources=20)
    exp1.run(num_steps=100, render_interval=20)
    exp1.visualize_results()
    
    # TODO: Add comparison experiments
    # - Baseline: Agents with no communication
    # - Varied swarm sizes: 10 vs 50 agents
    # - Varied resource densities
    
    print("\n" + "="*70)
    print("✅ PHASE 2.3 FORAGING EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
