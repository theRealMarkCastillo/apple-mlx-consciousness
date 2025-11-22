import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
import time
import random

# Import our brain
from cognitive_architecture import BicameralAgent

class AppleCatcherGame:
    """
    A simple 1D game where the agent must catch 'Good' apples and avoid 'Bad' apples.
    Used to demonstrate System 1 (Reflexes) vs System 2 (Rule Learning).
    """
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.player_x = width // 2
        self.score = 0
        self.apples = [] # List of {'x', 'y', 'type'}
        self.game_over = False
        
        # Rules
        # 0 = Red (Good initially), 1 = Green (Bad initially)
        self.good_type = 0 
        self.bad_type = 1
        
        # Statistics
        self.caught_good = 0
        self.caught_bad = 0
        self.missed_good = 0
        
    def spawn_apple(self):
        if random.random() < 0.15: # 15% chance per frame
            apple_type = 0 if random.random() < 0.6 else 1 # 60% chance of Red
            self.apples.append({
                'x': random.randint(0, self.width - 1),
                'y': self.height - 1,
                'type': apple_type
            })

    def step(self, action):
        """
        Action: 0=Left, 1=Stay, 2=Right
        """
        reward = 0.0
        
        # Track distance to closest GOOD apple before move
        closest_good = None
        min_dist_before = float('inf')
        for apple in self.apples:
            if apple['type'] == self.good_type:
                dist = abs(apple['x'] - self.player_x)
                if dist < min_dist_before:
                    min_dist_before = dist
                    closest_good = apple

        # 1. Move Player
        if action == 0: # Left
            if self.player_x > 0:
                self.player_x -= 1
            else:
                reward -= 0.1 # Wall penalty
        elif action == 2: # Right
            if self.player_x < self.width - 1:
                self.player_x += 1
            else:
                reward -= 0.1 # Wall penalty
            
        # Reward Shaping: Did we move closer to a good apple?
        if closest_good:
            dist_after = abs(closest_good['x'] - self.player_x)
            if dist_after < min_dist_before:
                reward += 0.2 # Stronger encouragement
            elif dist_after > min_dist_before:
                reward -= 0.1 # Stronger penalty
            
        # 2. Move Apples & Check Collisions
        for apple in self.apples[:]:
            apple['y'] -= 0.5 # Fall slower (gives agent time to think)
            
            # Check collision
            # Apple is at (x, y). Player is at (player_x, 0).
            # Collision if y <= 0 and x matches
            if apple['y'] <= 0:
                if abs(apple['x'] - self.player_x) < 0.5:
                    # Caught!
                    if apple['type'] == self.good_type:
                        reward = 1.0
                        self.caught_good += 1
                    else:
                        reward = -5.0 # POISON! Massive penalty to force learning.
                        self.caught_bad += 1
                    self.apples.remove(apple)
                elif apple['y'] < -1:
                    # Missed / Hit ground (allow it to go slightly below 0 before removing)
                    if apple['type'] == self.good_type:
                        self.missed_good += 1
                        reward = -0.1 # Slight penalty for missing good food
                    self.apples.remove(apple)
                
        self.spawn_apple()
        self.score += reward
        return reward

    def get_state(self):
        """
        Returns a state vector for the agent.
        [DeltaX, AppleY, AppleType, IsApplePresent, 0.0]
        DeltaX = (AppleX - PlayerX) / Width. Range approx -1 to 1.
        """
        # Find closest apple
        closest_apple = None
        min_dist = float('inf')
        
        for apple in self.apples:
            # Distance metric: Vertical distance is most important for "closest"
            dist = apple['y']
            if dist < min_dist:
                min_dist = dist
                closest_apple = apple
                
        state = np.zeros(5, dtype=np.float32)
        
        if closest_apple:
            # Relative X is much easier to learn than absolute positions
            # If DeltaX > 0, Apple is to the right. If < 0, to the left.
            delta_x = (closest_apple['x'] - self.player_x) / self.width
            state[0] = delta_x 
            state[1] = closest_apple['y'] / self.height
            state[2] = closest_apple['type'] # 0 or 1
            state[3] = 1.0 # Apple exists
        else:
            state[0] = 0.0
            state[1] = 1.0
            state[2] = 0.0
            state[3] = 0.0
            
        state[4] = 0.0 # Placeholder
            
        return mx.array(state)

# --- Visualization ---

def run_demo():
    print("🍎 Starting Apple Catcher Demo...")
    print("   Red Apples = Good (+1)")
    print("   Green Apples = Bad (-1)")
    print("   ...until the rules change!")

    # Initialize Game and Agent
    game = AppleCatcherGame(width=12, height=15)
    agent = BicameralAgent(state_dim=5, action_dim=3)
    
    # Setup Plot
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 1.5, 0.5])
    ax_game = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    ax_hardware = fig.add_subplot(gs[2])
    
    plt.subplots_adjust(bottom=0.2, wspace=0.3)
    
    # Game Elements
    player_rect = Rectangle((0, 0), 1, 1, color='blue', label='Agent')
    ax_game.add_patch(player_rect)
    apple_patches = []
    
    # Stats Elements
    x_data = []
    y_conf = []
    y_score = []
    line_conf, = ax_stats.plot([], [], 'r-', label='Sys2 Confidence', linewidth=2)
    line_score, = ax_stats.plot([], [], 'g-', label='Score', linewidth=2)
    
    ax_game.set_xlim(0, game.width)
    ax_game.set_ylim(0, game.height)
    ax_game.set_title("Apple Catcher: Learning Rules")
    ax_game.grid(True, alpha=0.3)
    
    ax_stats.set_xlim(0, 100)
    ax_stats.set_ylim(-5, 20)
    ax_stats.set_title("Brain Internals")
    ax_stats.legend(loc='upper left')
    ax_stats.grid(True, alpha=0.3)

    # Hardware Elements
    hardware_bars = ax_hardware.bar(['NPU\n(Sys1)', 'GPU\n(Sys2)'], [0, 0], color=['#00ff00', '#ff9900'])
    ax_hardware.set_ylim(0, 1.2)
    ax_hardware.set_title("Compute Load")
    
    # Text info
    info_text = ax_game.text(0.5, 1.05, "", transform=ax_game.transAxes, ha="center")
    rule_text = ax_game.text(0.5, 0.95, "RULE: RED IS GOOD", transform=ax_game.transAxes, ha="center", color="red", fontweight="bold")

    frame_count = 0
    phase_switched = False
    prev_reward = 0.0
    
    # Stuck detector
    last_x = -1
    stuck_frames = 0

    def update(frame):
        nonlocal frame_count, phase_switched, prev_reward, last_x, stuck_frames
        frame_count += 1
        
        # 1. AI Step
        state = game.get_state()
        
        # Stuck Detector: If we haven't moved horizontally for 20 frames, force a random move
        # This breaks "local minima" where the agent is content doing nothing
        if game.player_x == last_x:
            stuck_frames += 1
        else:
            stuck_frames = 0
            last_x = game.player_x
            
        force_random = False
        if stuck_frames > 20:
            force_random = True
            stuck_frames = 0 # Reset
        
        # Pass the reward from the PREVIOUS frame
        decision = agent.step(state, reward=prev_reward) 
        
        if force_random:
            # Override brain to break loop
            action = random.choice([0, 2]) # Force Left or Right (not stay)
            
            # CRITICAL: Tell the agent we overrode its decision!
            # Otherwise it thinks it decided to 'Stay' but sees the world move, causing confusion.
            action_vec = mx.zeros((agent.action_dim,))
            action_vec[action] = 1.0
            agent.last_action = action_vec
            
            # Visual feedback
            rule_text.set_text("⚠️ STUCK DETECTED: FORCING MOVE ⚠️")
        else:
            action = decision['action']
            if not phase_switched:
                rule_text.set_text("RULE: RED IS GOOD")
            else:
                rule_text.set_text("⚠️ RULE CHANGE: GREEN IS GOOD! ⚠️")
        
        # --- Hardware Simulation ---
        # NPU (System 1) is always active for inference (Fast)
        npu_load = 0.8 + (0.1 * random.random())
        
        # GPU (System 2) is active only when:
        # 1. Learning (Backprop) - triggered by reward
        # 2. Low Confidence (Planning/Imagination)
        gpu_load = 0.1 # Idle
        
        is_learning = abs(prev_reward) > 0
        is_thinking = not (decision['confidence'].item() > 0.5)
        
        if is_learning:
            gpu_load += 0.5
        if is_thinking:
            gpu_load += 0.3
            
        gpu_load = min(1.0, gpu_load)
        
        # Update Bars
        hardware_bars[0].set_height(npu_load)
        hardware_bars[1].set_height(gpu_load)
        
        # Color shift for GPU based on load
        if gpu_load > 0.6:
            hardware_bars[1].set_color('red') # Heavy Load
        else:
            hardware_bars[1].set_color('orange') # Idle/Low
        
        # 2. Game Step
        reward = game.step(action)
        prev_reward = reward
        
        # 3. Learn
        # We rely on agent.step() to handle learning in the next cycle using prev_reward
        # But for the very first success, we might want to boost it? 
        # No, let's trust the loop.
            
        # 4. Rule Switch Logic
        if frame_count == 200:
            game.good_type = 1 # Green is now good
            game.bad_type = 0  # Red is now bad
            rule_text.set_text("⚠️ RULE CHANGE: GREEN IS GOOD! ⚠️")
            rule_text.set_color("green")
            phase_switched = True
            # Reset score for visual clarity of new phase
            # game.score = 0 
            
        # --- Drawing ---
        
        # Update Player
        player_rect.set_xy((game.player_x, 0))
        
        # Update Apples
        # Remove old patches
        for p in apple_patches:
            p.remove()
        apple_patches.clear()
        
        for apple in game.apples:
            color = 'red' if apple['type'] == 0 else 'green'
            # If rule switched, maybe visually indicate "poison" vs "food"? 
            # No, let's keep colors constant so we see the agent struggle.
            
            circle = Circle((apple['x'] + 0.5, apple['y'] + 0.5), 0.4, color=color)
            ax_game.add_patch(circle)
            apple_patches.append(circle)
            
        # Update Stats
        x_data.append(frame_count)
        y_conf.append(decision['confidence'].item())
        y_score.append(game.score)
        
        # Keep window moving
        if len(x_data) > 100:
            ax_stats.set_xlim(frame_count - 100, frame_count)
            
        line_conf.set_data(x_data, y_conf)
        line_score.set_data(x_data, y_score)
        
        # Auto-scale score
        if game.score > ax_stats.get_ylim()[1]:
            ax_stats.set_ylim(ax_stats.get_ylim()[0], game.score + 5)
        if game.score < ax_stats.get_ylim()[0]:
            ax_stats.set_ylim(game.score - 5, ax_stats.get_ylim()[1])

        # Show probabilities in title for debugging
        probs = decision['probs']
        p_left = probs[0].item()
        p_stay = probs[1].item()
        p_right = probs[2].item()
        
        info_text.set_text(f"Score: {game.score:.1f} | Conf: {decision['confidence'].item():.2f}\nL:{p_left:.2f} S:{p_stay:.2f} R:{p_right:.2f}")
        
        return player_rect, line_conf, line_score, info_text, rule_text, hardware_bars[0], hardware_bars[1]

    ani = animation.FuncAnimation(fig, update, frames=500, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    run_demo()
