import json
import mlx.core as mx
import numpy as np
from heterogeneous_architecture import HeterogeneousAgent

def train_agent_from_file(filename="synthetic_experiences.json"):
    """
    Demonstrates how to train an agent using offline data.
    """
    print(f"Loading training data from {filename}...")
    
    try:
        with open(filename, 'r') as f:
            experiences = json.load(f)
    except FileNotFoundError:
        print("Error: File not found. Run generate_training_data.py first.")
        return

    print(f"Loaded {len(experiences)} experiences.")

    # Filter for positive experiences only (Behavioral Cloning)
    # Policy Gradient with negative rewards can be unstable (unbounded loss)
    # so we focus on learning "what to do" from successful examples.
    experiences = [e for e in experiences if e['reward'] > 0]
    print(f"Filtered to {len(experiences)} positive experiences for Behavioral Cloning.")
    
    # Initialize a fresh agent
    print("\nInitializing fresh Heterogeneous Agent...")
    agent = HeterogeneousAgent(use_quantization=True)
    
    # Pre-load the buffer
    # In a real scenario, we might batch this differently, but here we just fill the buffer
    # The agent's buffer has a max size, so we might need to sleep multiple times
    
    batch_size = agent.max_online_buffer
    num_batches = len(experiences) // batch_size
    
    # Loop over the dataset multiple times (Epochs)
    num_epochs = 5
    print(f"\nTraining for {num_epochs} epochs over the dataset...")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        # Shuffle experiences
        np.random.shuffle(experiences)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(experiences))
            
            batch = experiences[start_idx:end_idx]
            agent.online_buffer = batch
            
            # Train
            stats = agent.sleep(epochs=10) # Increased internal epochs
            
            if i % 5 == 0:
                print(f"   Batch {i+1}: Loss {stats['initial_loss']:.4f} -> {stats['final_loss']:.4f}")
                
    print(f"\n✅ Training Complete.")
    
    # Evaluation
    print("\nEvaluating learned behavior...")
    print("Pattern to learn: State[0] > 0 -> Action 0 | State[0] <= 0 -> Action 1")
    
    test_cases = [
        (0.8, 0),   # Should be Action 0
        (-0.8, 1),  # Should be Action 1
        (0.1, 0),   # Should be Action 0
        (-0.1, 1)   # Should be Action 1
    ]
    
    correct = 0
    for val, expected in test_cases:
        # Create state with specific value at index 0
        state = mx.zeros((128,))
        state[0] = val
        # Add some noise to other dimensions
        state = state + mx.random.normal((128,)) * 0.1
        
        # Get agent's decision (System 1 intuition)
        decision = agent.step(state)
        action = decision['action']
        probs = decision['action_probs']
        
        # Map action to 0 or 1 (in case agent outputs other actions, though we trained on 0/1)
        # The agent has action_dim=10 by default, but we only trained on 0 and 1.
        
        is_correct = (action == expected)
        if is_correct: correct += 1
        
        print(f"   Input: {val:.1f} -> Agent Action: {action} (Expected: {expected}) [{'✅' if is_correct else '❌'}]")
        print(f"      Probs: [0]={probs[0].item():.4f}, [1]={probs[1].item():.4f}, [2]={probs[2].item():.4f}")
        
    print(f"\nAccuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    train_agent_from_file()
