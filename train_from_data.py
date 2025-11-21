import json
import argparse
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from heterogeneous_architecture import HeterogeneousAgent

def train_agent_from_file(filename="synthetic_experiences.json", num_epochs: int = 5, sleep_epochs: int = 10, save_path: str = "agent_brain.npz", action_dim: int = 2, mode: str = "policy_gradient", defer_quantization: bool = False):
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
    if mode == "policy_gradient":
        experiences = [e for e in experiences if e['reward'] > 0]
        print(f"Filtered to {len(experiences)} positive experiences for Behavioral Cloning (Policy Gradient).")
    else:
        print(f"Using all {len(experiences)} experiences for Supervised Learning.")
    
    # Initialize a fresh agent
    print("\nInitializing fresh Heterogeneous Agent...")
    # If defer_quantization is True, we start with full precision for better training stability
    use_quantization = not defer_quantization
    agent = HeterogeneousAgent(state_dim=128, action_dim=action_dim, use_quantization=use_quantization)
    
    if defer_quantization:
        print("⚠️ Quantization deferred until after training (using FP32 for training).")

    # Pre-load the buffer
    # In a real scenario, we might batch this differently, but here we just fill the buffer
    # The agent's buffer has a max size, so we might need to sleep multiple times
    
    batch_size = 100 # Fixed batch size for supervised training
    if mode == "policy_gradient":
        batch_size = agent.max_online_buffer

    num_batches = len(experiences) // batch_size
    
    # Optimizer for supervised learning
    optimizer = optim.Adam(learning_rate=0.001)

    # Loop over the dataset multiple times (Epochs)
    print(f"\nTraining for {num_epochs} epochs over the dataset (Mode: {mode})...")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        # Shuffle experiences
        np.random.shuffle(experiences)
        
        epoch_loss = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(experiences))
            
            batch = experiences[start_idx:end_idx]
            
            if mode == "policy_gradient":
                agent.online_buffer = batch
                # Train using internal sleep mechanism
                stats = agent.sleep(epochs=sleep_epochs)
                if i % 5 == 0:
                    print(f"   Batch {i+1}: Loss {stats['initial_loss']:.4f} -> {stats['final_loss']:.4f}")
            
            elif mode == "supervised":
                # Prepare batch data
                states = mx.array([e['state'] for e in batch])
                actions = mx.array([e['action'] for e in batch])
                
                # Define loss function (Cross Entropy)
                def loss_fn(model):
                    # Forward pass to get logits (accessing internal layers of System 1)
                    x = mx.tanh(model.l1(states))
                    logits = model.l2(x)
                    return mx.mean(nn.losses.cross_entropy(logits, actions))
                
                # Training step
                loss_and_grad_fn = nn.value_and_grad(agent.system1, loss_fn)
                loss, grads = loss_and_grad_fn(agent.system1)
                optimizer.update(agent.system1, grads)
                
                # Force evaluation
                mx.eval(agent.system1.parameters(), optimizer.state)
                
                epoch_loss += loss.item()
                
                if i % 20 == 0:
                    print(f"   Batch {i+1}/{num_batches}: Loss {loss.item():.4f}")

        if mode == "supervised":
            print(f"   Average Epoch Loss: {epoch_loss / num_batches:.4f}")
                
    print(f"\n✅ Training Complete.")
    
    # Apply quantization if it was deferred
    if defer_quantization:
        print("Applying deferred quantization...")
        agent.system1.quantize_weights(force=True)
        agent.use_quantization = True
    
    # Save the trained brain
    agent.save_brain(save_path)
    if os.path.exists(save_path):
        print(f"Saved brain file size: {os.path.getsize(save_path)} bytes")
    
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
    parser = argparse.ArgumentParser(description="Offline training from synthetic experiences")
    parser.add_argument("--file", type=str, default="synthetic_experiences.json", help="Path to experiences JSON file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs over dataset")
    parser.add_argument("--sleep-epochs", type=int, default=10, help="Sleep() training epochs per batch")
    parser.add_argument("--save", type=str, default="agent_brain.npz", help="Output path for trained brain weights")
    parser.add_argument("--action-dim", type=int, default=2, help="Action dimension for agent (default 2 for this dataset)")
    parser.add_argument("--mode", type=str, default="policy_gradient", choices=["policy_gradient", "supervised"], help="Training mode")
    parser.add_argument("--defer-quantization", action="store_true", help="Train in FP32 and quantize after")
    args = parser.parse_args()

    train_agent_from_file(filename=args.file, num_epochs=args.epochs, sleep_epochs=args.sleep_epochs, save_path=args.save, action_dim=args.action_dim, mode=args.mode, defer_quantization=args.defer_quantization)
