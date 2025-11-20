import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt
from heterogeneous_architecture import HeterogeneousAgent
import time

def run_training_comparison(num_steps=500, sleep_interval=50):
    """
    Compare standard training (random/full buffer) vs selective training (hard examples only).
    """
    print("="*70)
    print("🧠 HETEROGENEOUS TRAINING COMPARISON")
    print("="*70)
    print("Hypothesis: Training only on 'hard' examples (where System 2 was needed)")
    print("            is more efficient than training on all examples.")
    
    # Initialize agents
    print("\n1. Initializing agents...")
    agent_standard = HeterogeneousAgent(use_quantization=True)
    agent_selective = HeterogeneousAgent(use_quantization=True)
    
    # Metrics
    history_standard = {'sys2_usage': [], 'loss': []}
    history_selective = {'sys2_usage': [], 'loss': []}
    
    print(f"\n2. Running {num_steps} steps with sleep every {sleep_interval} steps...")
    
    start_time = time.time()
    
    for step in range(num_steps):
        # Generate synthetic "hard" and "easy" examples
        # We simulate this by varying the input distribution
        # Some inputs are "familiar" (close to 0), others "novel" (large magnitude)
        if np.random.random() < 0.2:
            # Novel/Hard input
            sensory = mx.random.normal((128,)) * 2.0
        else:
            # Familiar/Easy input
            sensory = mx.random.normal((128,)) * 0.5
            
        # Generate a target action (synthetic ground truth)
        # Simple rule: action depends on first dimension sign
        target_action = 1 if sensory[0].item() > 0 else 0
        reward = 1.0  # Simplified reward
        
        # Step both agents
        res_std = agent_standard.step(sensory, reward)
        res_sel = agent_selective.step(sensory, reward)
        
        # Sleep cycle
        if (step + 1) % sleep_interval == 0:
            print(f"\n--- Sleep Cycle (Step {step+1}) ---")
            
            # Standard Agent: Train on full buffer
            print("Standard Agent:")
            stats_std = agent_standard.sleep(epochs=5, prioritize_hard_examples=False)
            history_standard['loss'].append(stats_std.get('final_loss', 0))
            
            # Selective Agent: Train only on hard examples
            print("Selective Agent:")
            stats_sel = agent_selective.sleep(epochs=5, prioritize_hard_examples=True)
            history_selective['loss'].append(stats_sel.get('final_loss', 0))
            
            # Record System 2 usage since last sleep
            stats_std_perf = agent_standard.get_performance_stats()
            stats_sel_perf = agent_selective.get_performance_stats()
            
            history_standard['sys2_usage'].append(stats_std_perf['system2_usage'])
            history_selective['sys2_usage'].append(stats_sel_perf['system2_usage'])
            
    elapsed = time.time() - start_time
    print(f"\n✅ Comparison Complete in {elapsed:.2f}s")
    
    return history_standard, history_selective

def plot_results(h_std, h_sel):
    """Visualize the comparison."""
    epochs = range(len(h_std['loss']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss Comparison
    ax1.plot(epochs, h_std['loss'], 'o-', label='Standard (Full Buffer)')
    ax1.plot(epochs, h_sel['loss'], 's-', label='Selective (Hard Examples)')
    ax1.set_title('Training Loss during Sleep')
    ax1.set_xlabel('Sleep Cycle')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # System 2 Usage Comparison
    ax2.plot(epochs, h_std['sys2_usage'], 'o-', label='Standard')
    ax2.plot(epochs, h_sel['sys2_usage'], 's-', label='Selective')
    ax2.set_title('System 2 Usage (Dependency)')
    ax2.set_xlabel('Sleep Cycle')
    ax2.set_ylabel('Usage Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('heterogeneous_training_results.png')
    print("\n📊 Results saved to heterogeneous_training_results.png")

if __name__ == "__main__":
    h_std, h_sel = run_training_comparison()
    plot_results(h_std, h_sel)
