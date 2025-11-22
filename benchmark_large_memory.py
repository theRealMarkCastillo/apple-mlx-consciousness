"""
Phase 2.3: Large Scale Data Benchmark

Benchmarks:
1. Loading large synthetic dataset (500MB+) into Unified Memory
2. Running 50 agents (optimal count) against real data patterns
3. Measuring memory bandwidth impact on step time

Target: Validate that the system can handle "Big Data" in Unified Memory
"""

import mlx.core as mx
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from swarm_architecture import ConsciousSwarm

# Visualization setup
sns.set_style('darkgrid')

def load_synthetic_data(filename="synthetic_experiences.json"):
    print(f"📂 Loading synthetic data from {filename}...")
    start = time.time()
    with open(filename, 'r') as f:
        data = json.load(f)
    load_time = time.time() - start
    print(f"✅ Loaded {len(data)} experiences in {load_time:.4f}s")
    return data

def benchmark_large_scale(num_agents=50, num_steps=50, data_file="synthetic_experiences.json"):
    print(f"\n{'='*70}")
    print(f"🐘 BENCHMARKING LARGE SCALE DATA ({num_agents} AGENTS)")
    print(f"{'='*70}")
    
    # 1. Load Data
    experiences = load_synthetic_data(data_file)
    
    # 2. Initialize Swarm with Large Memory
    print(f"\n🌐 Initializing Swarm with memory file: {data_file}")
    init_start = time.time()
    swarm = ConsciousSwarm(
        num_agents=num_agents,
        agent_state_dim=128,
        collective_dim=512,
        action_dim=5,
        memory_file=data_file  # Point to the large dataset
    )
    init_time = time.time() - init_start
    
    print(f"📊 Initialization:")
    print(f"   Time: {init_time:.4f}s")
    print(f"   Memory footprint: {swarm._estimate_memory_mb():.4f} MB (Active)")
    print(f"   Shared Memory: {len(swarm.shared_memory.memory.memories)} items")
    
    # 3. Run Simulation using Data as Input
    print(f"\n🚀 Running {num_steps}-step simulation with data-driven inputs...")
    
    step_times = []
    consensus_scores = []
    
    for step in range(num_steps):
        # Use data from the dataset as environment signals
        # We pick 'num_agents' random samples from the dataset to simulate "seeing" the world
        batch_indices = np.random.choice(len(experiences), num_agents)
        
        environment_signals = []
        for idx in batch_indices:
            # Convert list to mx.array
            signal = mx.array(experiences[idx]['state'])
            environment_signals.append(signal)
            
        # Time one step
        step_start = time.time()
        result = swarm.step(environment_signals)
        step_time = time.time() - step_start
        
        step_times.append(step_time)
        consensus_scores.append(result['consensus'])
        
        if (step + 1) % 10 == 0:
            avg_time = np.mean(step_times[-10:])
            print(f"   Step {step+1:3d} | "
                  f"Time: {step_time:.4f}s | "
                  f"Avg: {avg_time:.4f}s | "
                  f"Consensus: {result['consensus']:.3f}")

    # 4. Final Metrics
    print(f"\n✅ Large Scale Benchmark Complete!")
    print(f"   Avg step time: {np.mean(step_times):.4f}s")
    print(f"   Throughput: {num_agents * num_steps / sum(step_times):.1f} agent-steps/sec")
    print(f"   Final consensus: {consensus_scores[-1]:.3f}")
    
    return step_times, consensus_scores

if __name__ == "__main__":
    # Ensure data exists
    import os
    if not os.path.exists("synthetic_experiences.json"):
        print("⚠️  synthetic_experiences.json not found. Generating it now...")
        import generate_training_data
        generate_training_data.generate_synthetic_data(num_experiences=10000, output_file="synthetic_experiences.json")
        
    benchmark_large_scale()
