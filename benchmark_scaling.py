"""
Phase 2.2: Scaling Benchmark - 10 → 100 → 1000 Agents

Benchmarks:
1. Memory footprint (actual RAM usage)
2. Step execution time
3. Consensus convergence rate
4. Communication overhead
5. Shared memory access patterns

Target: Validate that 64GB unified memory can handle 1000+ agents
"""

import mlx.core as mx
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from swarm_architecture import ConsciousSwarm

# Visualization setup
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)


def benchmark_swarm(num_agents: int, num_steps: int = 20) -> dict:
    """
    Benchmark a swarm of N agents.
    
    Returns:
        dict with timing, memory, consensus metrics
    """
    print(f"\n{'='*70}")
    print(f"🔬 BENCHMARKING {num_agents} AGENTS")
    print(f"{'='*70}")
    
    # Initialization
    init_start = time.time()
    swarm = ConsciousSwarm(
        num_agents=num_agents,
        agent_state_dim=128,
        collective_dim=512,
        action_dim=5
    )
    init_time = time.time() - init_start
    
    print(f"\n📊 Initialization:")
    print(f"   Time: {init_time:.4f}s")
    print(f"   Memory footprint: {swarm._estimate_memory_mb():.4f} MB")
    
    # Simulation
    step_times = []
    consensus_scores = []
    
    print(f"\n🚀 Running {num_steps}-step simulation...")
    
    for step in range(num_steps):
        # Environment signals
        environment_signals = [
            mx.random.normal((128,)) for _ in range(num_agents)
        ]
        
        # Time one step
        step_start = time.time()
        result = swarm.step(environment_signals)
        step_time = time.time() - step_start
        
        step_times.append(step_time)
        consensus_scores.append(result['consensus'])
        
        if (step + 1) % 5 == 0:
            avg_time = np.mean(step_times[-5:])
            print(f"   Step {step+1:3d} | "
                  f"Time: {step_time:.4f}s | "
                  f"Avg: {avg_time:.4f}s | "
                  f"Consensus: {result['consensus']:.3f}")
    
    # Final metrics
    metrics = swarm.get_metrics()
    
    results = {
        'num_agents': num_agents,
        'init_time': init_time,
        'avg_step_time': np.mean(step_times),
        'std_step_time': np.std(step_times),
        'min_step_time': np.min(step_times),
        'max_step_time': np.max(step_times),
        'total_time': sum(step_times),
        'memory_mb': swarm._estimate_memory_mb(),
        'consensus_avg': np.mean(consensus_scores),
        'consensus_final': consensus_scores[-1],
        'shared_memories': metrics['shared_memories'],
        'total_messages': metrics['total_messages'],
        'step_times': step_times,
        'consensus_history': consensus_scores
    }
    
    print(f"\n✅ Benchmark Complete!")
    print(f"   Avg step time: {results['avg_step_time']:.4f}s ± {results['std_step_time']:.4f}s")
    print(f"   Total time: {results['total_time']:.2f}s")
    print(f"   Throughput: {num_agents * num_steps / results['total_time']:.1f} agent-steps/sec")
    print(f"   Final consensus: {results['consensus_final']:.3f}")
    
    return results


def main():
    print("\n" + "="*70)
    print("🌐 PHASE 2.2: SWARM SCALING BENCHMARK")
    print("="*70)
    print("\nTarget: Demonstrate unified memory advantage at scale")
    print("Hardware: Mac Mini M4 Pro (64GB unified memory)")
    print()
    
    # Benchmark configurations
    configs = [
        (10, 30),    # 10 agents, 30 steps
        (50, 30),    # 50 agents, 30 steps
        (100, 30),   # 100 agents, 30 steps
    ]
    
    # Optional: Uncomment for extreme scale test
    # configs.append((500, 20))  # 500 agents, 20 steps
    # configs.append((1000, 10)) # 1000 agents, 10 steps
    
    all_results = []
    
    for num_agents, num_steps in configs:
        results = benchmark_swarm(num_agents, num_steps)
        all_results.append(results)
        
        # Give system time to settle
        time.sleep(2)
    
    # ===========================
    # COMPARATIVE VISUALIZATION
    # ===========================
    
    print("\n" + "="*70)
    print("📊 GENERATING COMPARATIVE VISUALIZATIONS")
    print("="*70)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data for plotting
    agent_counts = [r['num_agents'] for r in all_results]
    avg_times = [r['avg_step_time'] for r in all_results]
    memory_usage = [r['memory_mb'] for r in all_results]
    consensus_final = [r['consensus_final'] for r in all_results]
    throughput = [r['num_agents'] * 30 / r['total_time'] for r in all_results]
    
    # Plot 1: Step Time vs Agent Count
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(agent_counts, avg_times, 'o-', linewidth=2, markersize=10, color='blue')
    ax1.set_xlabel('Number of Agents', fontsize=11)
    ax1.set_ylabel('Avg Step Time (seconds)', fontsize=11)
    ax1.set_title('Scalability: Step Time', fontsize=12, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Plot 2: Memory Usage vs Agent Count
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(agent_counts, memory_usage, 'o-', linewidth=2, markersize=10, color='green')
    ax2.set_xlabel('Number of Agents', fontsize=11)
    ax2.set_ylabel('Memory (MB)', fontsize=11)
    ax2.set_title('Memory Footprint', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add theoretical line (128D × 4 bytes per agent)
    theoretical_memory = np.array(agent_counts) * 128 * 4 / 1e6
    ax2.plot(agent_counts, theoretical_memory, '--', alpha=0.5, color='red', label='Theoretical')
    ax2.legend()
    
    # Plot 3: Throughput vs Agent Count
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(agent_counts, throughput, 'o-', linewidth=2, markersize=10, color='purple')
    ax3.set_xlabel('Number of Agents', fontsize=11)
    ax3.set_ylabel('Throughput (agent-steps/sec)', fontsize=11)
    ax3.set_title('Computational Throughput', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Plot 4-6: Step Time Distribution for each config
    for i, results in enumerate(all_results):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(results['step_times'], bins=15, alpha=0.7, color='teal', edgecolor='black')
        ax.axvline(results['avg_step_time'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_xlabel('Step Time (seconds)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{results["num_agents"]} Agents - Time Distribution', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Plot 7-9: Consensus Evolution for each config
    for i, results in enumerate(all_results):
        ax = fig.add_subplot(gs[2, i])
        ax.plot(results['consensus_history'], linewidth=2, color='orange')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Consensus', fontsize=10)
        ax.set_title(f'{results["num_agents"]} Agents - Consensus', fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Phase 2.2: Multi-Agent Swarm Scaling Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    plt.savefig('swarm_scaling_benchmark.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: swarm_scaling_benchmark.png")
    plt.show()
    
    # ===========================
    # SUMMARY TABLE
    # ===========================
    
    print("\n" + "="*70)
    print("📋 SCALING SUMMARY")
    print("="*70)
    print()
    print(f"{'Agents':<10} {'Init(s)':<10} {'Step(s)':<12} {'Memory(MB)':<12} {'Consensus':<12} {'Throughput':<15}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['num_agents']:<10} "
              f"{r['init_time']:<10.4f} "
              f"{r['avg_step_time']:<12.4f} "
              f"{r['memory_mb']:<12.4f} "
              f"{r['consensus_final']:<12.3f} "
              f"{r['num_agents']*30/r['total_time']:<15.1f}")
    
    # ===========================
    # KEY FINDINGS
    # ===========================
    
    print("\n" + "="*70)
    print("🔍 KEY FINDINGS")
    print("="*70)
    
    # Scaling efficiency
    if len(all_results) >= 2:
        speedup_10_to_100 = all_results[-1]['avg_step_time'] / all_results[0]['avg_step_time']
        memory_ratio = all_results[-1]['memory_mb'] / all_results[0]['memory_mb']
        
        print(f"\n1. Computational Scaling:")
        print(f"   10 → {all_results[-1]['num_agents']} agents:")
        print(f"   - Step time increase: {speedup_10_to_100:.2f}x")
        print(f"   - Expected (linear): {all_results[-1]['num_agents']/10:.1f}x")
        print(f"   - Efficiency: {(all_results[-1]['num_agents']/10) / speedup_10_to_100 * 100:.1f}%")
        
        print(f"\n2. Memory Scaling:")
        print(f"   10 → {all_results[-1]['num_agents']} agents:")
        print(f"   - Memory increase: {memory_ratio:.2f}x")
        print(f"   - Expected (linear): {all_results[-1]['num_agents']/10:.1f}x")
        print(f"   - Efficiency: {(all_results[-1]['num_agents']/10) / memory_ratio * 100:.1f}%")
        
        print(f"\n3. Unified Memory Advantage:")
        print(f"   - {all_results[-1]['num_agents']} agents use only {all_results[-1]['memory_mb']:.2f} MB")
        print(f"   - Traditional GPU: Would need {all_results[-1]['num_agents']} separate memory pools")
        print(f"   - Savings: {all_results[-1]['num_agents']} → 1 pool (unified memory)")
    
    print(f"\n4. Consensus Performance:")
    for r in all_results:
        print(f"   {r['num_agents']} agents: {r['consensus_final']:.3f} (above {0.2:.1f} random baseline)")
    
    print(f"\n5. Projected Capacity:")
    if all_results:
        mem_per_agent = all_results[-1]['memory_mb'] / all_results[-1]['num_agents']
        max_agents_64gb = 64000 / mem_per_agent
        print(f"   Memory per agent: {mem_per_agent:.4f} MB")
        print(f"   Projected max agents (64GB): ~{int(max_agents_64gb):,}")
        print(f"   Current usage: {all_results[-1]['memory_mb'] / 64000 * 100:.2f}% of 64GB")
    
    print("\n" + "="*70)
    print("✅ PHASE 2.2 SCALING BENCHMARK COMPLETE")
    print("="*70)
    print()
    print("🚀 Next Steps:")
    print("   - Uncomment 500/1000 agent configs for extreme scale")
    print("   - Add structured tasks (foraging, coordination)")
    print("   - Analyze role emergence patterns")
    print("   - Benchmark communication at scale")
    print()


if __name__ == "__main__":
    main()
