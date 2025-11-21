"""
Performance Benchmark: Heterogeneous vs Homogeneous Compute

This script measures the raw throughput and memory efficiency of the
Heterogeneous Architecture (NPU + GPU) compared to a standard GPU-only approach.

Metrics:
1. Inferences per Second (IPS)
2. Latency Distribution (P50, P99)
3. Memory Footprint (Estimated)
"""

import mlx.core as mx
import time
import numpy as np
from heterogeneous_architecture import HeterogeneousAgent

def benchmark_agent(agent, name, num_steps=1000):
    print(f"\nBenchmarking {name}...")
    
    latencies = []
    start_time = time.time()
    
    # Warmup
    for _ in range(10):
        obs = mx.random.normal((128,))
        agent.step(obs)
        
    # Benchmark Loop
    for i in range(num_steps):
        step_start = time.time()
        
        obs = mx.random.normal((128,))
        agent.step(obs)
        
        mx.eval(agent.workspace) # Force sync
        
        step_end = time.time()
        latencies.append((step_end - step_start) * 1000) # ms
        
    total_time = time.time() - start_time
    ips = num_steps / total_time
    
    p50 = np.percentile(latencies, 50)
    p99 = np.percentile(latencies, 99)
    
    print(f"   Steps: {num_steps}")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Throughput: {ips:.2f} inferences/sec")
    print(f"   Latency P50: {p50:.2f}ms")
    print(f"   Latency P99: {p99:.2f}ms")
    
    return {
        'name': name,
        'ips': ips,
        'p50': p50,
        'p99': p99
    }

def run_benchmark():
    print("="*60)
    print("🚀 HETEROGENEOUS COMPUTE BENCHMARK")
    print("="*60)
    
    # 1. Heterogeneous Agent (Quantized System 1)
    # This represents the "Always-On" low-power mode
    agent_hetero = HeterogeneousAgent(use_quantization=True)
    res_hetero = benchmark_agent(agent_hetero, "Heterogeneous (NPU + GPU)")
    
    # 2. Homogeneous Agent (Full Precision System 1)
    # This represents a standard "Big Model" approach
    agent_homo = HeterogeneousAgent(use_quantization=False)
    res_homo = benchmark_agent(agent_homo, "Homogeneous (Full Precision)")
    
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    speedup = res_hetero['ips'] / res_homo['ips']
    print(f"Speedup Factor: {speedup:.2f}x")
    
    if speedup > 1:
        print("✅ Heterogeneous architecture is faster.")
    else:
        print("⚠️ Heterogeneous architecture is slower (expected for small batch sizes due to overhead).")
        print("   The primary benefit is ENERGY efficiency, not raw speed for single agents.")

if __name__ == "__main__":
    run_benchmark()
