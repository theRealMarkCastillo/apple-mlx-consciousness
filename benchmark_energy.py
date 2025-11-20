import time
import mlx.core as mx
import argparse
from heterogeneous_architecture import HeterogeneousAgent

def stress_test_npu(agent, duration=15, batch_size=512):
    """
    Run a heavy quantized workload to stress the Neural Engine.
    Using large batch sizes to maximize throughput and power draw.
    """
    print("\n⚡ STRESS TEST: NPU (System 1 Quantized)")
    print(f"   Duration: {duration}s")
    print(f"   Batch Size: {batch_size}")
    print("   Target: ANE (Apple Neural Engine)")
    
    # Generate large batch of inputs
    inputs = mx.random.normal((batch_size, agent.state_dim))
    
    start = time.time()
    steps = 0
    
    while time.time() - start < duration:
        # Run quantized forward pass
        # We use the internal forward_quantized method directly to bypass overhead
        _ = agent.system1.forward_quantized(inputs)
        mx.eval(_)
        steps += 1
    
    elapsed = time.time() - start
    print(f"   ✅ Done. {steps} batches processed.")
    print(f"   Throughput: {steps*batch_size/elapsed:.1f} samples/sec")

def stress_test_gpu(agent, duration=15, batch_size=512):
    """
    Run a heavy full-precision workload to stress the GPU.
    """
    print("\n🔥 STRESS TEST: GPU (System 2 Full Precision)")
    print(f"   Duration: {duration}s")
    print(f"   Batch Size: {batch_size}")
    print("   Target: GPU")
    
    # Generate large batch of inputs
    inputs = mx.random.normal((batch_size, agent.state_dim))
    
    start = time.time()
    steps = 0
    
    while time.time() - start < duration:
        # Run full precision forward pass
        _ = agent.system2(inputs)
        mx.eval(_)
        steps += 1
        
    elapsed = time.time() - start
    print(f"   ✅ Done. {steps} batches processed.")
    print(f"   Throughput: {steps*batch_size/elapsed:.1f} samples/sec")

def main():
    parser = argparse.ArgumentParser(description="Energy Consumption Benchmark")
    parser.add_argument("--duration", type=int, default=15, help="Duration of each test in seconds")
    args = parser.parse_args()
    
    print("="*70)
    print("🔋 ENERGY CONSUMPTION BENCHMARK")
    print("="*70)
    print("This script runs sustained workloads to measure power usage.")
    print("\nINSTRUCTIONS:")
    print("1. Open a separate terminal window")
    print("2. Run this command to monitor power:")
    print("   sudo powermetrics --samplers cpu_power,gpu_power -i 1000")
    print("3. Watch the 'ANE Power' and 'GPU Power' metrics")
    print("\nStarting benchmark in 5 seconds...")
    time.sleep(5)
    
    # Initialize agent
    print("\nInitializing Heterogeneous Agent...")
    agent = HeterogeneousAgent(use_quantization=True)
    
    # Warmup
    print("Warming up...")
    mx.eval(mx.random.normal((100, 128)))
    time.sleep(2)
    
    # Run tests
    stress_test_npu(agent, duration=args.duration)
    
    print("\nCooldown (5s)...")
    time.sleep(5)
    
    stress_test_gpu(agent, duration=args.duration)
    
    print("\n" + "="*70)
    print("Benchmark Complete.")
    print("Compare the 'ANE Power' vs 'GPU Power' in your powermetrics output.")
    print("="*70)

if __name__ == "__main__":
    main()
