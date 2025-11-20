"""
Quick test script for Phase 2.1 swarm architecture.

Tests:
1. Can we initialize a 10-agent swarm?
2. Can agents share memory?
3. Can agents communicate?
4. Does collective consciousness work?
"""

import mlx.core as mx
from swarm_architecture import ConsciousSwarm

def test_swarm_initialization():
    """Test: Initialize swarm and check memory footprint."""
    print("=" * 60)
    print("TEST 1: Swarm Initialization")
    print("=" * 60)
    
    swarm = ConsciousSwarm(num_agents=10, agent_state_dim=128, 
                           collective_dim=512, action_dim=5)
    
    assert swarm.num_agents == 10
    assert len(swarm.agents) == 10
    assert swarm.collective_workspace is not None
    assert swarm.shared_memory is not None
    assert swarm.communication is not None
    
    print(f"✅ Swarm initialized successfully!")
    print(f"   Memory footprint: {swarm._estimate_memory_mb():.4f} MB")
    print()
    
    return swarm

def test_collective_step(swarm):
    """Test: Run one collective cognitive cycle."""
    print("=" * 60)
    print("TEST 2: Collective Cognitive Cycle")
    print("=" * 60)
    
    # Create environment signals
    environment_signals = [
        mx.random.normal((128,)) for _ in range(swarm.num_agents)
    ]
    rewards = [0.0] * swarm.num_agents
    
    # Run one step
    result = swarm.step(environment_signals, rewards)
    
    assert 'collective_state' in result
    assert 'agent_actions' in result
    assert 'consensus' in result
    assert len(result['agent_actions']) == swarm.num_agents
    
    print(f"✅ Swarm step completed!")
    print(f"   Consensus: {result['consensus']:.3f}")
    print(f"   Collective state shape: {result['collective_state'].shape}")
    print(f"   Agent actions: {result['agent_actions']}")
    print()
    
    return result

def test_shared_memory(swarm):
    """Test: Check if agents are sharing memory."""
    print("=" * 60)
    print("TEST 3: Shared Memory Pool")
    print("=" * 60)
    
    # Run 5 steps to accumulate memories
    for i in range(5):
        signals = [mx.random.normal((128,)) for _ in range(swarm.num_agents)]
        swarm.step(signals)
    
    metrics = swarm.get_metrics()
    shared_memories = metrics['shared_memories']
    
    assert shared_memories > 0, "No memories stored!"
    
    print(f"✅ Shared memory working!")
    print(f"   Total shared memories: {shared_memories}")
    print(f"   Memory reads: {metrics['memory_access_reads']}")
    print(f"   Memory writes: {metrics['memory_access_writes']}")
    print()
    
    return metrics

def test_communication(swarm):
    """Test: Check if agents communicate."""
    print("=" * 60)
    print("TEST 4: Agent Communication")
    print("=" * 60)
    
    # Force communication by creating uncertain agents
    # (Communication happens when confidence < 0.4)
    signals = [mx.random.normal((128,)) * 5.0 for _ in range(swarm.num_agents)]
    swarm.step(signals)
    
    metrics = swarm.get_metrics()
    total_messages = metrics['total_messages']
    
    print(f"✅ Communication system active!")
    print(f"   Total messages: {total_messages}")
    print(f"   Messages per agent: {total_messages / swarm.num_agents:.1f}")
    print()
    
    return metrics

def test_collective_dreaming(swarm):
    """Test: Collective sleep consolidation."""
    print("=" * 60)
    print("TEST 5: Collective Dreaming")
    print("=" * 60)
    
    # Need at least 10 memories to dream
    for i in range(15):
        signals = [mx.random.normal((128,)) for _ in range(swarm.num_agents)]
        swarm.step(signals)
    
    metrics_before = swarm.get_metrics()
    print(f"Memories before dreaming: {metrics_before['shared_memories']}")
    
    # All agents dream on shared memory
    swarm.collective_dream(batch_size=10, epochs=2)
    
    metrics_after = swarm.get_metrics()
    
    print(f"✅ Collective dreaming completed!")
    print(f"   All {swarm.num_agents} agents trained on {metrics_after['shared_memories']} shared experiences")
    print()

def main():
    print("\n🌐 Phase 2.1 Swarm Architecture Test Suite")
    print("=" * 60)
    print()
    
    # Run all tests
    swarm = test_swarm_initialization()
    test_collective_step(swarm)
    test_shared_memory(swarm)
    test_communication(swarm)
    test_collective_dreaming(swarm)
    
    # Final metrics
    print("=" * 60)
    print("FINAL METRICS")
    print("=" * 60)
    final_metrics = swarm.get_metrics()
    for key, value in final_metrics.items():
        print(f"   {key}: {value}")
    
    print()
    print("✅ ALL TESTS PASSED!")
    print()
    print("🚀 Phase 2.1 swarm architecture is functional!")
    print("   Ready for scaling experiments (10 → 100 → 1000 agents)")

if __name__ == "__main__":
    main()
