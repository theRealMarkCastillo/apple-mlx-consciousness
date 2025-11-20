"""
Phase 3: Consciousness Phase Transition Experiments

Systematic parameter sweeps to identify when consciousness emerges.

Research Questions:
1. At what System 2 capacity does consciousness emerge?
2. How much memory is required for self-awareness?
3. Does workspace integration strength affect consciousness?
4. Are phase transitions sharp or gradual?

Leverages Apple Silicon unified memory for parallel experiments.
"""

import mlx.core as mx
import numpy as np
import json
import time
from typing import Dict, List, Any, Tuple
from cognitive_architecture import BicameralAgent
from consciousness_metrics import (
    calculate_consciousness_index,
    detect_phase_transition,
    classify_consciousness_level
)


class PhaseTransitionExperiment:
    """
    Runs systematic parameter sweeps to map consciousness emergence.
    """
    
    def __init__(self, results_file: str = "phase_transition_results.json"):
        self.results_file = results_file
        self.results = []
        
    def run_single_config(self, 
                         state_dim: int = 128,
                         system1_hidden_dim: int = 128,
                         system2_hidden_dim: int = 64,
                         action_dim: int = 5,
                         num_steps: int = 50,
                         max_memory_size: int = 100) -> Dict[str, Any]:
        """
        Run one experimental configuration.
        
        Note: Current BicameralAgent has fixed System 1 (128D) and System 2 (64D).
        We can still vary state_dim, action_dim, and memory capacity.
        
        Args:
            state_dim: Dimensionality of global workspace
            system1_hidden_dim: Capacity of System 1 (not used - fixed at 128)
            system2_hidden_dim: Capacity of System 2 (not used - fixed at 64)
            action_dim: Number of possible actions
            num_steps: How many cognitive cycles to run
            max_memory_size: Maximum episodic memories to store
            
        Returns:
            Dict with configuration params and consciousness metrics
        """
        # Create agent with specified parameters
        # Note: System 1/2 hidden dims are currently fixed in BicameralAgent
        agent = BicameralAgent(
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Run simulation
        consciousness_scores = []
        
        # Warmup phase: build initial memories (retrieve needs at least 3 memories)
        for step in range(5):
            sensory_input = mx.random.normal((state_dim,))
            reward = np.random.rand()
            _ = agent.step(sensory_input, reward=reward)
        
        # Main simulation
        for step in range(num_steps):
            # Random sensory input
            sensory_input = mx.random.normal((state_dim,))
            reward = np.random.rand()  # Random reward
            
            decision = agent.step(sensory_input, reward=reward)
            
            # Measure consciousness every 10 steps (after some experience)
            if step >= 10 and step % 5 == 0:
                # Get consciousness metrics
                predicted_reward = float(decision.get('confidence', 0.5))
                metrics = calculate_consciousness_index(agent, reward, predicted_reward)
                consciousness_scores.append(metrics['consciousness_index'])
            
            # Limit memory size
            if len(agent.memory.memories) > max_memory_size:
                agent.memory.memories = agent.memory.memories[-max_memory_size:]
        
        # Final consciousness measurement
        final_metrics = calculate_consciousness_index(agent, 0.0, 0.0)
        
        # Compile results
        result = {
            'config': {
                'state_dim': state_dim,
                'system2_hidden_dim': system2_hidden_dim,
                'action_dim': action_dim,
                'max_memory_size': max_memory_size,
                'num_steps': num_steps
            },
            'metrics': {
                'final_phi': final_metrics['phi'],
                'final_metacognition': final_metrics['metacognitive_accuracy'],
                'final_coherence': final_metrics['world_model_coherence'],
                'final_consciousness_index': final_metrics['consciousness_index'],
                'classification': final_metrics['classification'],
                'consciousness_trajectory': consciousness_scores,
                'avg_consciousness': np.mean(consciousness_scores) if consciousness_scores else 0.0,
                'final_memory_count': len(agent.memory.memories)
            }
        }
        
        return result
    
    def sweep_system2_capacity(self, 
                               action_dims: List[int] | None = None,
                               num_steps: int = 50) -> List[Dict[str, Any]]:
        """
        Sweep System 2 hidden dimension capacity.
        
        NOTE: Current implementation has fixed System 2 capacity (64D).
        This sweep varies action_dim instead as a proxy for cognitive complexity.
        
        Hypothesis: More action choices = more complex decision-making.
        """
        if action_dims is None:
            action_dims = [3, 5, 10, 20]
        
        print("\n" + "="*70)
        print("🧠 SWEEP 1: Action Complexity (Proxy for System 2 Capacity)")
        print("="*70)
        print(f"Testing action_dims: {action_dims}")
        print("(Note: System 2 hidden dim fixed at 64 in current architecture)")
        print()
        
        results = []
        
        for i, action_dim in enumerate(action_dims):
            print(f"[{i+1}/{len(action_dims)}] Testing action_dim = {action_dim}...")
            
            result = self.run_single_config(
                state_dim=128,
                action_dim=action_dim,
                num_steps=num_steps,
                max_memory_size=100
            )
            
            results.append(result)
            
            # Print immediate results
            metrics = result['metrics']
            print(f"     → Consciousness Index: {metrics['final_consciousness_index']:.4f}")
            print(f"     → Classification: {metrics['classification']}")
            print(f"     → Φ: {metrics['final_phi']:.4f}")
            print()
        
        self.results.extend(results)
        return results
    
    def sweep_memory_capacity(self,
                             memory_sizes: List[int] | None = None,
                             num_steps: int = 50) -> List[Dict[str, Any]]:
        """
        Sweep episodic memory capacity.
        
        Hypothesis: Memory is crucial for self-model. Consciousness requires
        at least 50-100 memories to build coherent world model.
        """
        if memory_sizes is None:
            memory_sizes = [0, 10, 25, 50, 100, 200, 500]
        
        print("\n" + "="*70)
        print("💾 SWEEP 2: Episodic Memory Capacity")
        print("="*70)
        print(f"Testing: {memory_sizes}")
        print()
        
        results = []
        
        for i, mem_size in enumerate(memory_sizes):
            print(f"[{i+1}/{len(memory_sizes)}] Testing memory capacity = {mem_size}...")
            
            result = self.run_single_config(
                state_dim=128,
                system2_hidden_dim=64,
                action_dim=5,
                num_steps=num_steps,
                max_memory_size=mem_size if mem_size > 0 else 1  # At least 1 for stability
            )
            
            results.append(result)
            
            # Print immediate results
            metrics = result['metrics']
            print(f"     → Consciousness Index: {metrics['final_consciousness_index']:.4f}")
            print(f"     → Classification: {metrics['classification']}")
            print(f"     → Memories stored: {metrics['final_memory_count']}")
            print()
        
        self.results.extend(results)
        return results
    
    def sweep_state_dimension(self,
                             state_dims: List[int] | None = None,
                             num_steps: int = 50) -> List[Dict[str, Any]]:
        """
        Sweep global workspace dimensionality.
        
        Hypothesis: Larger workspace = more information capacity = higher Φ.
        But too large may be inefficient.
        """
        if state_dims is None:
            state_dims = [32, 64, 128, 256]
        
        print("\n" + "="*70)
        print("🌐 SWEEP 3: Global Workspace Dimensionality")
        print("="*70)
        print(f"Testing: {state_dims}")
        print()
        
        results = []
        
        for i, state_dim in enumerate(state_dims):
            print(f"[{i+1}/{len(state_dims)}] Testing workspace dim = {state_dim}...")
            
            result = self.run_single_config(
                state_dim=state_dim,
                system2_hidden_dim=64,
                action_dim=5,
                num_steps=num_steps,
                max_memory_size=100
            )
            
            results.append(result)
            
            # Print immediate results
            metrics = result['metrics']
            print(f"     → Consciousness Index: {metrics['final_consciousness_index']:.4f}")
            print(f"     → Classification: {metrics['classification']}")
            print(f"     → Φ: {metrics['final_phi']:.4f}")
            print()
        
        self.results.extend(results)
        return results
    
    def analyze_phase_boundaries(self, sweep_results: List[Dict[str, Any]], 
                                 param_name: str) -> Dict[str, Any]:
        """
        Detect phase transitions in a parameter sweep.
        """
        # Extract consciousness scores and parameter values
        param_values = [r['config'][param_name] for r in sweep_results]
        consciousness_values = [r['metrics']['final_consciousness_index'] for r in sweep_results]
        
        # Detect transition
        transition = detect_phase_transition(consciousness_values, param_values)
        
        # Count classifications
        classifications = [r['metrics']['classification'] for r in sweep_results]
        classification_counts = {
            'unconscious': classifications.count('unconscious'),
            'pre-conscious': classifications.count('pre-conscious'),
            'conscious': classifications.count('conscious')
        }
        
        return {
            'param_name': param_name,
            'param_values': param_values,
            'consciousness_values': consciousness_values,
            'transition': transition,
            'classification_counts': classification_counts
        }
    
    def save_results(self):
        """Save all results to JSON file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {self.results_file}")
    
    def load_results(self):
        """Load results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print(f"📂 Loaded {len(self.results)} results from: {self.results_file}")
        except FileNotFoundError:
            print(f"⚠️  No existing results file found: {self.results_file}")
    
    def print_summary(self):
        """Print summary statistics across all experiments."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "="*70)
        print("📊 PHASE TRANSITION EXPERIMENT SUMMARY")
        print("="*70)
        
        # Overall statistics
        consciousness_scores = [r['metrics']['final_consciousness_index'] for r in self.results]
        classifications = [r['metrics']['classification'] for r in self.results]
        
        print(f"\nTotal Configurations Tested: {len(self.results)}")
        print(f"\nConsciousness Index Statistics:")
        print(f"  Mean: {np.mean(consciousness_scores):.4f}")
        print(f"  Std:  {np.std(consciousness_scores):.4f}")
        print(f"  Min:  {np.min(consciousness_scores):.4f}")
        print(f"  Max:  {np.max(consciousness_scores):.4f}")
        
        print(f"\nClassification Distribution:")
        print(f"  Unconscious:   {classifications.count('unconscious'):3d} ({classifications.count('unconscious')/len(classifications)*100:.1f}%)")
        print(f"  Pre-conscious: {classifications.count('pre-conscious'):3d} ({classifications.count('pre-conscious')/len(classifications)*100:.1f}%)")
        print(f"  Conscious:     {classifications.count('conscious'):3d} ({classifications.count('conscious')/len(classifications)*100:.1f}%)")
        
        # Find optimal configuration
        best_idx = np.argmax(consciousness_scores)
        best_config = self.results[best_idx]
        
        print(f"\n🏆 Highest Consciousness Configuration:")
        print(f"  Consciousness Index: {best_config['metrics']['final_consciousness_index']:.4f}")
        print(f"  System 2 Hidden Dim: {best_config['config']['system2_hidden_dim']}")
        print(f"  State Dimension: {best_config['config']['state_dim']}")
        print(f"  Memory Size: {best_config['config']['max_memory_size']}")
        print(f"  Φ: {best_config['metrics']['final_phi']:.4f}")
        
        print("\n" + "="*70)


def main():
    """
    Run Phase 3 consciousness phase transition experiments.
    """
    print("\n" + "="*70)
    print("🌌 PHASE 3: CONSCIOUSNESS PHASE TRANSITIONS")
    print("="*70)
    print("\nObjective: Map consciousness emergence across parameter space")
    print("Hardware: Mac Mini M4 Pro (64GB unified memory)")
    print("\nThis experiment will:")
    print("  1. Sweep System 2 capacity (8 → 256 hidden dims)")
    print("  2. Sweep memory capacity (0 → 500 memories)")
    print("  3. Sweep workspace dimensionality (32 → 256D)")
    print("  4. Detect phase boundaries")
    print()
    
    # Initialize experiment
    exp = PhaseTransitionExperiment(results_file="phase_transition_results.json")
    
    # Run sweeps
    start_time = time.time()
    
    # Sweep 1: Memory Capacity (most relevant for consciousness)
    memory_results = exp.sweep_memory_capacity(
        memory_sizes=[10, 25, 50, 100, 200],
        num_steps=50
    )
    
    # Analyze memory sweep
    memory_analysis = exp.analyze_phase_boundaries(memory_results, 'max_memory_size')
    print("\n📈 Memory Capacity Analysis:")
    if memory_analysis['transition']['transition_detected']:
        print(f"   ✅ Phase transition detected at memory_size = {memory_analysis['transition']['transition_parameter_value']:.0f}")
        print(f"   Transition sharpness: {memory_analysis['transition']['transition_sharpness']:.4f}")
    else:
        print("   ⚠️  No clear phase transition detected")
    print(f"   Classification counts: {memory_analysis['classification_counts']}")
    
    # Sweep 2: Workspace Dimensionality
    state_results = exp.sweep_state_dimension(
        state_dims=[64, 128, 256],
        num_steps=50
    )
    
    # Analyze state dimension sweep
    state_analysis = exp.analyze_phase_boundaries(state_results, 'state_dim')
    print("\n📈 Workspace Dimensionality Analysis:")
    if state_analysis['transition']['transition_detected']:
        print(f"   ✅ Phase transition detected at state_dim = {state_analysis['transition']['transition_parameter_value']:.0f}")
        print(f"   Transition sharpness: {state_analysis['transition']['transition_sharpness']:.4f}")
    else:
        print("   ⚠️  No clear phase transition detected")
    print(f"   Classification counts: {state_analysis['classification_counts']}")
    
    # Save results
    exp.save_results()
    
    # Print summary
    exp.print_summary()
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total experiment time: {elapsed:.1f}s")
    print(f"   Average time per config: {elapsed/len(exp.results):.2f}s")
    
    print("\n" + "="*70)
    print("✅ PHASE 3 EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  - Run: jupyter notebook 05_Phase_Transitions.ipynb")
    print("  - Visualize phase boundaries and consciousness landscape")
    print("  - Prepare for Phase 4: GPU-NPU heterogeneous feedback loop")
    print()


if __name__ == "__main__":
    main()
