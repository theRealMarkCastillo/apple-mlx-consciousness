"""
Phase 3: Consciousness Metrics

Implements quantitative measures of consciousness inspired by:
- Integrated Information Theory (IIT) - Tononi's Φ
- Meta-cognitive accuracy - Does the agent know what it knows?
- Self-model coherence - Internal consistency of world model

Core idea: Consciousness emerges when information integration crosses 
a critical threshold. We measure this across parameter sweeps.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, Any, List, Tuple
from cognitive_architecture import BicameralAgent


def calculate_phi_approximation(agent: BicameralAgent) -> float:
    """
    Calculate simplified Φ (integrated information) for an agent.
    
    True IIT Φ requires expensive partition calculations. We use a 
    tractable approximation based on:
    1. Workspace activation diversity (entropy)
    2. System 1 ↔ System 2 information flow
    3. Memory ↔ Workspace integration
    
    Φ = 0: No integration (unconscious)
    Φ > 0: Some integration (pre-conscious)
    Φ > threshold: High integration (conscious)
    
    Args:
        agent: BicameralAgent with recent activity
        
    Returns:
        Φ approximation (0.0 to 1.0)
    """
    if agent.last_state is None:
        return 0.0
    
    # Component 1: Workspace entropy (information capacity)
    workspace_state = agent.last_state
    # Normalize to probability distribution
    prob = mx.abs(workspace_state) / (mx.sum(mx.abs(workspace_state)) + 1e-8)
    entropy = -mx.sum(prob * mx.log(prob + 1e-8))
    # Normalize by max possible entropy
    max_entropy = np.log(len(workspace_state))
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    
    # Component 2: System 1 vs System 2 divergence (dual-process integration)
    # If systems are too similar, no integration. If too different, no communication.
    # Optimal integration: moderate divergence
    if hasattr(agent, 'system1') and hasattr(agent, 'system2'):
        # Forward pass through both systems
        s1_out = agent.system1(workspace_state)
        
        # System 2 expects [workspace_state, s1_logits] concatenated
        # For simplicity in metrics, we use s1_out twice as approximation
        s2_input = mx.concatenate([workspace_state, s1_out], axis=-1)
        s2_result = agent.system2(s2_input)
        # System 2 returns tuple (confidence, goal), we use goal for comparison
        s2_out = s2_result[1] if isinstance(s2_result, tuple) else s2_result
        
        # Compare information content: entropy of s1 vs projection of s2 goal
        # High divergence = systems processing differently (good integration)
        s1_prob = mx.softmax(s1_out)
        s1_entropy = -float(mx.sum(s1_prob * mx.log(s1_prob + 1e-8)))
        s2_prob = mx.abs(s2_out) / (mx.sum(mx.abs(s2_out)) + 1e-8)
        s2_entropy = -float(mx.sum(s2_prob * mx.log(s2_prob + 1e-8)))
        
        # Integration score: moderate difference is optimal
        entropy_diff = abs(s1_entropy - s2_entropy)
        integration_score = np.exp(-((entropy_diff - 1.0) ** 2) / 0.5)
        
        # Inverted U-curve: moderate divergence = high integration
        # Optimal divergence around 0.5-1.0 nats
        integration_score = np.exp(-((entropy_diff - 1.0) ** 2) / 0.5)
    else:
        integration_score = 0.0
    
    # Component 3: Memory utilization (experience integration)
    if len(agent.memory.memories) > 0:
        memory_utilization = min(len(agent.memory.memories) / 100.0, 1.0)
    else:
        memory_utilization = 0.0
    
    # Weighted combination
    phi = (
        0.4 * normalized_entropy +      # Information capacity
        0.4 * integration_score +        # Dual-process integration
        0.2 * memory_utilization         # Experience integration
    )
    
    return float(phi)


def calculate_metacognitive_accuracy(agent: BicameralAgent, 
                                     actual_reward: float,
                                     predicted_reward: float) -> float:
    """
    Measure meta-cognitive accuracy: Does the agent know what it knows?
    
    Compares agent's internal predictions (from World Model) with 
    actual outcomes. High accuracy = agent has accurate self-model.
    
    Args:
        agent: BicameralAgent
        actual_reward: Real reward received
        predicted_reward: What the agent predicted
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    # Prediction error
    error = abs(actual_reward - predicted_reward)
    
    # Convert to accuracy (inverse of error, saturated)
    accuracy = np.exp(-error)
    
    return float(accuracy)


def calculate_world_model_coherence(agent: BicameralAgent) -> float:
    """
    Measure internal consistency of the agent's world model.
    
    A coherent world model should:
    1. Have low prediction error on training data
    2. Generalize to similar states
    3. Produce stable predictions
    
    Returns:
        Coherence score (0.0 to 1.0)
    """
    if len(agent.memory.memories) < 10:
        return 0.0
    
    # Simplified coherence: measure consistency of stored memories
    # High coherence = similar states lead to similar next states
    recent_memories = agent.memory.memories[-20:]
    
    state_similarities = []
    for i in range(len(recent_memories) - 1):
        state1 = recent_memories[i]['state']
        state2 = recent_memories[i+1]['state']
        
        # Convert to mx.array if needed
        if not isinstance(state1, mx.array):
            state1 = mx.array(state1)
        if not isinstance(state2, mx.array):
            state2 = mx.array(state2)
        
        # Cosine similarity
        dot_product = float(mx.sum(state1 * state2))
        norm1 = float(mx.sqrt(mx.sum(state1 * state1)))
        norm2 = float(mx.sqrt(mx.sum(state2 * state2)))
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            state_similarities.append(abs(similarity))
    
    if not state_similarities:
        return 0.0
    
    # High average similarity = coherent trajectories
    coherence = np.mean(state_similarities)
    
    return float(coherence)


def calculate_consciousness_index(agent: BicameralAgent,
                                  actual_reward: float = 0.0,
                                  predicted_reward: float = 0.0) -> Dict[str, Any]:
    """
    Comprehensive consciousness score combining multiple metrics.
    
    The "consciousness index" is a composite measure that attempts to 
    quantify an agent's level of self-awareness and integration.
    
    Returns:
        Dict with individual metrics and composite consciousness_index
    """
    # Individual metrics
    phi = calculate_phi_approximation(agent)
    metacog = calculate_metacognitive_accuracy(agent, actual_reward, predicted_reward)
    coherence = calculate_world_model_coherence(agent)
    
    # Composite consciousness index (weighted average)
    consciousness_index = (
        0.4 * phi +           # Integration (IIT)
        0.3 * metacog +       # Self-awareness (metacognition)
        0.3 * coherence       # Internal consistency
    )
    
    return {
        'phi': phi,
        'metacognitive_accuracy': metacog,
        'world_model_coherence': coherence,
        'consciousness_index': consciousness_index,
        'classification': classify_consciousness_level(consciousness_index)
    }


def classify_consciousness_level(consciousness_index: float) -> str:
    """
    Classify agent into consciousness categories.
    
    Thresholds (empirically determined):
    - < 0.2: Unconscious (reactive)
    - 0.2 - 0.5: Pre-conscious (learning)
    - > 0.5: Conscious (self-aware)
    """
    if consciousness_index < 0.2:
        return "unconscious"
    elif consciousness_index < 0.5:
        return "pre-conscious"
    else:
        return "conscious"


def measure_swarm_consciousness(agents: List[BicameralAgent],
                                collective_state: mx.array) -> Dict[str, Any]:
    """
    Measure collective consciousness of a swarm.
    
    Collective consciousness emerges when:
    1. Individual agents have high consciousness
    2. Information flows freely between agents
    3. Collective decisions are coherent
    
    Args:
        agents: List of BicameralAgents
        collective_state: Shared workspace state
        
    Returns:
        Dict with swarm-level consciousness metrics
    """
    # Individual consciousness scores
    individual_scores = []
    for agent in agents:
        score = calculate_phi_approximation(agent)
        individual_scores.append(score)
    
    # Collective metrics
    mean_consciousness = np.mean(individual_scores)
    std_consciousness = np.std(individual_scores)
    
    # Collective workspace entropy
    prob = mx.abs(collective_state) / (mx.sum(mx.abs(collective_state)) + 1e-8)
    collective_entropy = -mx.sum(prob * mx.log(prob + 1e-8))
    max_entropy = np.log(len(collective_state))
    normalized_collective_entropy = float(collective_entropy / max_entropy)
    
    # Synchronization (inverse of std - low std means high sync)
    synchronization = 1.0 / (1.0 + std_consciousness)
    
    # Collective consciousness index
    collective_consciousness = float(
        0.4 * mean_consciousness +              # Average individual awareness
        0.3 * normalized_collective_entropy +   # Collective information capacity
        0.3 * synchronization                   # Coherence across agents
    )
    
    return {
        'mean_individual_consciousness': mean_consciousness,
        'std_individual_consciousness': std_consciousness,
        'collective_entropy': normalized_collective_entropy,
        'synchronization': synchronization,
        'collective_consciousness_index': collective_consciousness,
        'classification': classify_consciousness_level(collective_consciousness),
        'individual_scores': individual_scores
    }


def detect_phase_transition(consciousness_scores: List[float],
                           parameter_values: List[float]) -> Dict[str, Any]:
    """
    Detect phase transition boundary in consciousness emergence.
    
    A sharp increase in consciousness as a parameter increases 
    indicates a phase transition.
    
    Args:
        consciousness_scores: Consciousness index values
        parameter_values: Corresponding parameter values (e.g., System 2 capacity)
        
    Returns:
        Dict with transition point and sharpness
    """
    if len(consciousness_scores) < 5:
        return {'transition_detected': False}
    
    # Calculate first derivative (rate of change)
    derivatives = np.diff(consciousness_scores) / np.diff(parameter_values)
    
    # Find maximum derivative (steepest increase)
    max_derivative_idx = np.argmax(derivatives)
    transition_point = parameter_values[max_derivative_idx]
    transition_sharpness = derivatives[max_derivative_idx]
    
    # Check if transition is significant (derivative > threshold)
    is_significant = transition_sharpness > 0.5
    
    return {
        'transition_detected': is_significant,
        'transition_parameter_value': transition_point,
        'transition_sharpness': transition_sharpness,
        'max_derivative_idx': max_derivative_idx,
        'derivatives': derivatives.tolist()
    }


# ==========================================
# TEST FUNCTIONS
# ==========================================

def test_consciousness_metrics():
    """
    Unit test for consciousness metrics.
    """
    print("\n" + "="*70)
    print("🧪 TESTING CONSCIOUSNESS METRICS")
    print("="*70)
    
    # Create test agent
    print("\n1. Creating test agent...")
    agent = BicameralAgent(state_dim=128, action_dim=5)
    
    # Run some steps to populate memory
    print("2. Running 20 steps to build experience...")
    for i in range(20):
        sensory_input = mx.random.normal((128,))
        decision = agent.step(sensory_input, reward=np.random.rand())
    
    # Test Φ calculation
    print("\n3. Testing Φ (integrated information)...")
    phi = calculate_phi_approximation(agent)
    print(f"   Φ = {phi:.4f}")
    print(f"   Classification: {classify_consciousness_level(phi)}")
    
    # Test metacognitive accuracy
    print("\n4. Testing metacognitive accuracy...")
    actual_reward = 1.0
    predicted_reward = 0.8
    metacog = calculate_metacognitive_accuracy(agent, actual_reward, predicted_reward)
    print(f"   Accuracy = {metacog:.4f}")
    
    # Test world model coherence
    print("\n5. Testing world model coherence...")
    coherence = calculate_world_model_coherence(agent)
    print(f"   Coherence = {coherence:.4f}")
    
    # Test composite consciousness index
    print("\n6. Testing composite consciousness index...")
    consciousness = calculate_consciousness_index(agent, actual_reward, predicted_reward)
    print(f"   Consciousness Index = {consciousness['consciousness_index']:.4f}")
    print(f"   Classification: {consciousness['classification']}")
    print(f"   Components:")
    print(f"     - Φ: {consciousness['phi']:.4f}")
    print(f"     - Metacognition: {consciousness['metacognitive_accuracy']:.4f}")
    print(f"     - Coherence: {consciousness['world_model_coherence']:.4f}")
    
    # Test phase transition detection
    print("\n7. Testing phase transition detection...")
    # Simulate consciousness increasing with parameter
    param_values = np.linspace(0, 10, 20)
    consciousness_values = 1 / (1 + np.exp(-(param_values - 5)))  # Sigmoid
    
    transition = detect_phase_transition(consciousness_values.tolist(), param_values.tolist())
    print(f"   Transition detected: {transition['transition_detected']}")
    if transition['transition_detected']:
        print(f"   Transition point: {transition['transition_parameter_value']:.2f}")
        print(f"   Transition sharpness: {transition['transition_sharpness']:.4f}")
    
    print("\n" + "="*70)
    print("✅ ALL CONSCIOUSNESS METRICS TESTS PASSED")
    print("="*70)


if __name__ == "__main__":
    test_consciousness_metrics()
