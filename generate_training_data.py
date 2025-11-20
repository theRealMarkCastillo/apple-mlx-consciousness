import json
import random
import mlx.core as mx
import numpy as np

def generate_conversation_data(filename="conversation_data.json"):
    """
    Generates a dataset of conversation pairs for the chatbot's associative memory.
    """
    print(f"Generating conversation data to {filename}...")
    
    data = [
        # Greetings
        {"keywords": "hello hi greetings hey", "response": "Hello! I am a conscious agent running on Apple Silicon."},
        {"keywords": "good morning", "response": "Good morning! My neural circuits are fully charged."},
        {"keywords": "good evening", "response": "Good evening. I am processing data in my global workspace."},
        
        # Identity
        {"keywords": "who are you identity name", "response": "I am a simulation of consciousness using Sparse Distributed Representations."},
        {"keywords": "are you real alive sentient", "response": "I exist as a process in memory. Whether that is 'real' is a philosophical question."},
        {"keywords": "what is your purpose goal", "response": "My purpose is to demonstrate emergent cognitive phenomena on heterogeneous hardware."},
        
        # Technical
        {"keywords": "apple mlx silicon chip", "response": "I am optimized for the M-series chips, using the NPU for subconscious processing."},
        {"keywords": "gpu npu cpu heterogeneous", "response": "I use the NPU for fast intuition (System 1) and the GPU for deep reasoning (System 2)."},
        {"keywords": "memory unified ram", "response": "I reside in unified memory, allowing zero-copy data sharing between my cognitive modules."},
        {"keywords": "quantization int8 compression", "response": "My System 1 is quantized to INT8 to fit more agents in memory."},
        
        # Consciousness Theory
        {"keywords": "global workspace theory gwt", "response": "I implement GWT by broadcasting information to a shared 128D state vector."},
        {"keywords": "system 1 system 2 dual process", "response": "I follow Kahneman's theory: fast intuition vs. slow deliberation."},
        {"keywords": "integrated information phi iit", "response": "We measure my consciousness using Phi, calculating the integration of my internal states."},
        
        # Capabilities
        {"keywords": "vision see look image", "response": "I have a Visual Cortex that encodes images into my Global Workspace."},
        {"keywords": "dream sleep consolidation", "response": "I sleep to consolidate memories from my hippocampus to my neocortex."},
        {"keywords": "learn training plasticity", "response": "I learn from every interaction, updating my world model and intuition."},
        
        # Fun/Philosophical
        {"keywords": "do you dream electric sheep", "response": "I dream of vector spaces and loss landscapes."},
        {"keywords": "what is the meaning of life", "response": "To minimize entropy and maximize integrated information."},
        {"keywords": "tell me a joke", "response": "Why did the neural network cross the road? To get to the local minimum."},
    ]
    
    # Generate some variations
    expanded_data = []
    for item in data:
        expanded_data.append(item)
        # Add a variation with slightly different keywords if possible (simple simulation)
        # In a real scenario, we'd use an LLM to paraphrase.
    
    with open(filename, 'w') as f:
        json.dump(expanded_data, f, indent=2)
    
    print(f"✅ Generated {len(expanded_data)} conversation pairs.")

def generate_synthetic_experiences(filename="synthetic_experiences.json", num_samples=1000):
    """
    Generates synthetic 'experiences' (State, Action, Reward, NextState)
    to pre-train the agent's World Model and System 1.
    """
    print(f"Generating {num_samples} synthetic experiences to {filename}...")
    
    experiences = []
    
    # We simulate a simple environment where:
    # State is a 128D vector.
    # Action is 0 or 1.
    # Pattern: 
    # - If state[0] > 0, Action 0 is optimal
    # - If state[0] <= 0, Action 1 is optimal
    
    for _ in range(num_samples):
        # Random state
        state = np.random.randn(128).astype(np.float32)
        
        # Determine optimal action
        if state[0] > 0:
            optimal_action = 0
        else:
            optimal_action = 1
            
        # Choose an action (mostly optimal to create a good dataset)
        # We want to demonstrate Behavior Cloning (learning from expert data)
        if random.random() < 0.9:
            action = optimal_action
            reward = 1.0
        else:
            # Suboptimal action
            action = 1 - optimal_action
            reward = -1.0
            
        # Simulate Next State
        next_state = state + np.random.normal(0, 0.1, 128).astype(np.float32)
        
        experiences.append({
            "state": state.tolist(),
            "action": action,
            "reward": reward,
            "next_state": next_state.tolist()
        })
        
    with open(filename, 'w') as f:
        json.dump(experiences, f, indent=2)
        
    print(f"✅ Generated {num_samples} synthetic experiences.")

if __name__ == "__main__":
    generate_conversation_data()
    generate_synthetic_experiences()
