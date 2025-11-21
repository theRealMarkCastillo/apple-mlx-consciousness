import json
import random
import argparse
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

def generate_synthetic_data(num_experiences=1000, output_file="synthetic_experiences.json", balanced=False, oversample_boundary=False):
    """
    Generates synthetic experiences where the optimal action depends on a simple rule.
    Rule: If state[0] > 0, Action 0 is correct. If state[0] <= 0, Action 1 is correct.
    """
    print(f"Generating {num_experiences} synthetic experiences...")
    experiences = []
    
    count_action_0 = 0
    count_action_1 = 0

    for i in range(num_experiences):
        # Create a random state vector (dim 128)
        state = np.random.randn(128).astype(np.float32)
        
        # Balanced Sampling Logic
        if balanced:
            # Force the decision variable (state[0]) to alternate or balance
            if i % 2 == 0:
                # Target Action 0 -> state[0] must be positive
                state[0] = abs(np.random.randn()) + 0.1 
            else:
                # Target Action 1 -> state[0] must be negative
                state[0] = -abs(np.random.randn()) - 0.1

        # Boundary Oversampling Logic
        if oversample_boundary:
            # 20% of samples are very close to 0
            if np.random.rand() < 0.2:
                state[0] = state[0] * 0.1 # Shrink magnitude to be near boundary

        # Determine correct action based on rule
        # Rule: state[0] > 0 -> Action 0, else Action 1
        correct_action = 0 if state[0] > 0 else 1
        
        if correct_action == 0:
            count_action_0 += 1
        else:
            count_action_1 += 1

        experience = {
            "state": state.tolist(),
            "action": correct_action,
            "reward": 1.0, # High reward for correct action
            "next_state": np.random.randn(128).tolist(), # Dummy next state
            "done": False
        }
        experiences.append(experience)
    
    print(f"Generated {len(experiences)} experiences.")
    print(f"Class Balance: Action 0: {count_action_0}, Action 1: {count_action_1}")
    
    with open(output_file, 'w') as f:
        json.dump(experiences, f)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training datasets for the MLX consciousness project")
    parser.add_argument("--convo-file", type=str, default="conversation_data.json",
                        help="Output path for conversation data JSON")
    parser.add_argument("--experiences-file", type=str, default="synthetic_experiences.json",
                        help="Output path for synthetic experiences JSON")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="Number of synthetic experiences to generate")
    parser.add_argument("--no-convo", action="store_true",
                        help="Skip generating conversation data")
    parser.add_argument("--no-experiences", action="store_true",
                        help="Skip generating synthetic experiences")
    parser.add_argument("--balanced", action="store_true", help="Ensure 50/50 split of positive/negative examples")
    parser.add_argument("--oversample-boundary", action="store_true", help="Generate more samples near the decision boundary")
    args = parser.parse_args()

    if not args.no_convo:
        generate_conversation_data(filename=args.convo_file)
    if not args.no_experiences:
        generate_synthetic_data(num_experiences=args.num_samples, output_file=args.experiences_file, balanced=args.balanced, oversample_boundary=args.oversample_boundary)
