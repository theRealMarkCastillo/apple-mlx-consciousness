import mlx.core as mx
import time
from cognitive_architecture import BicameralAgent

def run_simulation(steps=20):
    print("🧠 Initializing Bicameral Agent (System 1 + System 2)...")
    agent = BicameralAgent(state_dim=128, action_dim=5)
    
    print(f"🚀 Starting Stream of Consciousness Simulation ({steps} steps)...\n")
    print(f"{'STEP':<5} | {'ACTION':<8} | {'CONFIDENCE':<12} | {'SURPRISE':<10} | {'ENTROPY':<10} | {'INTERVENTION'}")
    print("-" * 80)

    history = []

    for t in range(steps):
        # Simulate random sensory input (e.g., vision/sound vector)
        # In a real experiment, this would come from an environment
        sensory_input = mx.random.normal((128,))
        
        # Agent thinks
        # We provide a dummy reward of 0.0 for now, or random reward to test learning
        reward = 0.0
        if t > 0:
            # Simple mock reward: if previous action was 0, reward 1.0
            reward = 1.0 if history[-1]['action'] == 0 else -0.1

        decision = agent.step(sensory_input, reward=reward)
        
        # Log
        # Check if memory was retrieved (System 2 intervention)
        intervention_mark = "🔴 RECALL" if decision["memory_retrieved"] else "🟢 Flow"
        
        conf = decision['confidence'].item()
        ent = decision['entropy'].item()
        surp = decision['surprise']
        
        print(f"{t+1:<5} | {decision['action']:<8} | {conf:.4f}       | {surp:.4f}     | {ent:.4f}     | {intervention_mark}")
        
        history.append(decision)
        
        # Simulate time for thought
        time.sleep(0.05)

    print("\n✅ Simulation Complete.")
    
    # Trigger Dreaming Cycle
    print("\n💤 Entering REM Sleep (Dreaming & Consolidation)...")
    agent.dream(batch_size=10, epochs=5)
    
    return history

if __name__ == "__main__":
    run_simulation()
