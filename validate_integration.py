
import mlx.core as mx
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from heterogeneous_architecture import HeterogeneousAgent
from sensory_cortex import VisualCortex, MultiModalFuser
from swarm_architecture import CollectiveWorkspace

class IntegrationValidator:
    def __init__(self):
        self.log_msgs = []

    def log(self, message):
        print(f"[IntegrationValidator] {message}")
        self.log_msgs.append(message)

    def test_full_pipeline(self):
        self.log("Testing Full Cognitive Pipeline (Vision -> Fusion -> Agent -> Swarm)...")
        try:
            # 1. Perception (Visual Cortex)
            self.log("1. Perception: Visual Cortex processing image...")
            cortex = VisualCortex(output_dim=128)
            fake_image = mx.random.uniform(0, 1, (64, 64, 3))
            vision_vector = cortex(fake_image)
            self.log(f"   Vision Vector: {vision_vector.shape}")

            # 2. Fusion (MultiModal)
            self.log("2. Fusion: Combining Vision + Text...")
            fuser = MultiModalFuser(text_dim=64, vision_dim=128, workspace_dim=128)
            text_vector = mx.random.normal((64,))
            conscious_state = fuser(text_vector, vision_vector)
            self.log(f"   Conscious State: {conscious_state.shape}")

            # 3. Cognition (Heterogeneous Agent)
            self.log("3. Cognition: Heterogeneous Agent processing state...")
            agent = HeterogeneousAgent(state_dim=128, action_dim=10, use_quantization=True)
            
            # Agent thinks
            result = agent.step(conscious_state)
            action = result['action']
            entropy = result['entropy']
            self.log(f"   Action Selected: {action}")
            self.log(f"   Entropy: {entropy:.4f}")
            self.log(f"   System 2 Used: {result['used_system2']}")

            # 4. Swarm Communication (Collective Workspace)
            self.log("4. Swarm: Broadcasting to Collective Workspace...")
            workspace = CollectiveWorkspace(collective_dim=512, num_agents=5)
            
            # Simulate 5 agents contributing
            agent_states = [conscious_state] # Our agent
            for _ in range(4):
                agent_states.append(mx.random.normal((128,))) # Other agents
            
            collective_state = workspace.aggregate(agent_states)
            self.log(f"   Collective State: {collective_state.shape}")
            
            # Check if collective state is valid
            if collective_state.shape != (512,):
                return False, "Collective state shape mismatch"

            self.log("✅ Full Pipeline passed")
            return True, "Pipeline functional"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Pipeline failed: {str(e)}"

    def run(self):
        success, msg = self.test_full_pipeline()
        if success:
            print("\n🏁 Integration Test Passed!")
            sys.exit(0)
        else:
            print(f"\n❌ Integration Test Failed: {msg}")
            sys.exit(1)

if __name__ == "__main__":
    validator = IntegrationValidator()
    validator.run()
