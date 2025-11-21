
import mlx.core as mx
import numpy as np
import sys
import os
import time

# Add current directory to path
sys.path.append(os.getcwd())

from heterogeneous_architecture import HeterogeneousAgent
from sensory_cortex import VisualCortex, MultiModalFuser

class AdvancedValidator:
    def __init__(self):
        self.results = []

    def log(self, message):
        print(f"[AdvancedValidator] {message}")

    def test_heterogeneous_agent(self):
        self.log("Testing Heterogeneous Agent (GPU/NPU)...")
        try:
            agent = HeterogeneousAgent(state_dim=128, action_dim=10, use_quantization=True)
            
            # 1. Test Step (System 1)
            sensory = mx.random.normal((128,))
            result = agent.step(sensory)
            
            if 'action' not in result or 'entropy' not in result:
                return False, "Missing keys in step result"
            
            # 2. Test System 2 Trigger (Force high entropy/uncertainty if possible, or just check logic)
            # It's hard to force entropy without controlling weights, but we can check if the code runs.
            # The agent.step() logic handles the switch.
            
            # 3. Test Sleep/Consolidation
            # Add some dummy experiences
            for _ in range(15):
                agent.step(mx.random.normal((128,)), reward=1.0)
            
            sleep_stats = agent.sleep(epochs=1)
            if sleep_stats['consolidated'] == 0:
                return False, "Sleep did not consolidate experiences"
                
            self.log("✅ Heterogeneous Agent passed")
            return True, "Heterogeneous Agent functional"
        except Exception as e:
            return False, f"Heterogeneous Agent failed: {str(e)}"

    def test_visual_cortex(self):
        self.log("Testing Visual Cortex...")
        try:
            cortex = VisualCortex(output_dim=128)
            
            # Create random image [64, 64, 3]
            # MLX expects arrays, standard image format
            fake_image = mx.random.uniform(0, 1, (64, 64, 3))
            
            output = cortex(fake_image)
            
            if output.shape != (128,):
                return False, f"Output shape mismatch: {output.shape} != (128,)"
                
            self.log("✅ Visual Cortex passed")
            return True, "Visual Cortex functional"
        except Exception as e:
            return False, f"Visual Cortex failed: {str(e)}"

    def test_multimodal_fusion(self):
        self.log("Testing MultiModal Fusion...")
        try:
            text_dim = 64
            vision_dim = 128
            workspace_dim = 256
            
            fuser = MultiModalFuser(text_dim, vision_dim, workspace_dim)
            
            text_vec = mx.random.normal((text_dim,))
            vision_vec = mx.random.normal((vision_dim,))
            
            fused = fuser(text_vec, vision_vec)
            
            if fused.shape != (workspace_dim,):
                return False, f"Fused shape mismatch: {fused.shape} != ({workspace_dim},)"
                
            self.log("✅ MultiModal Fusion passed")
            return True, "MultiModal Fusion functional"
        except Exception as e:
            return False, f"MultiModal Fusion failed: {str(e)}"

    def run_all(self):
        tests = [
            self.test_heterogeneous_agent,
            self.test_visual_cortex,
            self.test_multimodal_fusion
        ]
        
        passed = 0
        for test in tests:
            success, msg = test()
            status = "PASS" if success else "FAIL"
            print(f"{status}: {msg}")
            if success:
                passed += 1
        
        print(f"\nAdvanced Validation Complete: {passed}/{len(tests)} Tests Passed")
        if passed == len(tests):
            sys.exit(0)
        else:
            sys.exit(1)

if __name__ == "__main__":
    validator = AdvancedValidator()
    validator.run_all()
