import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib.pyplot as plt
from heterogeneous_architecture import HeterogeneousAgent

class AdversarialEvolution:
    """
    Implements Option B: Adversarial Co-Evolution.
    
    Mechanism:
    1. Adversarial Miner: Finds inputs 'x' that maximize disagreement between System 1 and System 2.
       Maximize KL(System1(x) || System2(x))
    2. Student (System 1): Trains to minimize this disagreement.
    3. Teacher (System 2): (In a full setup, would also evolve, here acts as the stable expert).
    """
    
    def __init__(self):
        self.agent = HeterogeneousAgent(use_quantization=True)
        self.optimizer = optim.Adam(learning_rate=0.001)
        
    def compute_disagreement(self, x):
        """Compute KL Divergence between System 1 and System 2."""
        # We use the FP32 version of System 1 for gradient computation w.r.t input
        # This approximates the disagreement with the quantized version
        logits1 = self.agent.system1.l2(mx.tanh(self.agent.system1.l1(x)))
        logits2, _ = self.agent.system2(x)
        # System 2 returns (goal, confidence), goal is used as logits for action modulation
        # In the agent step, sys2 goal modulates action probs. 
        # For direct comparison, let's assume System 2's goal vector *is* the logit distribution for actions (first 10 dims)
        logits2 = logits2[:, :self.agent.action_dim]
        
        log_probs1 = nn.log_softmax(logits1, axis=-1)
        probs2 = mx.softmax(logits2, axis=-1)
        
        # KL(P || Q) = sum(p * (log p - log q))
        # Here we want System 1 to match System 2, so minimize KL(Sys2 || Sys1)
        # But for finding adversarial examples, we want to MAXIMIZE it.
        
        # Let's use MSE of logits for stability in this demo, or KL.
        # KL is better for probabilities.
        
        log_probs2 = nn.log_softmax(logits2, axis=-1)
        
        # KL(Sys2 || Sys1)
        kl = mx.sum(probs2 * (log_probs2 - log_probs1), axis=-1)
        return mx.mean(kl)

    def mine_adversarial_examples(self, batch_size=32, steps=10, step_size=0.1):
        """
        Find inputs that maximize disagreement using gradient ascent.
        """
        # Start with random noise
        x = mx.random.normal((batch_size, self.agent.state_dim))
        
        def loss_fn(input_batch):
            return -self.compute_disagreement(input_batch) # Minimize negative disagreement = Maximize disagreement
        
        grad_fn = mx.grad(loss_fn)
        
        for _ in range(steps):
            grads = grad_fn(x)
            # Gradient ascent: x = x + step_size * grad (since we minimized negative)
            # The grad is of (-disagreement), so moving in direction of grad minimizes (-disagreement) -> maximizes disagreement
            # Wait, grad_fn gives gradient of loss. We want to decrease loss.
            # Loss is -Disagreement. Decreasing -Disagreement means Increasing Disagreement.
            # So we subtract the gradient: x = x - lr * grad
            x = x - step_size * grads
            
        return x

    def train_step(self, inputs):
        """Train System 1 to match System 2 on these inputs."""
        
        def loss_fn(model):
            # System 1 output
            logits1 = model.l2(mx.tanh(model.l1(inputs)))
            log_probs1 = nn.log_softmax(logits1, axis=-1)
            
            # System 2 target (detached, we don't train Sys 2 here)
            logits2, _ = self.agent.system2(inputs)
            logits2 = logits2[:, :self.agent.action_dim]
            probs2 = mx.softmax(logits2, axis=-1)
            probs2 = mx.stop_gradient(probs2)
            
            # Cross Entropy / KL matching
            # Minimize KL(Sys2 || Sys1) -> Minimize -sum(P2 * log P1)
            loss = -mx.mean(mx.sum(probs2 * log_probs1, axis=-1))
            return loss
            
        loss_and_grad = nn.value_and_grad(self.agent.system1, loss_fn)
        loss, grads = loss_and_grad(self.agent.system1)
        self.optimizer.update(self.agent.system1, grads)
        mx.eval(self.agent.system1.parameters(), self.optimizer.state)
        return loss.item()

    def run_evolution(self, cycles=20):
        print("="*70)
        print("⚔️  ADVERSARIAL CO-EVOLUTION (GPU vs NPU)")
        print("="*70)
        print("Phase 1: NPU Challenge - Find states where NPU disagrees with GPU")
        print("Phase 2: NPU Adaptation - Train NPU to match GPU on these states")
        
        disagreements = []
        losses = []
        
        print(f"\nRunning {cycles} evolutionary cycles...")
        
        for i in range(cycles):
            # 1. Mine Adversarial Examples
            adv_inputs = self.mine_adversarial_examples(batch_size=64)
            
            # Measure disagreement before training
            current_disagreement = self.compute_disagreement(adv_inputs).item()
            disagreements.append(current_disagreement)
            
            # 2. Train System 1 (Adaptation)
            # Train for a few epochs on these hard examples
            cycle_loss = 0
            for _ in range(5):
                cycle_loss = self.train_step(adv_inputs)
            losses.append(cycle_loss)
            
            # Re-quantize to update the "deployed" NPU model
            self.agent.system1.quantize_weights(force=True)
            
            print(f"Cycle {i+1:02d}: Disagreement = {current_disagreement:.4f} -> Loss = {cycle_loss:.4f}")
            
        return disagreements, losses

def plot_evolution(disagreements, losses):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(disagreements, 'r-o', linewidth=2)
    ax1.set_title('System Disagreement (Adversarial Robustness)')
    ax1.set_xlabel('Evolution Cycle')
    ax1.set_ylabel('KL Divergence (Maximized)')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(losses, 'g-s', linewidth=2)
    ax2.set_title('Adaptation Loss (Distillation)')
    ax2.set_xlabel('Evolution Cycle')
    ax2.set_ylabel('Cross Entropy Loss')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adversarial_evolution.png')
    print("\n📊 Results saved to adversarial_evolution.png")

if __name__ == "__main__":
    evo = AdversarialEvolution()
    d, l = evo.run_evolution()
    plot_evolution(d, l)
