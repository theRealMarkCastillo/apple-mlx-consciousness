import sys, os
import mlx.core as mx
import numpy as np

# Ensure local imports
sys.path.append(os.getcwd())

from heterogeneous_architecture import HeterogeneousAgent


def pure_s1_action_probs(agent: HeterogeneousAgent, state: mx.array):
    # Use quantized or full-precision System 1 directly
    if agent.use_quantization and agent.system1.quantized:
        return agent.system1.forward_quantized(state)
    else:
        return agent.system1(state)


def main(weights_path: str = "agent_brain_10k_e5_s10.npz", action_dim: int = 2):
    print(f"Loading agent with weights: {weights_path}")
    agent = HeterogeneousAgent(state_dim=128, action_dim=action_dim, use_quantization=True)
    ok = agent.load_brain(weights_path)
    if not ok:
        print("Failed to load weights.")
        sys.exit(1)

    test_cases = [
        (0.8, 0),
        (-0.8, 1),
        (0.1, 0),
        (-0.1, 1),
    ]

    correct = 0
    for val, expected in test_cases:
        # Build state with desired first component
        state_np = np.zeros(128, dtype=np.float32)
        state_np[0] = val
        state = mx.array(state_np)
        state = state + mx.random.normal((128,)) * 0.1

        probs = pure_s1_action_probs(agent, state)
        mx.eval(probs)
        action = int(mx.argmax(probs).item())
        is_correct = action == expected
        correct += int(is_correct)
        print(f"val={val:+.1f} -> action={action} expected={expected} [{'✅' if is_correct else '❌'}] probs={probs.tolist()}")

    acc = correct / len(test_cases)
    print(f"\nPure System 1 Accuracy: {correct}/{len(test_cases)} ({acc*100:.1f}%)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="agent_brain_10k_e5_s10.npz")
    p.add_argument("--action-dim", type=int, default=2)
    args = p.parse_args()
    main(weights_path=args.weights, action_dim=args.action_dim)
