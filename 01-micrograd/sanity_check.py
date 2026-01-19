import math

from value import Value


def test_manual_neuron():
    # 1. Setup inputs and weights (Karpathy's exact values)
    x1 = Value(2.0)
    x2 = Value(0.0)
    w1 = Value(-3.0)
    w2 = Value(1.0)
    b = Value(6.8813735870195432)  # A specific bias to make tanh results "clean"

    # 2. Forward pass: (x1*w1 + x2*w2) + b
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    o = n.tanh()

    # 3. Backward pass
    o.backward()

    # 4. The Expected Results
    # If your engine is correct, these gradients should match:
    expected_grads = {"x1": -1.5, "w1": 1.0, "x2": 0.5, "w2": 0.0, "b": 0.5}

    print("--- Sunday Night Check ---")
    for name, val in [("x1", x1), ("w1", w1), ("x2", x2), ("w2", w2), ("b", b)]:
        match = math.isclose(val.grad, expected_grads[name], rel_tol=1e-5)
        status = (
            "✅ PASS"
            if match
            else f"❌ FAIL (Expected {expected_grads[name]}, got {val.grad})"
        )
        print(f"{name}.grad: {val.grad:.4f} | {status}")


if __name__ == "__main__":
    test_manual_neuron()
