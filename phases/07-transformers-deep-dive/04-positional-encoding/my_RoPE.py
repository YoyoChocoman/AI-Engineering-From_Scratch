import numpy as np
import math

# Step 1: RoPE applied to Q, K
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = np.copy(x)

    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)

        a, b = x[2 * i], x[2 * i + 1]

        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c

    return out

# Step 2: verify relative-distance property of RoPE
q = np.array([1.0, 0.5, -0.5, 1.0])
k = np.array([0.5, 1.0, 1.0, -0.5])

print(f"Original Q and K scoring (without position): {np.dot(q, k):.4f}")

# Experiment A：Q at the 3nd word，K at the 5th word (distance 3)
pos_q_A = 2
pos_k_A = 5
q_rotated_A = apply_rope(q, pos_q_A)
k_rotated_A = apply_rope(k, pos_k_A)
score_A = np.dot(q_rotated_A, k_rotated_A)

print(f"\nExperiment A (Q=2, K=5, distance=3): {score_A:.4f}")

# Experiment B：Q at the 102th word，K at the 105th word (distance still 3 ！)
pos_q_B = 102
pos_k_B = 105
q_rotated_B = apply_rope(q, pos_q_B)
k_rotated_B = apply_rope(k, pos_k_B)
score_B = np.dot(q_rotated_B, k_rotated_B)

print(f"Experiment B (Q=102, K=105, distance=3): {score_B:.4f}")

# Experiment C：Q at the 2nd word，K at the 10th word (distance 8)
pos_k_C = 10
k_rotated_C = apply_rope(k, pos_k_C)
score_C = np.dot(q_rotated_A, k_rotated_C)

print(f"Experiment C (Q=2, K=10, distance=8): {score_C:.4f}")