import numpy as np

# Step 1: split heads from the single-head attention we already have
def split_heads(X, n_heads):
    n_tokens, d_model = X.shape
    d_head = d_model // n_heads

    # transpose it to make sure the head is in the first dimension
    return X.reshape(n_tokens, n_heads, d_head).transpose(1, 0, 2)

def combine_heads(H):
    n_heads, n_tokens, d_head = H.shape

    # rebuild it in the right direction
    return H.transpose(1, 0, 2).reshape(n_tokens, n_heads * d_head)

# test
N = 10
d_model = 512
n_heads = 8

print(f"=== Initial State ===")
X = np.random.randn(N, d_model)
print(f"Input X's shape: {X.shape} ( {N} words, total dimension: {d_model} )")

# 1. split
print("\n=== Split ===")
X_heads = split_heads(X, n_heads)
print(f"X_heads: {X_heads.shape}")
print(f"turns into {X_heads.shape[0]} heads, each with {X_heads.shape[1]} words and distributed into {X_heads.shape[2]} for scoring")

# In real Transformer, we execute 8 times Q@K.T in each X_heads independently
# skip that part

# 2. combine
print("\n=== Combine ===")
X_combined = combine_heads(X_heads)
print(f"X_combined: {X_combined.shape}")
print(f"Perfectly rebuild back to {N} words, total dimension:{d_model}")