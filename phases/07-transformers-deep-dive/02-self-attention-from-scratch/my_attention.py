import numpy as np

# Step 1: Softmax from scratch
# use exponential to deal with negative values and enhance the confident
# minus the maxium value to make sure all values after exponential will be between 0 and 1
def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Step 2: Scaled dot-product attention
def scaled_dot_product_attention(Q, K, V):
    # dk represent the dimension length
    dk = Q.shape[-1]

    # divided by sqrt(dk) to avoid the number becoming too large
    scores = (Q @ K.T) / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V

    return output, weights

# Step 3: Self-attention class with learned projections
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)

        # initialized with Xavier-like scaling
        scale = np.sqrt(2.0 / (d_model + dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))

        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv

        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights

# Step 4: Run it on a sentence
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv)
output, weights = attn.forward(X)

print("=== Attention Matirx ===")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        marker = "*" if w > 0.2 else " "
        print(f"{w:5.2f}{marker}", end="")
    print()