import numpy as np

# RMSNorm is much more efficient than LayerNorm
def rms_norm(x, eps=1e-6):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms

# FFN (Swish)
# we use a simple matrix to pretend FFN
# the real SwiGLU contains 3 weight matrix, but here we just want to simulate its transformation
def modern_ffn(x, d_model):
    expansion_factor = 3

    W1 = np.random.randn(d_model, d_model * expansion_factor) * 0.01
    W2 = np.random.randn(d_model * expansion_factor, d_model) * 0.01

    # Swish function: x * sigmoid(x)
    hidden = x @ W1
    hidden = hidden * (1.0 / (1.0 + np.exp(-hidden)))

    return hidden @ W2

# Attention (pretend to call Multi-Heas written in the last class)
def mock_attention(x):
    return x + np.random.randn(*x.shape) * 0.1


# Transformer Block (Pre-Norm Structure)
def modern_decoder_block(x, d_model):
    print("  -> Enter Block...")

    # 1. Pre-Norm
    norm_x1 = rms_norm(x)

    # 2. Attention
    attn_out = mock_attention(norm_x1)

    # 3. Residual Connection
    h = x + attn_out
    print("  -> Attention Done. Residual Connection Done.")

    # 4. Pre-Norm Again
    norm_x2 = rms_norm(h)

    # 5. FFN
    ffn_out = modern_ffn(norm_x2, d_model)

    # 6. Residual Connection Again
    out = h + ffn_out
    print("  -> FFN Done. Residual Connection Done.")

    return out

# Test
N_tokens = 10
d_model = 512
layers = 3

print(f"=== Activate Modern Transformer ===")
X = np.random.randn(N_tokens, d_model)
print(f"Initial Input: {X.shape}")

current_X = X
for i in range(layers):
    print(f"\n[Run {i+1} Layer Transformer Block]")
    current_X = modern_decoder_block(current_X, d_model)

print(f"\nFinal Output: {current_X.shape}")