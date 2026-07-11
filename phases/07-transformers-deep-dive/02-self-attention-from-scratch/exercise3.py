import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    # dk represent the dimension length
    dk = Q.shape[-1]

    # divided by sqrt(dk) to avoid the number becoming too large
    scores = (Q @ K.T) / np.sqrt(dk)

    if mask is not None:
        scores = np.where(mask == 1, -1e9, scores)

    weights = softmax(scores)
    output = weights @ V

    return output, weights