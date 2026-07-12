import time
import math

# a simple attention function (no normalization)
def simple_attention(q, K, V):
    scores = [sum(qi * ki for qi, ki in zip(q,k)) for k in K]
    m = max(scores) if scores else 0
    exps = [math.exp(s - m) for s in scores]
    s_sum = sum(exps)
    weights = [e / s_sum for e in exps]

    out = [0.0] * len(V[0])
    for w, v in zip(weights, V):
        for i in range(len(out)):
            out[i] += w * v[i]

    return out

# AI with no KV cache (O(N^2))
def naive_generation(tokens_to_generate):
    print(f"\n=== Activate naive AI (without KV cache) ===")
    history_K = []
    history_V = []
    operations = 0

    for step in range(tokens_to_generate):
        current_K = [[1.0, 0.5]] * (step + 1)
        current_V = [[0.5, 1.0]] * (step + 1)

        new_q = [1.0, 1.0]

        out = simple_attention(new_q, current_K, current_V)

        operations += (step + 1)

    print(f"formed {tokens_to_generate} words with total {operations} times Attention.")

# AI with KV cache (O(N))
def cached_generation(tokens_to_generate):
    print(f"\n=== Activate Cached AI (with KV cache) ===")

    kv_cache_K = []
    kv_cache_V = []
    operations = 0

    for step in range(tokens_to_generate):
        kv_cache_K.append([1.0, 0.5])
        kv_cache_V.append([0.5, 1.0])

        new_q = [1.0, 1.0]

        out = simple_attention(new_q, kv_cache_K, kv_cache_V)

        operations += (step + 1)

    print(f"formed {tokens_to_generate} words with total {operations} times Attention.")
    print(f"but we reduce {sum(range(tokens_to_generate))} times of \"reforming K, V matrix\"'s waste.")

# test
tokens = 100
naive_generation(tokens)
cached_generation(tokens)