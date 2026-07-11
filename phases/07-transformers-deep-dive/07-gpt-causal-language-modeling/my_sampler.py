import numpy as np

# the difference between encoder and decoder is mask
# instead of building the network, i decide to build the sampling
# it determines how AI response
vocab = ["蘋果", "香蕉", "大便", "石頭", "鞋子"]
logits = np.array([2.5, 2.0, -1.0, 0.5, 0.0])

# turns Logits into prob (Softmax)
def get_probs(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

print(f"The Original probability distribution: {np.round(get_probs(logits), 3)}")

# Strategy 1: Greedy
def greedy_sample(logits):
    return np.argmax(logits)

print(f"\n[Stratedy 1 - Greedy] GPT's choice: {vocab[greedy_sample(logits)]}")
print("-> pros: stable. cons: boring. response remains the same every time.")

# Strategy 2: Temperature
# divided the score by temperature
# temp < 1 (cool down): higher scores get higher, lower ones get lower. AI turns conservative.
# temp > 1 (warm up): scores become closer. AI turns creative.
def temperature_sample(logits, temperature=1.0):
    adjusted_logits = logits / temperature
    probs = get_probs(adjusted_logits)
    return np.random.choice(len(logits), p=probs)

print(f"\n[Stratedy 2 - Temperature]")
np.random.seed(42)
print(f"temp 0.1 (conserved) GPT's choice: {vocab[temperature_sample(logits, 0.1)]}")
print(f"temp 2.0 (insane) GPT's choice: {vocab[temperature_sample(logits, 2.0)]}")

# Strategy 3: Top-K
def top_k_sample(logits, k=2):
    indices_to_remove = np.argsort(logits)[:-k]
    modified_logits = np.copy(logits)
    modified_logits[indices_to_remove] = -float('inf')

    probs = get_probs(modified_logits)
    return np.random.choice(len(logits), p=probs)

print(f"\n[Strategy 3 - Top-K]")
print(f"Top-2 GPT's choice: {vocab[top_k_sample(logits, k=2)]}")