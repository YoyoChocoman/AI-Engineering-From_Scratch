import random

# Step 1: masking logic
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15):
    MASK_ID = "[MASK]"  # in real code it would be a number ID

    input_ids = list(tokens)
    labels = [-100] * len(tokens)   # -100 is the special code in PyTorch, which means the word isn't masked

    for i, t in enumerate(tokens):
        if random.random() < mask_prob:
            labels[i] = t

            # Google's tricky rule: how is the chosen word masked
            # 80% chance turns into MASK
            # 10% chance turns into a random wrong word
            # 10% chance remains the same (test if AI can doubt itself)
            r = random.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = f"[RANDOM_{random.randint(0, vocab_size)}]"
            else:
                pass

    return input_ids, labels

# Test
sentence = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"] * 5
vocab_size = 1000

print(f"Original Length: {len(sentence)} words")
input_ids, labels = create_mlm_batch(sentence, vocab_size, mask_prob=0.15)

print("\n=== Input ===")
print(" ".join(input_ids[:20]), "...")

print("\n=== Labels ===")
for i in range(20):
    if labels[i] != -100:
        print(f"the {i:2d} word: AI's view: '{input_ids[i]:<10}', correct ans: '{labels[i]}'")