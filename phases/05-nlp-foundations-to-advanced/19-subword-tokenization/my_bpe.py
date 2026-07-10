from collections import Counter

# Step 1: BPE from scratch
def train_bpe(corpus, num_merges):
    # seperate every word into alphabets and adds </w> to represent the end of a word
    vocab = {tuple(word) + ("</w",): count for word, count in corpus.items()}
    merges = []

    print("=== initial state ===")
    for k, v in vocab.items(): print(f"{k}: {v} times")

    # start merging
    for i in range(num_merges):
        # pairing with the adjacent alphabet
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq

        if not pairs:
            break

        # find out the most common pair
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        print(f"\n[{i+1} merge] most common pair: {best[0]} + {best[1]} (appears {pairs[best]} times)")

        # update the library
        new_vocab = {}
        for symbols, freq in vocab.items():
            new_symbols = []
            j = 0

            while j < len(symbols):
                if j < len(symbols) - 1 and symbols[j] == best[0] and symbols[j+1] == best[1]:
                    new_symbols.append(best[0] + best[1])
                    j += 2
                else:
                    new_symbols.append(symbols[j])
                    j += 1

            new_vocab[tuple(new_symbols)] = freq
        vocab = new_vocab

        print("Dictionary after merge:")
        for k, v in vocab.items(): print(f"{k}: {v} times")

    return merges

# Test
corpus = {
    "low": 5,
    "lower": 2,
    "newest": 6,
    "widest": 3
}

print("\n=== start training BPE ===")
learned_merges = train_bpe(corpus, num_merges=5)

print("\n=== BPE after merge ===")
for i, merge in enumerate(learned_merges):
    print(f"{i+1}. {merge[0]} + {merge[1]} -> {merge[0]+merge[1]}")