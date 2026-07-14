def reorder_lost_in_middle(items, scores):
    print("\n[system] Start Reordering...")

    # Binf the docs with its score and sort
    paired = sorted(zip(scores, items), reverse=True)
    sorted_items = [items for _, items in paired]

    if len(sorted_items) <= 2:
        return sorted_items

    # cut the docs into two parts
    first_half = sorted_items[::2]
    second_half = sorted_items[1::2]
    second_half.reverse()

    return first_half + second_half

# test
docs = [
    "Docs A (boring)",
    "Docs B (kinda relative)",
    "Docs C (perfect clue)",
    "Docs D (kinda relative)",
    "Docs E (nonsense)"
]

scores = [0.1, 0.6, 0.99, 0.7, 0.2]

print("=== Original State ===")
for d, s in zip(docs, scores):
    print(f"Score: {s:.2f} | {d}")

reordered_docs = reorder_lost_in_middle(docs, scores)

print("\n=== After Reorder ===")
for i, doc in enumerate(reordered_docs):
    position = ""
    if i == 0:
        position = "(At the start)"
    elif i == len(reordered_docs) - 1:
        position = "(At the end)"
    else:
        position = "(In the middle)"

    print(f"Position {i+1}: {doc:<20} {position}")