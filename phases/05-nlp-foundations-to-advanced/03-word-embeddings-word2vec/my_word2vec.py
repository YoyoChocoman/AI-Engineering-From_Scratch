import numpy as np

# here is the concept of word vector
# since it is now abandoned and is mostly about math
# I decide to understand only what embedding is

vocab = {"king": 0, "man": 1, "woman": 2}

np.random.seed(42)
embedding_matrix = np.random.normal(0, 0.1, size=(3, 4))

print("\n=== The Embedding Matrix ===")
print(np.round(embedding_matrix, 3))

king_index = vocab["king"]
king_vector = embedding_matrix[king_index]

print("\n=== \"King\" in AI's view ===" )
print(np.round(king_vector, 3))

# Use it
from gensim.models import Word2Vec

# the train data
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
    ["the", "puppy", "ran", "across", "the", "park"],
    ["the", "king", "is", "a", "strong", "man"],
    ["the", "queen", "is", "a", "strong", "woman"],
    ["the", "prince", "is", "a", "young", "man"],
    ["the", "princess", "is", "a", "young", "woman"]
]

# train the model
print("\n=== Training Word2Vec Model ===")
model = Word2Vec(
    sentences,
    vector_size=10,
    window=2,
    min_count=1,
    epochs=100
)

# turns into coordinates
print("\n=== \"king\" in AI's view ===")
print(model.wv["king"])

# finds out the similarity
print("\n=== word most similar to \"dog\" ===")
similar_to_dog = model.wv.most_similar("dog", topn=3)
for word, score in similar_to_dog:
    print(f"word: {word:8s} | similarity: {score:.3f}")

print("\n=== word most similar to \"king\" ===")
similar_to_king = model.wv.most_similar("king", topn=3)
for word, score in similar_to_king:
    print(f"word: {word:8s} | similarity: {score:.3f}")