import math

# Step 1: build the vocabulary
# build up a vocab dictionary
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

# Step 2: bag of words
# build a matrix with a row of docs number and a column of vocab size
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1

    return matrix

# Step 3: term frequency and document frequency
# calculate how many times the word appears in which docs
def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1

    return df

def inverse_document_frequency(df, n_docs):
    # IDF formula = log(total docs / appear docs)
    # the more rare the more higher point
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]

# Step 4: TF-IDF
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)

    out = []
    for row  in bow_matrix:
        doc_length = sum(row)

        # tf = appear times / total words
        tf = [c / doc_length if doc_length else 0 for c in row]

        # multiply tf and idf
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])

    return out

docs = [
    ["the", "cat", "sat"],       # 文章 0
    ["the", "dog", "sat"],       # 文章 1
    ["the", "cat", "ran", "fast"], # 文章 2
]

print("=== Dictionary ===")
vocab = build_vocab(docs)
print(vocab)

print("\n=== Bag of Words ===")
bow = bag_of_words(docs, vocab)
for i, row in enumerate(bow):
    print(f"Docs {i}: {row}")

print("\n=== TF-IDF ===")
tfidf_matrix = tfidf(bow)
for i, row in enumerate(tfidf_matrix):
    rounded_row = [round(x, 3) for x in row]
    print(f"Docs {i}: {rounded_row}")

# Step 5: L2-normalize rows
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])

    return out

print("\n=== L2 Normalization ===")
final_matrix = l2_normalize(tfidf_matrix)

for i, row in enumerate(final_matrix):
    rounded_row = [round(x, 3) for x in row]
    print(f"Docs {i}: {rounded_row}")

    length = sum(x * x for x in row)
    print(f"  -> The length turns into {round(length, 5)}")