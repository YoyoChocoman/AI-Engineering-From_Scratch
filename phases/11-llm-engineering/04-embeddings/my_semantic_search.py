import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

print("=== Semantic Search Simulation ===")
query = np.array([0.2, 0.9])

doc1 = np.array([0.9, 0.1])
doc2 = np.array([0.8, 0.2])
doc3 = np.array([0.3, 0.85])

documents = [
    ("iPhone 15 開箱評測", doc1),
    ("蘋果發表會時間", doc2),
    ("支付失敗與退款流程", doc3)
]

print("\n客戶發問：「我的信用卡刷不過」")
print("-" * 40)

results = []
for title, doc_vec in documents:
    score = cosine_similarity(query, doc_vec)
    results.append((title, score))

results.sort(key=lambda x: x[1], reverse=True)

for rank, (title, score) in enumerate(results, 1):
    print(f"Rank {rank} (Similarity {score:.3f}): {title}")