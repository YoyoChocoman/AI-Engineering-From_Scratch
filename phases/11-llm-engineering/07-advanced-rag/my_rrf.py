# RRF (an amalgam of keywords and semantic)
def reciprocal_rank_fusion(vector_ranking, keyword_ranking, k=60):
    final_scores = {}

    # core formula = 1 / (k + rank)
    for rank, (doc_id, _) in enumerate(vector_ranking):
        final_scores[doc_id] = 1.0/ (k + rank + 1)

    for rank, (doc_id, _) in enumerate(keyword_ranking):
        if doc_id not in final_scores:
            final_scores[doc_id] = 0.0

        final_scores[doc_id] += 1.0 / (k + rank + 1)

    fused_ranking = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_ranking

# test
print("=== Simulation: Fused Ranking ===")

vector_results = [("Doc_103", 0.95), ("Doc_101", 0.90), ("Doc_105", 0.85)]
print(f"Vector Results: {[doc for doc, _ in vector_results]}")

keyword_results = [("Doc_105", 15.2), ("Doc_109", 12.1), ("Doc_103", 8.5)]
print(f"Keyword Results: {[doc for doc, _ in keyword_results]}")

print("\n=== RRF Result ===")
final_results = reciprocal_rank_fusion(vector_results, keyword_results)

for rank, (doc_id, score) in enumerate(final_results, 1):
    print(f"Rank {rank} | Doc_id: {doc_id} | Score: {score:.5f}")