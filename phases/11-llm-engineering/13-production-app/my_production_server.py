import time
import math
import hashlib

print("=== Semantic Cache ===")

# Step 1: A simplified Embedding and Cosine Similarity
def dummy_embedding(text):
    val = sum(ord(c) for c in text)
    vec = [math.sin(val), math.cos(val)]
    norm = math.sqrt(vec[0]**2 + vec[1]**2)
    return [vec[0]/norm, vec[1]/norm]

def cosine_similarity(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1]

# Step 2: Semantic Cache
class SemanticCache:
    def __init__(self, threshold=0.95):
        self.entries = []
        self.threshold = threshold
        self.hits = 0
        self.misses = 0

    def get(self, query):
        print(f"[Server] Recieved Query: {query}")
        query_emb = dummy_embedding(query)

        for entry in self.entries:
            score = cosine_similarity(query_emb, entry["embedding"])
            if score >= self.threshold:
                self.hits += 1
                print(f"[Cache HIT] Find a similar query '{entry['query']}' (similarity: {score:.3f})")
                return entry["response"]

        self.misses += 1
        print(f"[Cache MISS] Find no similar query, calling API...")
        return None

    def put(self, query, response):
        self.entries.append({
            "query": query,
            "embedding": dummy_embedding(query),
            "response": response
        })
        print(f"Storing '{query}' into cache")

# Step 3: Simulate Server working
cache = SemanticCache(threshold=0.95)

q1 = "How to reset the password?"
ans1 = cache.get(q1)
if not ans1:
    time.sleep(1)
    ans1 = "You can reset it in the privacy setting at the top right corner."
    cache.put(q1, ans1)

q2 = "I forget my password and want to reset it. How?"
ans2 = cache.get(q2)
if not ans2:
    time.sleep(1)
    ans1 = "You can reset it in the privacy setting at the top right corner."
    cache.put(q1, ans1)

print("\n=== Server Report ===")
print(f"Total Calls: {cache.hits + cache.misses}")
print(f"Total Hits: {cache.hits}")
print(f"Cache Hit Rate: {(cache.hits / (cache.hits + cache.misses)) * 100:.1f}%")