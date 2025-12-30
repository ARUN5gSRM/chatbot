import os
import numpy as np

# 🔒 FORCE FULL OFFLINE MODE
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 🔹 IMPORT YOUR EXISTING EMBEDDER
from embedding_model import default_embedder   # <-- change filename if needed


def cosine(a, b):
    return float(np.dot(a, b))


print("\n=== OFFLINE EMBEDDING TEST ===\n")

# 1️⃣ BASIC ENCODING TEST
text = "hello world"
vec = default_embedder.generate_embedding(text)

print("Text:", text)
print("Vector type:", type(vec))
print("Embedding dimension:", len(vec))
print("First 10 values:", vec[:10])
print("Last 10 values:", vec[-10:])

assert len(vec) == 768, "❌ Embedding dimension mismatch"
print("✔ Dimension check passed\n")


# 2️⃣ DETERMINISM TEST
v1 = default_embedder.generate_embedding("hello world")
v2 = default_embedder.generate_embedding("hello world")

sim = cosine(v1, v2)
print("Determinism cosine similarity:", sim)

assert sim > 0.999, "❌ Embedding is not deterministic"
print("✔ Determinism test passed\n")


# 3️⃣ SEMANTIC SIMILARITY TEST
texts = [
    "hello world",
    "hi world",
    "machine learning",
    "deep neural networks",
]

vecs = default_embedder.generate_batch(texts)

print("Semantic similarity checks:")
print("hello vs hi:", cosine(vecs[0], vecs[1]))
print("hello vs ML:", cosine(vecs[0], vecs[2]))
print("ML vs DNN:", cosine(vecs[2], vecs[3]))

assert cosine(vecs[0], vecs[1]) > cosine(vecs[0], vecs[2]), "❌ Semantic mismatch"
print("✔ Semantic similarity test passed\n")


print("🎉 ALL OFFLINE EMBEDDING TESTS PASSED")
