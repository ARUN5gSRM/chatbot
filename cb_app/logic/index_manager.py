"""
In-memory FAISS index manager (namespaced).
- Use IndexFlatIP for normalized vectors (inner-product == cosine).
- Ensures normalization and float32 conversion before adding/searching.
"""
from typing import Callable, Dict, List, Tuple, Any
import numpy as np
import faiss
from threading import Lock
import ast

EMBED_DIM = 384

def _ensure_ndarray(vec: Any) -> np.ndarray:
    """Convert embeddings stored as list, tuple, or string to ndarray float32."""
    if isinstance(vec, str):
        # if stored as textual list like "[0.1, ...]"
        try:
            vec = ast.literal_eval(vec)
        except Exception:
            raise ValueError("Cannot parse string embedding")
    arr = np.array(vec, dtype="float32")
    return arr

def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D numpy array, safe against zero vectors."""
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms

class InMemoryFaissIndex:
    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner-product; use normalized vectors
        self.id_map: List[int] = []          # internal idx -> object id
        self.lock = Lock()

    def add(self, object_ids: List[int], vectors: np.ndarray):
        """Add vectors to FAISS. Vectors shape must be (n, dim)."""
        if vectors is None or vectors.size == 0:
            return
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vectors.shape[1]}")
        # normalize and ensure float32
        vecs = vectors.astype("float32")
        vecs = _normalize_matrix(vecs)
        with self.lock:
            self.index.add(vecs)
            self.id_map.extend([int(x) for x in object_ids])

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Return list of (object_id, score) where score is inner-product == cosine if vectors normalized."""
        if query_vec is None:
            return []
        if isinstance(query_vec, list) or isinstance(query_vec, tuple):
            query_vec = np.array(query_vec, dtype="float32")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[1] != self.dim:
            raise ValueError("Query vector has wrong dim")
        # normalize
        q = _normalize_matrix(query_vec.astype("float32"))
        with self.lock:
            if self.index.ntotal == 0:
                return []
            D, I = self.index.search(q, top_k)
            res = []
            for dist, idx in zip(D[0], I[0]):
                if idx < 0:
                    continue
                obj_id = self.id_map[idx]
                score = float(dist)
                res.append((obj_id, score))
            return res

    def clear(self):
        with self.lock:
            self.index = faiss.IndexFlatIP(self.dim)
            self.id_map = []

class FaissIndexManager:
    def __init__(self):
        self.indices: Dict[str, InMemoryFaissIndex] = {}
        self.lock = Lock()

    def get(self, namespace: str) -> InMemoryFaissIndex:
        with self.lock:
            if namespace not in self.indices:
                self.indices[namespace] = InMemoryFaissIndex(dim=EMBED_DIM)
            return self.indices[namespace]

    def build_from_db(self, namespace: str, fetch_fn: Callable[[], List[Tuple[int, Any]]]):
        """
        fetch_fn: returns list of tuples (object_id, embedding_list_or_str).
        """
        idx = self.get(namespace)
        idx.clear()
        items = fetch_fn() or []
        if not items:
            return
        object_ids = []
        vectors = []
        for obj_id, emb in items:
            if emb is None:
                continue
            try:
                arr = _ensure_ndarray(emb)
            except Exception:
                # skip corrupt embeddings
                continue
            if arr.size == 0:
                continue
            if arr.shape[0] != EMBED_DIM:
                # try reshape if stored as 1D string with commas? otherwise skip
                continue
            object_ids.append(int(obj_id))
            vectors.append(arr)
        if vectors:
            mat = np.vstack(vectors).astype("float32")
            idx.add(object_ids, mat)

    def add(self, namespace: str, object_ids: List[int], vectors: List[Any]):
        idx = self.get(namespace)
        arrs = []
        for v in vectors:
            arrs.append(_ensure_ndarray(v))
        mat = np.vstack(arrs).astype("float32")
        idx.add(object_ids, mat)

    def search(self, namespace: str, query_vec: Any, top_k: int = 5):
        idx = self.get(namespace)
        # ensure ndarray
        if isinstance(query_vec, list) or isinstance(query_vec, tuple) or isinstance(query_vec, str):
            try:
                q = _ensure_ndarray(query_vec)
            except Exception:
                return []
        else:
            q = np.array(query_vec, dtype="float32")
        return idx.search(q, top_k=top_k)

# Singleton for app usage
faiss_manager = FaissIndexManager()
