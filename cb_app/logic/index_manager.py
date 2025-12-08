# cb_app/sub_models/index_manager.py
"""
In-memory FAISS index manager (namespaced) — hardened.

Changes made:
- EMBED_DIM set to 768
- Per-namespace locks to avoid race conditions
- safe_get_or_create, safe_build_from_db_if_empty, safe_add, safe_search helpers
- validation of vector shapes prior to adding/searching with explicit errors
- minimal API compatibility with previous usage: get, add, search remain,
  and new safe_* wrappers added for callers that want atomic semantics.
"""
from typing import Callable, Dict, List, Tuple, Any, Optional
import numpy as np
import faiss
from threading import Lock, RLock
import ast

# Embed dim changed to 768 to match nomic-embed-text-v1.5
EMBED_DIM = 768

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
        self.lock = RLock()

    def add(self, object_ids: List[int], vectors: np.ndarray):
        """Add vectors to FAISS. Vectors shape must be (n, dim)."""
        if vectors is None:
            raise ValueError("vectors is None")
        if isinstance(vectors, list):
            vectors = np.vstack([_ensure_ndarray(v) for v in vectors]).astype("float32")
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
        if isinstance(query_vec, list) or isinstance(query_vec, tuple) or isinstance(query_vec, str):
            query_vec = _ensure_ndarray(query_vec)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[1] != self.dim:
            raise ValueError(f"Query vector has wrong dim: expected {self.dim}, got {query_vec.shape[1]}")
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
        # main dict of namespace -> InMemoryFaissIndex
        self.indices: Dict[str, InMemoryFaissIndex] = {}
        # lock protecting indices dict
        self.lock = RLock()
        # per-namespace locks for fine-grained concurrency
        self._ns_locks: Dict[str, RLock] = {}

    def _get_ns_lock(self, namespace: str) -> RLock:
        with self.lock:
            if namespace not in self._ns_locks:
                self._ns_locks[namespace] = RLock()
            return self._ns_locks[namespace]

    # --- Basic operations (backwards compatible) ---
    def get(self, namespace: str) -> InMemoryFaissIndex:
        """Return an index object for namespace, creating if necessary (not IO heavy)."""
        with self.lock:
            if namespace not in self.indices:
                self.indices[namespace] = InMemoryFaissIndex(dim=EMBED_DIM)
            return self.indices[namespace]

    def add(self, namespace: str, object_ids: List[int], vectors: List[Any]):
        """Add vectors to the namespace. Vectors can be numpy arrays, lists, or strings representing lists."""
        idx = self.get(namespace)
        # convert vectors to ndarray (validate shapes)
        arrs = []
        for v in vectors:
            arr = _ensure_ndarray(v)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            if arr.shape[0] != EMBED_DIM:
                raise ValueError(f"Attempt to add vector with dim {arr.shape[0]} to index dim {EMBED_DIM}")
            arrs.append(arr)
        mat = np.vstack(arrs).astype("float32")
        idx.add(object_ids, mat)

    def search(self, namespace: str, query_vec: Any, top_k: int = 5):
        idx = self.get(namespace)
        # ensure ndarray and shape
        if isinstance(query_vec, list) or isinstance(query_vec, tuple) or isinstance(query_vec, str):
            try:
                q = _ensure_ndarray(query_vec)
            except Exception:
                return []
        else:
            q = np.array(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != EMBED_DIM:
            # wrong dimension: return empty so callers fallback to SQL search
            return []
        return idx.search(q, top_k=top_k)

    # --- Safe wrappers for concurrency & defensive checks ---

    def safe_get_or_create(self, namespace: str) -> InMemoryFaissIndex:
        """Atomically get or create an index for the namespace."""
        with self.lock:
            if namespace not in self.indices:
                self.indices[namespace] = InMemoryFaissIndex(dim=EMBED_DIM)
            return self.indices[namespace]

    def safe_build_from_db_if_empty(self, namespace: str, fetch_fn: Callable[[], List[Tuple[int, Any]]]):
        """
        If the namespace index has zero vectors, atomically fetch from DB via fetch_fn and populate it.
        fetch_fn should return iterable of (object_id, embedding) pairs.
        """
        ns_lock = self._get_ns_lock(namespace)
        with ns_lock:
            idx = self.get(namespace)
            # defensive check for presence of index internals
            try:
                nt = idx.index.ntotal
            except Exception:
                nt = 0
            if nt and nt > 0:
                return  # already populated
            # fetch items
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
                if arr.ndim != 1:
                    arr = arr.reshape(-1)
                if arr.shape[0] != EMBED_DIM:
                    # skip wrong-shape embeddings
                    continue
                object_ids.append(int(obj_id))
                vectors.append(arr)
            if vectors:
                mat = np.vstack(vectors).astype("float32")
                idx.add(object_ids, mat)

    def safe_add(self, namespace: str, object_ids: List[int], vectors: List[Any]):
        """Add vectors under the namespace with per-namespace locking and validation."""
        ns_lock = self._get_ns_lock(namespace)
        with ns_lock:
            # reuse add() which will validate shapes
            return self.add(namespace, object_ids, vectors)

    def safe_search(self, namespace: str, query_vec: Any, top_k: int = 5):
        """Search with per-namespace lock; returns [] on dim mismatch or empty index."""
        ns_lock = self._get_ns_lock(namespace)
        with ns_lock:
            return self.search(namespace, query_vec, top_k=top_k)

    def safe_pop(self, namespace: str) -> Optional[InMemoryFaissIndex]:
        """Atomically pop and return an index (used for cleanup on logout)."""
        with self.lock:
            return self.indices.pop(namespace, None)

# Singleton for app usage
faiss_manager = FaissIndexManager()
