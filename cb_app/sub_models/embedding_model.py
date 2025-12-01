# cb_app/sub_models/embedding_model.py
"""
Embedding wrapper (sentence-transformers).
Model: local offline copy of multi-qa-MiniLM-L6-cos-v1
Outputs normalized 384-d vectors (list[float]).
"""

from typing import List, Optional
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Old (online): "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
# New (offline): load from local folder inside project
EMBED_DIM = 384

# Build path:  cb_app/sub_models/embedding_model.py → ../../local_models/multi-qa-MiniLM-L6-cos-v1
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "..",
        "local_models",
        "multi-qa-MiniLM-L6-cos-v1"
    )
)


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        """
        Loads the embedding model from local directory only.
        Internet is NOT required.
        """
        model_path = LOCAL_MODEL_PATH

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"❌ Local embedding model not found at: {model_path}\n"
                f"Make sure the folder is present (Git LFS) and contains pytorch_model.bin."
            )

        # Load SentenceTransformer model from disk
        self.model = SentenceTransformer(model_path)
        self.dim = EMBED_DIM

    def _normalize(self, vec: np.ndarray) -> List[float]:
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return [0.0] * self.dim
        return (vec / norm).tolist()

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not isinstance(text, str):
            return None
        vec = self.model.encode(text, show_progress_bar=False)
        return self._normalize(np.array(vec, dtype=float))

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self.model.encode(texts, show_progress_bar=False)
        arr = np.array(vecs, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr.tolist()


# default global instance for convenience
default_embedder = EmbeddingModel()
