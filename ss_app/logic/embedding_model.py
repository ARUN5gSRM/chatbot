import os
from typing import List, Optional

# ------------------------------------------------------------------
# FORCE HUGGING FACE & TRANSFORMERS TO OFFLINE MODE (BEFORE IMPORTS)
# ------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import numpy as np
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
EMBED_DIM = 768

# IMPORTANT: This path MUST point to the folder that contains:
# config_sentence_transformers.json, modules.json, 1_Pooling/, model.safetensors
LOCAL_MODEL_PATH = (
    r"C:\Users\venka\PycharmProjects\upchat\Chatbot_project"
    r"\local_models\nomic-embed-text-v1.5"
)


class EmbeddingModel:
    """
    Lazy-loaded embedding model for OFFLINE use.
    NO INTERNET CALLS.
    Django-safe.
    """

    def __init__(self):
        self.model = None
        self.dim = EMBED_DIM

    def _ensure_loaded(self):
        """Load model lazily to avoid Django startup crashes."""
        if self.model is not None:
            return

        if not os.path.exists(LOCAL_MODEL_PATH):
            raise RuntimeError(
                f"❌ Local embedding model not found:\n{LOCAL_MODEL_PATH}\n"
                "Ensure the SentenceTransformer model files exist."
            )

        # ------------------------------------------------------------------
        # CORRECT LOADER FOR THIS MODEL STRUCTURE
        # ------------------------------------------------------------------
        self.model = SentenceTransformer(
            LOCAL_MODEL_PATH,
            device="cpu",
        )

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return [0.0] * self.dim
        return (vec / norm).tolist()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        if embeddings.shape[1] != self.dim:
            raise RuntimeError(
                f"Model returned {embeddings.shape[1]} dims, expected {self.dim}"
            )

        return embeddings.tolist()

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not isinstance(text, str):
            return None
        return self._embed_batch([text])[0]

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed_batch(texts)


# ------------------------------------------------------------------
# SINGLETON INSTANCE (DO NOT CREATE PER REQUEST)
# ------------------------------------------------------------------
default_embedder = EmbeddingModel()
