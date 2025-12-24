import os
from typing import List, Optional
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

EMBED_DIM = 768

# ✅ FIX 1: correct folder name (local_models, not local_model)
LOCAL_MODEL_PATH = r"C:\Users\venka\PycharmProjects\upchat\Chatbot_project\local_models\nomic-embed-text-v1.5"




class EmbeddingModel:
    """
    Lazy-loaded embedding model for offline use.
    """
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.dim = EMBED_DIM

    def _ensure_loaded(self):
        """Load model/tokenizer lazily to avoid Django import crashes."""
        if self.model is not None:
            return

        if not os.path.exists(LOCAL_MODEL_PATH):
            raise RuntimeError(
                f"❌ Local embedding model not found:\n{LOCAL_MODEL_PATH}\n"
                "Make sure the model files are downloaded properly."
            )

        # ✅ FIX 2: load AutoConfig WITH trust_remote_code=True
        config = AutoConfig.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model = AutoModel.from_pretrained(
            LOCAL_MODEL_PATH,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
        )

        self.model.eval()

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0 or np.isnan(norm):
            return [0.0] * self.dim
        return (vec / norm).tolist()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using mean pooling."""
        self._ensure_loaded()

        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        )

        with torch.no_grad():
            outputs = self.model(**encoded)

        # Mean pooling
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, dim)
        mask = encoded["attention_mask"].unsqueeze(-1)

        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)

        pooled = summed / counts  # (batch, dim)
        arr = pooled.cpu().numpy()

        if arr.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"Model returned {arr.shape[1]} dims, expected {EMBED_DIM}"
            )

        # L2 normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms

        return arr.tolist()

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not isinstance(text, str):
            return None
        return self._embed_batch([text])[0]

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed_batch(texts)


default_embedder = EmbeddingModel()
