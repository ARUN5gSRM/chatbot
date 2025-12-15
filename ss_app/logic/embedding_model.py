# ss_app/sub_models/embedding_model.py
"""
Fully offline embedding loader for nomic-embed-text-v1.5 (768-d)

IMPORTANT:
- This does NOT use SentenceTransformer
- Uses ONLY local files
- NEVER connects to internet
"""

from typing import List, Optional
import numpy as np
import os
import torch

# 🔒 Force HuggingFace into offline mode (NO network, NO SSL calls)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer, AutoModel

EMBED_DIM = 768

# ✅ Correct local path (unchanged)
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
        """Load model/tokenizer lazily using ONLY local files."""
        if self.model is not None:
            return

        if not os.path.isdir(LOCAL_MODEL_PATH):
            raise RuntimeError(
                f"❌ Local embedding model not found:\n{LOCAL_MODEL_PATH}"
            )

        # ✅ Tokenizer: local only
        self.tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
        )

        # ✅ Model: local only + safetensors enforced
        self.model = AutoModel.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True,
            use_safetensors=True,
        )

        self.model.eval()

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

        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)

        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

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
