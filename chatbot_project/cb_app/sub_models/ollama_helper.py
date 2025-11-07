# cb_app/sub_models/ollama_helper.py
import requests
import subprocess
import json
import time
import numpy as np

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"


def ensure_ollama_running():
    """Ensure Ollama service is running; try to start it if not responding."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass

    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(4)
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def generate_embedding(text: str):
    """Generate an embedding vector using the Ollama embeddings endpoint."""
    if not text or not isinstance(text, str):
        return None

    if not ensure_ollama_running():
        print("[ollama_helper] Ollama not running")
        return None

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        if resp.status_code != 200:
            print(f"[ollama_helper] embedding failed: {resp.status_code} {resp.text}")
            return None
        data = resp.json()
        return data.get("embedding")
    except Exception as e:
        print(f"[ollama_helper] embedding exception: {e}")
        return None


def embed_text_mean(text: str, max_chars_per_chunk: int = 800):
    """Split long text into chunks and return the mean embedding."""
    text = str(text or "").strip()
    if not text:
        return None

    if not ensure_ollama_running():
        print("[ollama_helper] Ollama not running for embed_text_mean")
        return None

    chunks = [text[i:i + max_chars_per_chunk] for i in range(0, len(text), max_chars_per_chunk)]
    embeddings = []
    for chunk in chunks:
        emb = generate_embedding(chunk)
        if emb:
            embeddings.append(np.array(emb, dtype=float))
    if not embeddings:
        return None
    mean_emb = np.mean(embeddings, axis=0).tolist()
    return mean_emb


# 🔹 Wrapper class used by chatbot.py
class OllamaEmbeddingModel:
    """Simple wrapper to provide .generate_embedding() API for chatbot."""
    def generate_embedding(self, text: str):
        return generate_embedding(text)
