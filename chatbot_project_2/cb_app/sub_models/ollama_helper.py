import requests
import subprocess
import json
import time
import numpy as np

# Ollama configuration
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text:latest"
CHAT_MODEL = "llama3.1:latest"


def ensure_ollama_running():
    """Ensure Ollama service is running; start it if necessary."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            print("[✅] Ollama is running.")
            return True
    except requests.exceptions.RequestException:
        print("[⚠️] Ollama not responding — trying to start...")

    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if resp.status_code == 200:
            print("[✅] Ollama started successfully.")
            return True
    except Exception as e:
        print(f"[❌] Failed to start Ollama: {e}")
        return False

    print("[❌] Ollama could not be started.")
    return False


def generate_embedding(text):
    """Generate a vector embedding for text using Ollama."""
    if not text or not isinstance(text, str):
        print("[⚠️] Empty or invalid text for embedding.")
        return None

    ensure_ollama_running()

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )

        if response.status_code != 200:
            print(f"[❌] Ollama embedding failed: {response.text}")
            return None

        data = response.json()
        vector = data.get("embedding")
        if vector:
            print(f"[✅] Embedding generated (len={len(vector)}).")
        else:
            print("[⚠️] No embedding data returned.")
        return vector

    except Exception as e:
        print(f"[❌] Error generating embedding: {e}")
        return None


def embed_text_mean(text, max_chars_per_chunk=800):
    """
    Splits long text into chunks, generates embeddings for each,
    and returns the mean vector.
    """
    if not text.strip():
        return None

    ensure_ollama_running()

    chunks = [text[i:i + max_chars_per_chunk] for i in range(0, len(text), max_chars_per_chunk)]
    embeddings = []

    for i, chunk in enumerate(chunks):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": chunk},
                timeout=60,
            )
            data = response.json()
            emb = data.get("embedding")
            if emb:
                embeddings.append(emb)
            else:
                print(f"[⚠️] Chunk {i} had no embedding — skipped.")
        except Exception as e:
            print(f"[❌] Embedding chunk {i} failed: {e}")

    if not embeddings:
        print("[⚠️] No valid embeddings generated.")
        return None

    mean_emb = np.mean(embeddings, axis=0).tolist()
    print(f"[✅] Averaged {len(embeddings)} chunks → {len(mean_emb)}D vector.")
    return mean_emb


def generate_rag_response(context, query="", temperature=0.3):
    """Send a RAG-style query (context + question) to the Ollama model."""
    prompt = f"Context:\n{context}\n\nUser Question: {query}\n\nAnswer:"

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": CHAT_MODEL,
                "prompt": prompt,
                "options": {"temperature": temperature},
            },
            stream=True,
            timeout=180,
        )

        if response.status_code != 200:
            print(f"[❌] Ollama returned {response.status_code}: {response.text}")
            return f"Error: Ollama returned status {response.status_code}"

        output_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
                if "response" in chunk:
                    output_text += chunk["response"]
                elif "message" in chunk:
                    output_text += chunk["message"]
            except json.JSONDecodeError:
                continue  # skip partial chunks safely

        print("[✅] LLM streaming response assembled.")
        return output_text.strip() or "[WARN] Empty response from model."

    except requests.exceptions.RequestException as e:
        print(f"[❌] Failed to connect to Ollama: {e}")
        return f"Error connecting to Ollama: {e}"
