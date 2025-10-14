import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1"

def ensure_ollama_running():
    try:
        r = requests.get(f"{OLLAMA_URL.replace('/api/generate','')}/api/tags")
        r.raise_for_status()
        return True
    except:
        raise RuntimeError("Ollama server not running. Start it with `ollama serve`")

def generate_rag_response(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        raw_text = data.get("response", "")
        return raw_text.strip()
    except Exception as e:
        return f"LLM error: {e}"
