# ss_app/logic/chatbot_core.py
from typing import List, Dict, Any, Optional
from .session_helpers import (
    init_session_history_if_needed,
    append_user_message,
    append_assistant_message,
    get_recent_conversation,
)
from .embedding_model import default_embedder
from .index_manager import faiss_manager
from ss_app.models import Ticket
import numpy as np

DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.80
FETCH_FACTOR = 5

NAMESPACE_TICKETS = "tickets"


# No SQL fallback – FAISS only.


def _normalize_vector_safe(vec: Any) -> Optional[np.ndarray]:
    try:
        arr = np.array(vec, dtype="float32")
    except Exception:
        return None
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    norm = np.linalg.norm(arr)
    if norm == 0 or np.isnan(norm):
        return None
    return (arr / norm).astype("float32")


def semantic_search(
    query: str,
    embedding_model=default_embedder,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    namespace: str = NAMESPACE_TICKETS,
):
    # Embed query
    q_emb = embedding_model.generate_embedding(query)
    if not q_emb:
        return []

    q_arr = _normalize_vector_safe(q_emb)
    if q_arr is None:
        return []

    # Ensure FAISS index is ready
    faiss_manager.safe_build_from_db_if_empty(
        namespace,
        lambda: list(
            Ticket.objects.filter(embedding__isnull=False)
            .values_list("id", "embedding")
        )
    )

    # Run FAISS search
    candidates = faiss_manager.safe_search(
        namespace,
        q_arr,
        top_k=top_k * FETCH_FACTOR
    )

    results: List[Dict[str, Any]] = []

    if candidates:
        ids = [c[0] for c in candidates]
        tickets = {t.id: t for t in Ticket.objects.filter(id__in=ids)}

        for obj_id, score in candidates:
            t = tickets.get(obj_id)
            if not t:
                continue

            results.append({
                "id": t.id,
                "short_description": t.short_description,
                "solution": t.solution,
                "rca": t.rca,
                "score": round(float(score), 4),
            })

    if not results:
        return []

    # Apply threshold
    filtered = [r for r in results if r["score"] >= threshold]
    filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)
    return filtered[:top_k]


def chatbot_search(
    request,
    query: str,
    embedding_model=default_embedder,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    namespace: str = NAMESPACE_TICKETS,
):
    init_session_history_if_needed(request)
    append_user_message(request, query)

    hits = semantic_search(
        query,
        embedding_model=embedding_model,
        top_k=top_k,
        threshold=threshold,
        namespace=namespace,
    )

    request.session["last_query"] = query
    request.session["last_threshold"] = threshold
    request.session["last_results"] = [h["id"] for h in hits]
    request.session.modified = True

    if not hits:
        msg = f"No strong matches found above {int(threshold * 100)}% similarity."
        append_assistant_message(request, msg)
        return {
            "message": msg,
            "semantic": [],
            "chat_history": get_recent_conversation(request)
        }

    # Response formatting in required order:
    # Short Description → RCA → Solution → Similarity
    combined_text = "\n\n".join([
        f"Ticket {i+1}:\n"
        f"Short Description: {h['short_description']}\n"
        f"RCA: {h['rca']}\n"
        f"Solution: {h['solution']}\n"
        f"(Similarity: {h['score']})"
        for i, h in enumerate(hits)
    ])

    append_assistant_message(request, combined_text)

    return {
        "semantic": hits,
        "message": combined_text,
        "chat_history": get_recent_conversation(request)
    }
