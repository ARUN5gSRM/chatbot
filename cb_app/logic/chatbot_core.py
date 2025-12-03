# cb_app/logic/chatbot_core.py
from typing import List, Dict, Any, Optional
from django.db import connection
from .session_helpers import (
    init_session_history_if_needed,
    append_user_message,
    append_assistant_message,
    get_recent_conversation,
)
from .embedding_model import default_embedder
from .index_manager import faiss_manager
from cb_app.models import Ticket
import numpy as np

DEFAULT_TOP_K = 3
# Recommended default for multi-qa-MiniLM-L6-cos-v1 (calibrate later if needed)
DEFAULT_THRESHOLD = 0.70
FETCH_FACTOR = 5

NAMESPACE_TICKETS = "tickets"


def _pg_cosine_search(query_vec: List[float], top_k: int = DEFAULT_TOP_K * FETCH_FACTOR):
    """
    Perform cosine-like search inside Postgres by computing dot product between
    stored embedding and provided normalized query vector. Expects query_vec to
    be a Python list of floats (normalized).
    """
    if not query_vec:
        return []

    q_pg = "ARRAY[%s]::double precision[]" % ",".join(map(str, query_vec))
    sql = f"""
    SELECT id, short_description, solution,
      (SELECT SUM(e1 * e2)
         FROM unnest(embedding) WITH ORDINALITY AS a(e1, idx)
         JOIN unnest({q_pg}) WITH ORDINALITY AS b(e2, idx) USING (idx)
      ) AS score
    FROM tickets_final
    WHERE embedding IS NOT NULL
    ORDER BY score DESC
    LIMIT %s;
    """
    with connection.cursor() as cur:
        cur.execute(sql, [top_k])
        rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "short_description": r[1],
            "solution": r[2],
            "score": float(r[3] or 0.0)
        })
    return results


def _normalize_vector_safe(vec: Any) -> Optional[np.ndarray]:
    """Return a float32 L2-normalized 1D numpy array or None if invalid."""
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
    """
    Perform semantic retrieval:
    1) Embed + normalize query
    2) Try FAISS first (fast) in the given namespace
    3) Fallback to Postgres dot-product if FAISS has no vectors
    4) Filter by configured threshold and return top_k results
    """
    q_emb = embedding_model.generate_embedding(query)
    if not q_emb:
        return []

    # Defensive normalization / dtype conversion
    q_arr = _normalize_vector_safe(q_emb)
    if q_arr is None:
        return []

    # Try FAISS (faiss_manager will also normalize internally, but we normalize here too)
    candidates = faiss_manager.search(namespace, q_arr, top_k=top_k * FETCH_FACTOR)
    results: List[Dict[str, Any]] = []

    if candidates:
        # candidates: list of (object_id, score) where score is inner-product (== cosine if normalized)
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
                "score": round(float(score), 4)
            })
    else:
        # Fallback to Postgres cosine (pass normalized list)
        q_list = q_arr.tolist()
        results = _pg_cosine_search(q_list, top_k=top_k * FETCH_FACTOR)

    # Filter by threshold and return top_k
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
    """
    Main chatbot wrapper: maintain session history, run semantic search and
    append assistant message.
    """
    init_session_history_if_needed(request)
    append_user_message(request, query)

    hits = semantic_search(
        query,
        embedding_model=embedding_model,
        top_k=top_k,
        threshold=threshold,
        namespace=namespace,
    )

    # Store context for ticket generation / debugging
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

    combined_text = "\n\n".join([
        f"Ticket {i+1}: {h['short_description']}\nSolution: {h['solution']}\n(Similarity: {h['score']})"
        for i, h in enumerate(hits)
    ])

    append_assistant_message(request, combined_text)

    return {
        "semantic": hits,
        "message": combined_text,
        "chat_history": get_recent_conversation(request)
    }
