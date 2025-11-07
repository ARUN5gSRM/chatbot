import re
from urllib.parse import quote_plus
from typing import List, Dict, Any
from sqlalchemy import create_engine, text
from django.conf import settings

# ---------------- CONFIG ----------------
DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.80
FETCH_FACTOR = 5
CONVERSATION_HISTORY_KEY = "chat_history"
MAX_CONVERSATION_MESSAGES = 8
STATE_KEY = "chat_state"

# ---------------- DATABASE ----------------
def get_engine():
    db = settings.DATABASES["default"]
    conn_str = (
        f"postgresql+psycopg2://{db['USER']}:{quote_plus(db['PASSWORD'])}"
        f"@{db['HOST']}:{db['PORT']}/{db['NAME']}"
    )
    return create_engine(conn_str)


# ---------------- SESSION HELPERS ----------------
def init_session_history_if_needed(request):
    """Ensure session keys for conversation exist."""
    if CONVERSATION_HISTORY_KEY not in request.session:
        request.session[CONVERSATION_HISTORY_KEY] = []
    if STATE_KEY not in request.session:
        request.session[STATE_KEY] = "IDLE"
    request.session.modified = True


def append_user_message(request, text: str):
    """Add user message to session history."""
    init_session_history_if_needed(request)
    history = request.session[CONVERSATION_HISTORY_KEY]
    history.append({"role": "user", "content": text})
    request.session[CONVERSATION_HISTORY_KEY] = history[-MAX_CONVERSATION_MESSAGES:]
    # 🔹 Store last query for ticket generation
    request.session["last_query"] = text
    request.session.modified = True


def append_assistant_message(request, text: str):
    """Add assistant message to session history."""
    init_session_history_if_needed(request)
    history = request.session[CONVERSATION_HISTORY_KEY]
    history.append({"role": "assistant", "content": text})
    request.session[CONVERSATION_HISTORY_KEY] = history[-MAX_CONVERSATION_MESSAGES:]
    # 🔹 Store last bot response
    request.session["last_bot_response"] = text
    request.session.modified = True


def get_recent_conversation(request):
    """Return recent conversation messages."""
    init_session_history_if_needed(request)
    return request.session[CONVERSATION_HISTORY_KEY][-MAX_CONVERSATION_MESSAGES:]


def clear_conversation(request):
    """Reset conversation session."""
    if CONVERSATION_HISTORY_KEY in request.session:
        del request.session[CONVERSATION_HISTORY_KEY]
    request.session[STATE_KEY] = "IDLE"
    request.session.modified = True
    return {"status": "cleared"}


# ---------------- SEMANTIC SEARCH ----------------
def semantic_search(query: str, embedding_model, top_k: int = DEFAULT_TOP_K, threshold: float = DEFAULT_THRESHOLD):
    """
    Perform semantic retrieval using pgvector cosine similarity.
    """
    try:
        q_emb = embedding_model.generate_embedding(query)
        if not q_emb:
            return []

        q_vec_str = "[" + ",".join(str(float(x)) for x in q_emb) + "]"

        engine = get_engine()
        sql = text(f"""
            SELECT id, short_description, resolution_code, solution,
                   1 - (embedding <=> '{q_vec_str}') AS similarity
            FROM tickets_final
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> '{q_vec_str}'
            LIMIT :limit
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, {"limit": top_k * FETCH_FACTOR}).fetchall()

        results = []
        for r in rows:
            sim = float(r.similarity or 0)
            if sim >= threshold:
                results.append({
                    "id": r.id,
                    "short_description": r.short_description,
                    "resolution_code": getattr(r, "resolution_code", None),
                    "solution": r.solution,
                    "score": round(sim, 4)
                })
            if len(results) >= top_k:
                break

        return sorted(results, key=lambda x: x["score"], reverse=True)

    except Exception as e:
        return [{
            "id": None,
            "short_description": "SEMANTIC ERROR",
            "resolution_code": "SEM-ERR",
            "solution": str(e),
            "score": 0.0
        }]


# ---------------- MAIN CHATBOT FUNCTION ----------------
def chatbot_search(request, query: str, embedding_model, top_k: int = DEFAULT_TOP_K, threshold: float = DEFAULT_THRESHOLD):
    """
    Chatbot logic using semantic retrieval.
    """
    init_session_history_if_needed(request)
    append_user_message(request, query)

    hits = semantic_search(query, embedding_model, top_k=top_k, threshold=threshold)

    # 🔹 Store semantic context for ticket generation
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
