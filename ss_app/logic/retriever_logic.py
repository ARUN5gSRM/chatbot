# ss_app/logic/retriever_logic.py
from ss_app.sub_models.webcrawl_models import Paragraph
from ss_app.logic.embedding_model import default_embedder
from ss_app.logic.index_manager import faiss_manager

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import threading

nltk.download("punkt", quiet=True)

_lock = threading.Lock()
_cached = {"bm25": None, "paras": None}


def _tok(s):
    return [t.lower() for t in word_tokenize(s) if any(c.isalnum() for c in t)]


def _load_bm25():
    with _lock:
        paras = list(Paragraph.objects.select_related("page"))
        corpus = [_tok(p.text) for p in paras]
        bm25 = BM25Okapi(corpus) if corpus else None
        _cached["bm25"] = bm25
        _cached["paras"] = paras
        return bm25, paras


def bm25_search(query: str, top_k: int):
    bm25, paras = _cached["bm25"], _cached["paras"]
    if not bm25:
        bm25, paras = _load_bm25()

    q = _tok(query)
    scores = bm25.get_scores(q)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    out = []
    for i in idxs:
        para = paras[i]
        best_sent = None
        best = -1
        for s in sent_tokenize(para.text):
            overlap = len(set(_tok(s)).intersection(q))
            sc = overlap * math.log(1 + scores[i]) if scores[i] > 0 else overlap
            if sc > best:
                best = sc
                best_sent = s

        out.append({
            "paragraph_id": para.id,
            "page_url": para.page.url,
            "page_title": para.page.title,
            "paragraph": para.text,
            "best_sentence": best_sent,
            "sentence_score": float(best),
            "bm25_score": float(scores[i]),
        })

    return out


def semantic_search(query: str, top_k: int = 5):
    q_vec = default_embedder.generate_embedding(query)
    if not q_vec:
        return bm25_search(query, top_k)

    # ensure FAISS index is ready
    faiss_manager.safe_build_from_db_if_empty(
        "web_paragraphs",
        lambda: list(Paragraph.objects.filter(embedding__isnull=False)
                     .values_list("id", "embedding"))
    )

    try:
        hits = faiss_manager.safe_search("web_paragraphs", q_vec, top_k)
    except Exception:
        return bm25_search(query, top_k)

    if not hits:
        return bm25_search(query, top_k)

    ids = [h[0] for h in hits]
    paras = {p.id: p for p in Paragraph.objects.filter(id__in=ids).select_related("page")}

    out = []
    q_tokens = set(_tok(query))

    for pid, score in hits:
        p = paras.get(pid)
        if not p:
            continue

        best_sent = None
        best = -1
        for s in sent_tokenize(p.text):
            overlap = len(set(_tok(s)).intersection(q_tokens))
            sc = overlap * (score + 1)
            if sc > best:
                best = sc
                best_sent = s

        out.append({
            "paragraph_id": p.id,
            "page_url": p.page.url,
            "page_title": p.page.title,
            "paragraph": p.text,
            "best_sentence": best_sent,
            "semantic_score": float(score),
            "sentence_score": float(best),
        })

    return sorted(out, key=lambda x: x["sentence_score"], reverse=True)
