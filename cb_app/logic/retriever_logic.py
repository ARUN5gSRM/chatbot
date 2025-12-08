# cb_app/logic/retriever.py
from typing import List, Dict, Optional
from cb_app.sub_models.webcrawl_models import Paragraph
from cb_app.logic.index_manager import faiss_manager
from cb_app.logic.embedding_model import default_embedder, EMBED_DIM

# BM25 fallback
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import math
import threading

nltk.download('punkt', quiet=True)

_index_lock = threading.Lock()
_cached = {'bm25': None, 'corpus': None, 'paras': None}


def tokenize_text(text: str):
    tokens = [t.lower() for t in word_tokenize(text) if any(c.isalnum() for c in t)]
    return tokens


def build_bm25(force: bool = False):
    with _index_lock:
        if _cached['bm25'] and not force:
            return _cached['bm25'], _cached['paras']
        paras_qs = list(Paragraph.objects.select_related('page').all())
        paras = paras_qs
        corpus = [tokenize_text(p.text) for p in paras]
        if len(corpus) == 0:
            bm25 = None
        else:
            bm25 = BM25Okapi(corpus)
        _cached['bm25'] = bm25
        _cached['corpus'] = corpus
        _cached['paras'] = paras
        return bm25, paras


def semantic_search(question: str, top_k: int = 5) -> List[Dict]:
    """
    Use FAISS semantic search over namespace 'web_paragraphs'. If FAISS returns nothing,
    fallback to BM25. Returns list of candidates with paragraph, score, and best_sentence.
    """
    if not question or not isinstance(question, str):
        return []

    # generate query embedding
    q_emb = default_embedder.generate_embedding(question)
    if not q_emb:
        # fallback to BM25
        return _bm25_search(question, top_k)

    # ensure index exists & is populated
    faiss_manager.safe_build_from_db_if_empty(
        "web_paragraphs",
        lambda: list(Paragraph.objects.filter(embedding__isnull=False).values_list("id", "embedding"))
    )

    candidates = []
    try:
        faiss_results = faiss_manager.safe_search("web_paragraphs", q_emb, top_k=top_k)
        if faiss_results:
            ids = [r[0] for r in faiss_results]
            paras = {p.id: p for p in Paragraph.objects.filter(id__in=ids).select_related('page')}
            for idx, (pid, score) in enumerate(faiss_results):
                p = paras.get(pid)
                if not p:
                    continue
                # extract best sentence using simple overlap heuristic
                sentences = sent_tokenize(p.text)
                best_sent = None
                best_sent_score = -1.0
                q_tokens = set(tokenize_text(question))
                for s in sentences:
                    s_tokens = set(tokenize_text(s))
                    overlap = len(s_tokens.intersection(q_tokens))
                    if overlap > 0:
                        # weight by overlap and faiss score
                        sent_score = overlap * (score + 1)
                    else:
                        sent_score = 0.0
                    if sent_score > best_sent_score:
                        best_sent_score = sent_score
                        best_sent = s
                candidates.append({
                    'paragraph_id': p.id,
                    'page_url': p.page.url,
                    'page_title': p.page.title,
                    'paragraph': p.text,
                    'best_sentence': best_sent,
                    'bm25_score': None,
                    'semantic_score': float(score),
                    'sentence_score': float(best_sent_score)
                })
            # sort by sentence_score (desc) then semantic_score
            candidates = sorted(candidates, key=lambda c: (c['sentence_score'], c['semantic_score']), reverse=True)
            return candidates[:top_k]
    except Exception:
        # On any FAISS error, fall back to BM25
        return _bm25_search(question, top_k)

    # if no faiss hits -> BM25
    return _bm25_search(question, top_k)


def _bm25_search(question: str, top_k: int = 5) -> List[Dict]:
    bm25, paras = build_bm25()
    if not bm25:
        return []

    q_tokens = tokenize_text(question)
    scores = bm25.get_scores(q_tokens)
    idx_s = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    candidates = []
    for i in idx_s:
        para = paras[i]
        paragraph_text = para.text
        sentences = sent_tokenize(paragraph_text)
        best_sent = None
        best_score = -1.0
        for s in sentences:
            s_tokens = tokenize_text(s)
            overlap = len(set(s_tokens).intersection(set(q_tokens)))
            score = overlap * math.log(1 + scores[i]) if scores[i] > 0 else overlap
            if score > best_score:
                best_score = score
                best_sent = s
        candidates.append({
            'paragraph_id': para.id,
            'page_url': para.page.url,
            'page_title': para.page.title,
            'paragraph': paragraph_text,
            'best_sentence': best_sent,
            'bm25_score': float(scores[i]),
            'semantic_score': None,
            'sentence_score': float(best_score)
        })
    return candidates
