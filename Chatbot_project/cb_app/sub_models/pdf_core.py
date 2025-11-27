# cb_app/sub_models/pdf_core.py
from typing import List, Tuple
import re
import fitz
from .embedding_model import default_embedder
from .index_manager import faiss_manager
from ..models import PDFChunk
from django.db import connection

NAMESPACE_PDF = "pdf_chunks"

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        txt = re.sub(r"\s+", " ", page.get_text("text")).strip()
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start >= L:
            break
    return [c for c in chunks if c]

def embed_texts(texts: List[str]) -> List[List[float]]:
    return default_embedder.generate_batch(texts)

def _pg_pdf_fullscan(q_emb: List[float], top_k: int = 3):
    q_pg = "ARRAY[%s]::double precision[]" % ",".join(map(str, q_emb))
    sql = f"""
    SELECT id, text,
      (SELECT SUM(e1 * e2)
         FROM unnest(embedding) WITH ORDINALITY AS a(e1, idx)
         JOIN unnest({q_pg}) WITH ORDINALITY AS b(e2, idx) USING (idx)
      ) AS score
    FROM pdf_chunks
    WHERE embedding IS NOT NULL
    ORDER BY score DESC
    LIMIT %s;
    """
    with connection.cursor() as cur:
        cur.execute(sql, [top_k])
        rows = cur.fetchall()
    res = []
    for i, r in enumerate(rows, start=1):
        res.append({"rank": i, "text": r[1], "score": float(r[2] or 0.0)})
    return res

def pdf_search(query: str, top_k: int = 3):
    q_emb = default_embedder.generate_embedding(query)
    if not q_emb:
        return []
    candidates = faiss_manager.search(NAMESPACE_PDF, q_emb, top_k=top_k)
    results = []
    if candidates:
        chunk_ids = [c[0] for c in candidates]
        chunks = {c.id: c for c in PDFChunk.objects.filter(id__in=chunk_ids)}
        for rank, (obj_id, score) in enumerate(candidates, start=1):
            c = chunks.get(obj_id)
            if c:
                results.append({"rank": rank, "text": c.text, "score": round(float(score), 4)})
    else:
        results = _pg_pdf_fullscan(q_emb, top_k=top_k)
    return results
