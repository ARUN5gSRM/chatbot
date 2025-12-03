# cb_app/logic/pdf_core.py

from typing import List, Dict
import re
import fitz  # PyMuPDF
import numpy as np
from nltk.tokenize import sent_tokenize
from .embedding_model import default_embedder
from .index_manager import faiss_manager
from cb_app.models import PDFChunk
from django.db import connection

NAMESPACE_PDF = "pdf_chunks"


# -------------------------------
# 1. STRUCTURED PDF EXTRACTION
# -------------------------------

def extract_text_from_pdf(pdf_path: str) -> Dict[str, List[str]]:
    """
    Extracts:
    - Headings
    - Structured paragraphs
    - Tables (if detected)
    Returns structured dict for downstream semantic chunking.
    """
    doc = fitz.open(pdf_path)

    headings = []
    paragraphs = []
    tables = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" not in block:
                continue

            full_line = " ".join(
                span["text"].strip()
                for line in block["lines"]
                for span in line["spans"]
            ).strip()

            if not full_line:
                continue

            # --------- Heading Detection ---------
            if (
                full_line.isupper()
                or len(full_line.split()) <= 6
                or re.match(r"^\d+(\.\d+)*\s+", full_line)
            ):
                headings.append(full_line)
            else:
                paragraphs.append(full_line)

        # --------- Table Extraction (Native PyMuPDF) ---------
        try:
            tables_on_page = page.find_tables()
            for table in tables_on_page.tables:
                table_text = "\n".join([" | ".join(row) for row in table.extract()])
                if table_text.strip():
                    tables.append(table_text)
        except Exception:
            # Table detection not supported in older PyMuPDF versions
            pass

    return {
        "headings": headings,
        "paragraphs": paragraphs,
        "tables": tables,
    }


# -------------------------------
# 2. SEMANTIC CHUNKING (SENTENCE-BASED)
# -------------------------------

def chunk_text(text_data: Dict[str, List[str]],
               max_tokens: int = 160,
               overlap_sentences: int = 2) -> List[str]:
    """
    Sentence-aware chunking using semantic boundaries.
    """

    combined_texts = []

    # Preserve hierarchy: Headings → Paragraphs → Tables
    for h in text_data.get("headings", []):
        combined_texts.append(f"HEADING: {h}")

    combined_texts.extend(text_data.get("paragraphs", []))

    for t in text_data.get("tables", []):
        combined_texts.append(f"TABLE:\n{t}")

    sentences = []
    for block in combined_texts:
        try:
            sentences.extend(sent_tokenize(block))
        except Exception:
            sentences.append(block)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        token_estimate = len(sent.split())

        if current_len + token_estimate >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_len = sum(len(s.split()) for s in current_chunk)

        current_chunk.append(sent)
        current_len += token_estimate

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


# -------------------------------
# 3. EMBEDDING
# -------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    return default_embedder.generate_batch(texts)


# -------------------------------
# 4. SQL FALLBACK SEARCH
# -------------------------------

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
        res.append({
            "rank": i,
            "text": r[1],
            "score": float(r[2] or 0.0)
        })

    return res


# -------------------------------
# 5. MAIN PDF SEARCH
# -------------------------------

def pdf_search(query: str, top_k: int = 3, namespace: str = NAMESPACE_PDF):
    q_emb = default_embedder.generate_embedding(query)
    if not q_emb:
        return []

    # Use the given FAISS namespace
    candidates = faiss_manager.search(namespace, q_emb, top_k=top_k)

    results = []

    if candidates:
        chunk_ids = [c[0] for c in candidates]
        chunks = {c.id: c for c in PDFChunk.objects.filter(id__in=chunk_ids)}

        for rank, (obj_id, score) in enumerate(candidates, start=1):
            c = chunks.get(obj_id)
            if c:
                results.append({
                    "rank": rank,
                    "text": c.text,
                    "score": round(float(score), 4)
                })
    else:
        results = _pg_pdf_fullscan(q_emb, top_k=top_k)

    return results
