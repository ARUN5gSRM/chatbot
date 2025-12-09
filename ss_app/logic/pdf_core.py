# ss_app/logic/pdf_core.py

from typing import List, Dict
import re
import fitz
import numpy as np
from nltk.tokenize import sent_tokenize
from .embedding_model import default_embedder
from .index_manager import faiss_manager
from ss_app.models import PDFChunk

NAMESPACE_PDF = "pdf_chunks"


def extract_text_from_pdf(pdf_path: str) -> Dict[str, List[str]]:
    doc = fitz.open(pdf_path)

    headings = []
    paragraphs = []
    tables = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue

            line_text = " ".join(
                span["text"].strip()
                for line in block["lines"]
                for span in line["spans"]
            ).strip()

            if not line_text:
                continue

            # crude heading detection
            if (
                line_text.isupper()
                or len(line_text.split()) <= 6
                or re.match(r"^\d+(\.\d+)*\s+", line_text)
            ):
                headings.append(line_text)
            else:
                paragraphs.append(line_text)

        # table extraction
        try:
            tables_on_page = page.find_tables()
            for table in tables_on_page.tables:
                t_text = "\n".join([" | ".join(row) for row in table.extract()])
                if t_text.strip():
                    tables.append(t_text)
        except Exception:
            pass

    return {
        "headings": headings,
        "paragraphs": paragraphs,
        "tables": tables,
    }


def chunk_text(text_data: Dict[str, List[str]], max_tokens: int = 160, overlap_sentences: int = 2) -> List[str]:
    combined = []

    # prepend headings
    for h in text_data.get("headings", []):
        combined.append(f"HEADING: {h}")

    # main body
    combined.extend(text_data.get("paragraphs", []))

    # append table blocks
    for t in text_data.get("tables", []):
        combined.append(f"TABLE:\n{t}")

    # sentence tokenize
    sentences = []
    for block in combined:
        try:
            sentences.extend(sent_tokenize(block))
        except Exception:
            sentences.append(block)

    # chunk construction
    chunks = []
    current = []
    curr_len = 0

    for sent in sentences:
        token_est = len(sent.split())

        if curr_len + token_est >= max_tokens:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:]
            curr_len = sum(len(s.split()) for s in current)

        current.append(sent)
        curr_len += token_est

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def embed_texts(texts: List[str]) -> List[List[float]]:
    return default_embedder.generate_batch(texts)


def pdf_search(query: str, top_k: int = 3, namespace: str = NAMESPACE_PDF):
    """
    FAISS-only semantic search.
    """
    q_emb = default_embedder.generate_embedding(query)
    if not q_emb:
        return []

    # Ensure FAISS index exists or build it once
    faiss_manager.safe_build_from_db_if_empty(
        namespace,
        lambda: list(
            PDFChunk.objects.filter(embedding__isnull=False)
            .values_list("id", "embedding")
        )
    )

    # FAISS vector search
    candidates = faiss_manager.safe_search(namespace, q_emb, top_k=top_k)

    results = []
    if candidates:
        ids = [c[0] for c in candidates]
        chunks = {c.id: c for c in PDFChunk.objects.filter(id__in=ids)}

        for rank, (obj_id, score) in enumerate(candidates, start=1):
            c = chunks.get(obj_id)
            if c:
                results.append({
                    "rank": rank,
                    "text": c.text,
                    "score": round(float(score), 4)
                })

    return results
