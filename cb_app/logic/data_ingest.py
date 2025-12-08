# cb_app/sub_models/data_ingest.py
"""
Data ingestion utilities: read Excel rows, parse resolution notes, create Ticket rows,
generate embeddings, store in DB (ArrayField), and update FAISS in-memory index.
"""
import re
import pandas as pd
from django.contrib.auth.models import User
from cb_app.models import Ticket, PDFDocument, PDFChunk
from .embedding_model import default_embedder, EMBED_DIM
from .index_manager import faiss_manager

def parse_resolution_notes(notes: str):
    category, issue, rca, solution = "", "", "", ""
    cat_match = re.search(
        r"(?:category|classification)\s*[:\-]?\s*(.*?)(?=\s*(?:issue|issue\s*description|rca|cause|solution|score|data\s*fix)\s*[:\-]|$)",
        str(notes), re.IGNORECASE | re.DOTALL
    )
    issue_match = re.search(
        r"(?:issue|issue\s*description)\s*[:\-]?\s*(.*?)(?=\s*(?:rca|cause|solution|score|data\s*fix|category|classification)\s*[:\-]|$)",
        str(notes), re.IGNORECASE | re.DOTALL
    )
    rca_match = re.search(
        r"(?:rca|cause)\s*[:\-]?\s*(.*?)(?=\s*(?:solution|score|data\s*fix|category|classification|issue)\s*[:\-]|$)",
        str(notes), re.IGNORECASE | re.DOTALL
    )
    sol_match = re.search(
        r"(?:solution|data\s*fix)\s*[:\-]?\s*(.*?)(?=\s*(?:score|category|classification|issue|rca|cause)\s*[:\-]|$)",
        str(notes), re.IGNORECASE | re.DOTALL
    )
    if cat_match:
        category = cat_match.group(1).strip()
    if issue_match:
        issue = issue_match.group(1).strip()
    if rca_match:
        rca = rca_match.group(1).strip()
    if sol_match:
        solution = sol_match.group(1).strip()
    elif notes:
        solution = str(notes).strip()
    return category, issue, rca, solution

def ingest_excel_file(file_obj, uploaded_by_user):
    """
    file_obj: Django uploaded file (Excel). uploaded_by_user: User instance.
    Returns: dict with counts and summary.
    """
    df = pd.read_excel(file_obj).fillna("")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "resolution_notes" in df.columns:
        parsed = df["resolution_notes"].apply(lambda x: parse_resolution_notes(x))
        df[["category", "issue", "rca", "solution"]] = pd.DataFrame(parsed.tolist(), index=df.index)

    # extract keywords basic
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        stop_words = set(stopwords.words("english"))
        def extract_keywords(text):
            tokens = word_tokenize(str(text).lower())
            keywords = [w for w in tokens if w.isalpha() and w not in stop_words]
            return " ".join(keywords)
        if "description" in df.columns and "solution" in df.columns:
            df["keywords"] = (df["description"].fillna("") + " " + df["solution"].fillna("")).apply(extract_keywords)
    except Exception:
        # nltk resources might not be present; skip keywords
        df["keywords"] = df.get("keywords", "")

    created = []
    new_ids = []
    new_vectors = []
    for _, row in df.iterrows():
        t = Ticket.objects.create(
            short_description = row.get("short_description") or row.get("description") or "",
            description = row.get("description") or "",
            keywords = row.get("keywords") or "",
            solution = row.get("solution") or "",
            category = row.get("category") or "",
            issue = row.get("issue") or "",
            rca = row.get("rca") or "",
            uploaded_by = uploaded_by_user
        )
        combined_text = " ".join(filter(None, [t.short_description, t.description, t.keywords])).strip()
        emb = default_embedder.generate_embedding(combined_text) if combined_text else [0.0] * EMBED_DIM
        t.embedding = emb
        t.save(update_fields=["embedding"])
        new_ids.append(t.id)
        new_vectors.append(emb)
        created.append(t.id)

    # update FAISS
    if new_ids:
        # use safe_add to avoid races
        try:
            faiss_manager.safe_add("tickets", new_ids, new_vectors)
        except Exception as e:
            # fallback to non-safe add if desired, but propagate error for now
            faiss_manager.add("tickets", new_ids, new_vectors)

    return {"created_count": len(created), "ids": created}
