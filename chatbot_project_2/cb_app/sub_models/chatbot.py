import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25L
import nltk
from nltk.corpus import wordnet
from functools import lru_cache
import numpy as np
from django.conf import settings
import re
import json
from .ollama_helper import ensure_ollama_running, generate_rag_response, generate_embedding


def get_engine():
    DB_USER = settings.DATABASES['default']['USER']
    DB_PASS = settings.DATABASES['default']['PASSWORD']
    DB_HOST = settings.DATABASES['default']['HOST']
    DB_PORT = settings.DATABASES['default']['PORT']
    DB_NAME = settings.DATABASES['default']['NAME']
    DB_PASS_ENC = quote_plus(DB_PASS)
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS_ENC}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(conn_str)


@lru_cache(maxsize=1)
def load_tickets():
    engine = get_engine()
    df = pd.read_sql(
        "SELECT id, short_description, resolution_code, description, keywords, solution FROM tickets_final",
        engine
    )
    df["combined"] = df["description"].fillna("") + " " + df["keywords"].fillna("")
    return df


@lru_cache(maxsize=5000)
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word)[:2]:
        for lemma in syn.lemmas()[:3]:
            synonym = lemma.name().replace("_", " ")
            if synonym.lower() != word.lower():
                synonyms.append(synonym)
    return list(set(synonyms))


def expand_query(query):
    words = query.split()
    expanded = []
    pos_tags = nltk.pos_tag(words)
    for word, pos in pos_tags:
        expanded.append(word)
        if pos.startswith("N") or pos.startswith("V"):
            expanded.extend(get_synonyms(word))
    return list(set(expanded))


def normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.ones_like(scores) * 0.5
    return (scores - scores.min()) / (scores.max() - scores.min())


def semantic_search(query, top_k=3):
    """
    Performs semantic retrieval using pgvector and stored embeddings.
    """
    try:
        ensure_ollama_running()
        engine = get_engine()
        query_embedding = generate_embedding(query)
        if not query_embedding:
            raise ValueError("Query embedding is None.")

        query_vec_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = text(f"""
            SELECT id, short_description, resolution_code, solution,
                   1 - (embedding <=> '{query_vec_str}') AS similarity
            FROM tickets_final
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> '{query_vec_str}'
            LIMIT :limit
        """)

        with engine.connect() as conn:
            rows = conn.execute(sql, {"limit": top_k}).fetchall()

        results = [
            {
                "short_description": row.short_description,
                "resolution_code": row.resolution_code,
                "solution": row.solution,
                "score": float(row.similarity)
            }
            for row in rows
        ]

        return results

    except Exception as e:
        return [{
            "short_description": "Semantic Search Error",
            "resolution_code": "SEM-ERR",
            "solution": str(e),
            "score": 0.0
        }]


def chatbot_search(query, top_k=3):
    tickets_df = load_tickets()
    results = {}
    expanded_query = " ".join(expand_query(query))

    # --- TF-IDF ---
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(tickets_df["combined"].fillna(""))
    query_vec = vectorizer.transform([expanded_query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    tfidf_norm = normalize(cosine_similarities)
    top_idx = tfidf_norm.argsort()[-top_k:][::-1]
    results["tfidf"] = [
        {
            "short_description": tickets_df.iloc[idx]["short_description"],
            "resolution_code": tickets_df.iloc[idx]["resolution_code"],
            "solution": tickets_df.iloc[idx]["solution"],
            "score": float(tfidf_norm[idx])
        }
        for idx in top_idx
    ]

    # --- BM25 ---
    bm25_corpus = [doc.split() for doc in tickets_df["combined"].fillna("")]
    bm25 = BM25L(bm25_corpus)
    tokenized_query = expanded_query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_norm = normalize(bm25_scores)
    top_idx = sorted(range(len(bm25_norm)), key=lambda i: bm25_norm[i], reverse=True)[:top_k]
    results["bm25"] = [
        {
            "short_description": tickets_df.iloc[idx]["short_description"],
            "resolution_code": tickets_df.iloc[idx]["resolution_code"],
            "solution": tickets_df.iloc[idx]["solution"],
            "score": float(bm25_norm[idx])
        }
        for idx in top_idx
    ]

    # --- Hybrid ---
    hybrid_scores = [(i, 0.6 * bm25_norm[i] + 0.4 * tfidf_norm[i]) for i in range(len(tickets_df))]
    top_idx = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_k]
    results["hybrid"] = [
        {
            "short_description": tickets_df.iloc[idx]["short_description"],
            "resolution_code": tickets_df.iloc[idx]["resolution_code"],
            "solution": tickets_df.iloc[idx]["solution"],
            "score": float(score)
        }
        for idx, score in top_idx
    ]

    # --- RapidFuzz ---
    choices = tickets_df["combined"].tolist()
    matches = process.extract(expanded_query, choices, scorer=fuzz.token_sort_ratio, limit=top_k)
    results["rapidfuzz"] = [
        {
            "short_description": tickets_df.iloc[idx]["short_description"],
            "resolution_code": tickets_df.iloc[idx]["resolution_code"],
            "solution": tickets_df.iloc[idx]["solution"],
            "score": score / 100.0
        }
        for match, score, idx in matches
    ]

    # --- Semantic (pgvector) ---
    results["semantic"] = semantic_search(query, top_k=top_k)

    # --- RAG LLM Synthesis ---
    try:
        ensure_ollama_running()
        top_tickets = results["semantic"]

        rag_prompt = "You are an expert support assistant.\n"
        rag_prompt += "Analyze the following tickets' solutions and generate a synthesized answer to the user's query.\n"
        rag_prompt += "Do NOT include ``` or any code formatting.\n"
        rag_prompt += "Return the output in the following format exactly:\n"
        rag_prompt += "- short_description: \"Synthesized Solution\"\n"
        rag_prompt += "- resolution_code: \"LLM-SYNTH\"\n"
        rag_prompt += "- solution: \"Your synthesized solution here.\"\n"
        rag_prompt += "- score: 1.0\n\n"
        # Build the context string for the LLM
        for t in top_tickets:
            rag_prompt += f"- Short description: {t['short_description']}\n"
            rag_prompt += f"  Resolution code: {t['resolution_code']}\n"
            rag_prompt += f"  Solution: {t['solution']}\n\n"

        rag_prompt += f"User query: {query}\n\n"
        rag_prompt += "Return strictly in the specified format, one synthesized ticket."

        rag_output = generate_rag_response(rag_prompt,query)

        # Parse output into dictionary
        pattern = r"- short_description:\s*\"(.*?)\".*?- resolution_code:\s*\"(.*?)\".*?- solution:\s*\"(.*?)\".*?- score:\s*([0-9.]+)"
        match = re.search(pattern, rag_output, re.DOTALL)
        if match:
            results["rag_llm"] = [{
                "short_description": match.group(1),
                "resolution_code": match.group(2),
                "solution": match.group(3),
                "score": float(match.group(4))
            }]
        else:
            results["rag_llm"] = [{
                "short_description": "LLM output error",
                "resolution_code": "LLM-ERR",
                "solution": rag_output,
                "score": 0.0
            }]

    except Exception as e:
        results["rag_llm"] = [{
            "short_description": "LLM error",
            "resolution_code": "LLM-ERR",
            "solution": str(e),
            "score": 0.0
        }]

    return results
