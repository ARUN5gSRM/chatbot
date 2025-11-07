# cb_app/sub_views/home_view.py

import os
import re
import nltk
import pandas as pd
from urllib.parse import quote_plus
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings
from sqlalchemy import create_engine
from django.shortcuts import render, redirect, get_object_or_404
import numpy as np
import psycopg2
import requests

from cb_app.sub_models.ollama_helper import (
    embed_text_mean,
    ensure_ollama_running,
)


# Ensure nltk resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


# ----------------- Signup -----------------
def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            messages.success(request, "Account created successfully! You can now log in.")
            return redirect("cb_app:login")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})


# ----------------- Login -----------------
def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome {user.username}!")
            return redirect("cb_app:index")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


# ----------------- Logout -----------------
@login_required
def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect("cb_app:login")


# ----------------- Home -----------------
@login_required
def index_view(request):
    return render(request, "index.html")

# ----------------- Upload + Embedding -----------------
@login_required
def upload_view(request):
    message = ""
    if request.method == "POST" and request.FILES.get("file"):
        excel_file = request.FILES["file"]
        try:
            # ------------------ Read Excel ------------------
            df = pd.read_excel(excel_file).fillna("")
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # ------------------ Parse resolution_notes ------------------
            # 3️⃣ Parse resolution_notes
            def parse_resolution_notes(notes):
                category, issue, rca, solution = "", "", "", ""

                # --- Updated regex: stops at next field (Issue, RCA, Solution, or Score)
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

                return pd.Series([category, issue, rca, solution])

            if "resolution_notes" in df.columns:
                df[["category", "issue", "rca", "solution"]] = df["resolution_notes"].apply(parse_resolution_notes)

            # ------------------ Extract keywords ------------------
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            stop_words = set(stopwords.words("english"))

            def extract_keywords(text):
                tokens = word_tokenize(str(text).lower())
                keywords = [w for w in tokens if w.isalpha() and w not in stop_words]
                return " ".join(keywords)

            if "description" in df.columns and "solution" in df.columns:
                df["keywords"] = (df["description"].fillna("") + " " + df["solution"].fillna("")).apply(extract_keywords)

            df["uploaded_by_id"] = request.user.id

            # ------------------ Store to PostgreSQL ------------------
            DB_USER = settings.DATABASES['default']['USER']
            DB_PASS = settings.DATABASES['default']['PASSWORD']
            DB_HOST = settings.DATABASES['default']['HOST']
            DB_PORT = settings.DATABASES['default']['PORT']
            DB_NAME = settings.DATABASES['default']['NAME']

            DB_PASS_ENC = quote_plus(DB_PASS)
            conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS_ENC}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(conn_str)

            df.to_sql("tickets_final", con=engine, if_exists="append", index=False, method="multi", chunksize=1000)
            message = f"✅ File '{excel_file.name}' uploaded successfully!"
            print(f"[INFO] Inserted {len(df)} rows into tickets_final via to_sql().")

            # ------------------ Ensure Ollama ------------------
            try:
                ensure_ollama_running()
                print("[INFO] Ollama server is running ✅")
            except Exception as e:
                message = f"❌ Ollama server not running: {e}"
                print(message)
                return render(request, "upload.html", {"message": message})

            # ------------------ Generate embeddings ------------------
            conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT)
            cur = conn.cursor()

            cur.execute("SELECT id, short_description, description, keywords FROM tickets_final WHERE embedding IS NULL;")
            rows = cur.fetchall()
            print(f"[DEBUG] {len(rows)} rows without embeddings.")

            for r in rows:
                ticket_id = r[0]
                combined_text = " ".join(filter(None, [r[1], r[2], r[3]])).strip()
                if not combined_text:
                    print(f"[WARN] Empty text for ticket {ticket_id}, storing zero-vector")
                    mean_emb = [0.0] * 768
                else:
                    try:
                        mean_emb = embed_text_mean(combined_text, max_chars_per_chunk=800)
                        if mean_emb is None:
                            print(f"[WARN] Ollama returned None for ticket {ticket_id}, storing zero-vector")
                            mean_emb = [0.0] * 768
                    except Exception as e:
                        print(f"[❌] embed_text_mean failed for ticket {ticket_id}: {e}")
                        mean_emb = [0.0] * 768

                # Ensure correct dimension
                if len(mean_emb) != 768:
                    print(f"[WARN] Dimension mismatch for ticket {ticket_id}, using zero-vector")
                    mean_emb = [0.0] * 768

                # Convert to pgvector format
                arr_str = "[" + ",".join(map(str, mean_emb)) + "]"

                try:
                    cur.execute(
                        "UPDATE tickets_final SET embedding = %s::vector WHERE id = %s;",
                        (arr_str, ticket_id)
                    )
                    print(f"[OK] Stored embedding for ticket {ticket_id}")
                except Exception as e:
                    print(f"[❌] Failed storing embedding for ticket {ticket_id}: {e}")

            try:
                conn.commit()
                print("✅ Embeddings committed to DB.")
            except Exception as e:
                print(f"[❌] Commit failed: {e}")

            cur.close()
            conn.close()

        except Exception as e:
            message = f"❌ Error uploading file: {e}"
            print(f"[EXCEPTION] {e}")

    return render(request, "upload.html", {"message": message})
