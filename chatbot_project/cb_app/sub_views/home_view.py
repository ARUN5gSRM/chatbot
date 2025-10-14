from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.conf import settings

import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download('punkt_tab')
nltk.download("stopwords")


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


# ----------------- Excel Upload -----------------
@login_required
def upload_view(request):
    message = ""
    if request.method == "POST" and request.FILES.get("file"):
        excel_file = request.FILES["file"]
        try:
            # 1️⃣ Read Excel
            df = pd.read_excel(excel_file)
            df = df.fillna("")

            # 2️⃣ Normalize column names
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # 3️⃣ Parse resolution_notes
            def parse_resolution_notes(notes):
                category, issue, rca, solution = "", "", "", ""
                cat_match = re.search(r"category[:\-]\s*(.+)", str(notes), re.IGNORECASE)
                issue_match = re.search(r"issue[:\-]\s*(.+)", str(notes), re.IGNORECASE)
                rca_match = re.search(r"rca[:\-]\s*(.+)", str(notes), re.IGNORECASE)
                sol_match = re.search(r"solution[:\-]\s*(.+)", str(notes), re.IGNORECASE)
                if cat_match: category = cat_match.group(1).strip()
                if issue_match: issue = issue_match.group(1).strip()
                if rca_match: rca = rca_match.group(1).strip()
                if sol_match:
                    solution = sol_match.group(1).strip()
                elif notes:
                    solution = str(notes).strip()
                return pd.Series([category, issue, rca, solution])

            if "resolution_notes" in df.columns:
                df[["category", "issue", "rca", "solution"]] = df["resolution_notes"].apply(parse_resolution_notes)

            # 4️⃣ Extract keywords
            stop_words = set(stopwords.words("english"))

            def extract_keywords(text):
                tokens = word_tokenize(str(text).lower())
                keywords = [w for w in tokens if w.isalpha() and w not in stop_words]
                return " ".join(keywords)

            if "description" in df.columns and "solution" in df.columns:
                df["keywords"] = (df["description"].fillna("") + " " + df["solution"].fillna("")).apply(
                    extract_keywords)

            # 5️⃣ Add uploaded_by
            df["uploaded_by_id"] = request.user.id

            # 6️⃣ Save to PostgreSQL
            DB_USER = settings.DATABASES['default']['USER']
            DB_PASS = settings.DATABASES['default']['PASSWORD']
            DB_HOST = settings.DATABASES['default']['HOST']
            DB_PORT = settings.DATABASES['default']['PORT']
            DB_NAME = settings.DATABASES['default']['NAME']
            DB_PASS_ENC = quote_plus(DB_PASS)
            conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS_ENC}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(conn_str)

            df.to_sql(
                name="tickets_final",
                con=engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000
            )

            message = f"File '{excel_file.name}' uploaded successfully to database!"
        except Exception as e:
            message = f"Error uploading file: {e}"

    return render(request, "upload.html", {"message": message})


# ----------------- Chatbot -----------------
@login_required
def chatbot_view(request):
    results = {}
    query = ""
    if request.method == "POST":
        query = request.POST.get("query", "")
        if query.strip():
            # Placeholder for actual chatbot function
            results = {"example": [{"short_description": "Demo", "solution": "Demo solution", "score": 1.0}]}

    return render(request, "chatbot.html", {"results": results, "query": query})

