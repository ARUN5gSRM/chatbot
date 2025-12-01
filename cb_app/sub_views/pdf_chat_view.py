# cb_app/sub_views/pdf_chat_view.py

import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

from cb_app.sub_models.index_manager import faiss_manager
from cb_app.sub_models.pdf_core import pdf_search
from cb_app.models import PDFChunk

# Each session gets its own FAISS namespace
NAMESPACE_PDF_BASE = "pdf_session_"


def _ensure_pdf_namespace(request):
    """
    Ensures a FAISS index exists for the current session.
    Loads all PDFChunk embeddings from DB into the session-scoped index.
    """
    if not request.session.session_key:
        request.session.save()

    session_key = request.session.session_key
    namespace = f"{NAMESPACE_PDF_BASE}{session_key}"

    index = faiss_manager.get(namespace)

    # Lazy-build the FAISS index if empty
    if index.index.ntotal == 0:
        def fetch():
            # Retrieves (id, embedding_list)
            qs = PDFChunk.objects.filter(embedding__isnull=False).values_list("id", "embedding")
            return list(qs)

        faiss_manager.build_from_db(namespace, fetch)

    return namespace


@login_required
@require_http_methods(["GET", "POST"])
def pdf_chat_view(request):
    """
    PDF similarity search using FAISS cosine similarity.
    Falls back to SQL cosine if FAISS is empty.
    """

    # POST = user submitted a question
    if request.method == "POST":
        try:
            # Accept both JSON (AJAX) and form POST
            if request.content_type == "application/json":
                payload = json.loads(request.body)
                query = payload.get("query", "").strip()
            else:
                query = request.POST.get("query", "").strip()

        except Exception:
            return HttpResponseBadRequest("Invalid request payload")

        if not query:
            return render(request, "cb_app/pdf_chat.html", {
                "error": "Please enter a question."
            })

        # Build or load FAISS index for the session
        _ensure_pdf_namespace(request)

        # Use pdf_core.pdf_search (FAISS first, SQL fallback)
        results = pdf_search(query, top_k=3)

        # If AJAX request → return JSON
        if request.content_type == "application/json" or request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"query": query, "results": results})

        # Render template results
        if not results:
            return render(request, "cb_app/pdf_chat.html", {
                "query": query,
                "error": "No relevant information found."
            })

        return render(request, "cb_app/pdf_chat.html", {
            "query": query,
            "results": results,
            "context_text": "\n\n---\n\n".join([r["text"] for r in results])
        })

    # GET request → empty chat interface
    return render(request, "cb_app/pdf_chat.html")
