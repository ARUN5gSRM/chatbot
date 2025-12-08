# cb_app/sub_views/pdf_chat_view.py

import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

from cb_app.logic.index_manager import faiss_manager
from cb_app.logic.pdf_core import pdf_search
from cb_app.models import PDFChunk

NAMESPACE_PDF_BASE = "pdf_session_"


def _ensure_pdf_namespace(request):
    """
    Ensures a FAISS index exists for the current session.
    Loads all PDFChunk embeddings from DB into the session-scoped index.
    Uses SAFE FAISS wrappers.
    """
    if not request.session.session_key:
        request.session.save()

    session_key = request.session.session_key
    namespace = f"{NAMESPACE_PDF_BASE}{session_key}"

    # Ensure FAISS index object exists
    faiss_manager.safe_get_or_create(namespace)

    # Build index safely if empty
    faiss_manager.safe_build_from_db_if_empty(
        namespace,
        lambda: list(
            PDFChunk.objects.filter(embedding__isnull=False)
            .values_list("id", "embedding")
        )
    )

    return namespace


@login_required
@require_http_methods(["GET", "POST"])
def pdf_chat_view(request):
    """
    PDF similarity search using FAISS cosine similarity.
    Falls back to SQL cosine if FAISS is empty.
    SAFE FAISS integration.
    """

    if request.method == "POST":
        try:
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

        # SAFE: build or load FAISS index
        ns = _ensure_pdf_namespace(request)

        # Perform PDF similarity search using session FAISS namespace
        results = pdf_search(query, top_k=3, namespace=ns)

        if request.content_type == "application/json" or request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse({"query": query, "results": results})

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

    return render(request, "cb_app/pdf_chat.html")
