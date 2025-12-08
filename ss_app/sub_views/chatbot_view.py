# ss_app/sub_views/chatbot_view.py
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

from ss_app.logic.index_manager import faiss_manager
from ss_app.logic.chatbot_core import chatbot_search
from ss_app.logic.session_helpers import get_recent_conversation, clear_conversation
from ss_app.models import Ticket

NAMESPACE_TICKETS_BASE = "tickets_session_"


def _ensure_session_namespace(request) -> str:
    """
    Ensure the current HTTP session has a DB-backed session_key and a
    session-scoped FAISS namespace built from DB embeddings (lazy build).
    Uses SAFE FAISS wrappers.
    Returns the namespace name.
    """
    if not request.session.session_key:
        request.session.save()

    session_key = request.session.session_key
    ns = f"{NAMESPACE_TICKETS_BASE}{session_key}"

    # SAFE: ensures index exists
    faiss_manager.safe_get_or_create(ns)

    # SAFE: build from DB if empty
    faiss_manager.safe_build_from_db_if_empty(
        ns,
        lambda: list(
            Ticket.objects.filter(embedding__isnull=False).values_list("id", "embedding")
        )
    )

    return ns


@login_required
@require_http_methods(["GET", "POST"])
def chatbot_view(request):
    """
    Renders chatbot page on GET.
    Accepts form POSTs or JSON/AJAX POSTs to submit queries.
    Returns JSON for AJAX requests, otherwise renders template.
    """
    query = ""
    results = {}

    if request.method == "POST":
        try:
            if request.content_type == "application/json":
                payload = json.loads(request.body.decode("utf-8") or "{}")
                query = (payload.get("query") or "").strip()
            else:
                query = (request.POST.get("query") or "").strip()

        except Exception:
            return HttpResponseBadRequest("Invalid request payload")

        if query:
            # SAFE namespace creation / FAISS build
            ns = _ensure_session_namespace(request)

            # Run chatbot search using safe FAISS namespace
            results = chatbot_search(request, query, namespace=ns)

            if request.content_type == "application/json" or request.headers.get("x-requested-with") == "XMLHttpRequest":
                return JsonResponse(results)

    else:
        results = {"chat_history": get_recent_conversation(request)}

    return render(request, "ss_app/chatbot.html", {"results": results, "query": query})


@login_required
@require_http_methods(["POST", "GET"])
def clear_chat_view(request):
    """
    Clear conversation session state and remove session-scoped FAISS index.
    SAFE removal using safe_pop().
    """
    clear_conversation(request)

    if not request.session.session_key:
        request.session.save()

    session_key = request.session.session_key
    if session_key:
        ns = f"{NAMESPACE_TICKETS_BASE}{session_key}"
        faiss_manager.safe_pop(ns)

    return redirect("ss_app:chatbot")
