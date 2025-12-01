# cb_app/sub_views/chatbot_view.py
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

from cb_app.logic.index_manager import faiss_manager
from cb_app.logic.chatbot_core import chatbot_search
from cb_app.logic.session_helpers import get_recent_conversation, clear_conversation
from cb_app.models import Ticket

NAMESPACE_TICKETS_BASE = "tickets_session_"


def _ensure_session_namespace(request) -> str:
    """
    Ensure the current HTTP session has a DB-backed session_key and a
    session-scoped FAISS namespace built from DB embeddings (lazy build).
    Returns the namespace name.
    """
    # Ensure session exists and has a session_key
    if not request.session.session_key:
        request.session.save()
    session_key = request.session.session_key
    ns = f"{NAMESPACE_TICKETS_BASE}{session_key}"

    idx = faiss_manager.get(ns)
    # Build lazily if empty
    if idx.index.ntotal == 0:
        def fetch():
            qs = Ticket.objects.filter(embedding__isnull=False).values_list("id", "embedding")
            return list(qs)
        faiss_manager.build_from_db(ns, fetch)
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
        # Accept both JSON (AJAX) and form POST
        try:
            if request.content_type == "application/json":
                payload = json.loads(request.body.decode("utf-8") or "{}")
                query = (payload.get("query") or "").strip()
            else:
                query = (request.POST.get("query") or "").strip()
        except Exception:
            return HttpResponseBadRequest("Invalid request payload")

        if query:
            # ensure session-scoped index exists and is built
            _ensure_session_namespace(request)

            # Run chatbot search (this will update session history inside)
            results = chatbot_search(request, query)

            # If AJAX/json requested, return JSON
            if request.content_type == "application/json" or request.headers.get("x-requested-with") == "XMLHttpRequest":
                return JsonResponse(results)

    else:
        results = {"chat_history": get_recent_conversation(request)}

    return render(request, "cb_app/chatbot.html", {"results": results, "query": query})


@login_required
@require_http_methods(["POST", "GET"])
def clear_chat_view(request):
    """
    Clear conversation session state and remove session-scoped FAISS index.
    Prefer POST for destructive action (form in template uses POST).
    """
    # Clear session conversation and related keys centrally
    clear_conversation(request)

    # Remove session-scoped FAISS index if present
    if not request.session.session_key:
        request.session.save()
    session_key = request.session.session_key
    if session_key:
        tickets_ns = f"{NAMESPACE_TICKETS_BASE}{session_key}"
        if tickets_ns in faiss_manager.indices:
            faiss_manager.indices.pop(tickets_ns, None)

    return redirect("cb_app:chatbot")
