# ss_app/sub_views/api_chat_view.py
import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

from ss_app.logic.chatbot_core import chatbot_search
from ss_app.logic.index_manager import faiss_manager

@login_required
@require_POST
def api_chat(request):
    try:
        payload = json.loads(request.body)
        query = payload.get("query", "").strip()
    except Exception:
        return HttpResponseBadRequest("Invalid payload")

    if not query:
        return JsonResponse({"error": "Empty query"}, status=400)

    # Ensure session-scoped index is present
    if not request.session.session_key:
        request.session.save()
    ns = f"tickets_session_{request.session.session_key}"
    if ns not in faiss_manager.indices:
        def fetch():
            from ss_app.models import Ticket
            return list(Ticket.objects.filter(embedding__isnull=False).values_list("id", "embedding"))
        faiss_manager.build_from_db(ns, fetch)

    res = chatbot_search(request, query)
    return JsonResponse(res)
