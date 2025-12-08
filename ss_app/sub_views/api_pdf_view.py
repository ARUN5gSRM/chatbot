# ss_app/sub_views/api_pdf_view.py
import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

from ss_app.logic.pdf_core import pdf_search
from ss_app.logic.index_manager import faiss_manager

@login_required
@require_POST
def api_pdf_search(request):
    try:
        payload = json.loads(request.body)
        query = payload.get("query", "").strip()
    except Exception:
        return HttpResponseBadRequest("Invalid payload")

    if not query:
        return JsonResponse({"error": "Empty query"}, status=400)

    if not request.session.session_key:
        request.session.save()
    ns = f"pdf_session_{request.session.session_key}"
    if ns not in faiss_manager.indices:
        def fetch():
            from ss_app.models import PDFChunk
            return list(PDFChunk.objects.filter(embedding__isnull=False).values_list("id", "embedding"))
        faiss_manager.build_from_db(ns, fetch)

    results = pdf_search(query, top_k=3)
    return JsonResponse({"results": results})
