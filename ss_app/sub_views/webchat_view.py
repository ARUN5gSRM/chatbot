# ss_app/sub_views/webchat_view.py
import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods

from ss_app.logic.retriever_logic import semantic_search


@login_required
@require_http_methods(["GET", "POST"])
def webchat_view(request):
    if request.method == "POST":
        try:
            if request.content_type and request.content_type.startswith("application/json"):
                payload = json.loads(request.body)
                query = (payload.get("query") or "").strip()
            else:
                query = (request.POST.get("query") or "").strip()
        except Exception:
            return HttpResponseBadRequest("Invalid payload")

        if not query:
            return JsonResponse({"error": "Empty query"}, status=400)

        results = semantic_search(query, top_k=5)
        best = results[0] if results else None

        # JSON response for AJAX client
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({
                "query": query,
                "answer": best["best_sentence"] if best else None,
                "source": best["page_url"] if best else None,
                "candidates": results,
                "message": best["best_sentence"] if best else "No relevant information found."
            })

        # Standard POST (non-AJAX)
        return render(request, "ss_app/webchat.html", {
            "query": query,
            "answer": best["best_sentence"] if best else None,
            "source": best["page_url"] if best else None,
            "results": results,
        })

    # GET: empty chat page
    return render(request, "ss_app/webchat.html")
