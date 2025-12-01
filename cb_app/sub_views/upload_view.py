# cb_app/sub_views/upload_view.py
import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib import messages

from cb_app.sub_models.data_ingest import ingest_excel_file
from cb_app.sub_models.index_manager import faiss_manager

@login_required
def upload_view(request):
    message = ""
    if request.method == "POST":
        # Support AJAX/form file upload
        excel_file = request.FILES.get("file")
        if not excel_file:
            message = "No file uploaded."
            if request.content_type == "application/json":
                return JsonResponse({"error": message}, status=400)
            return render(request, "cb_app/upload.html", {"message": message})

        try:
            res = ingest_excel_file(excel_file, request.user)
            message = f"✅ Uploaded. Inserted {res.get('created_count',0)} rows."

            # Incrementally add new embeddings to any active session-scoped indices
            # Each active session index whose name startswith 'tickets_session_' gets the new vectors
            # We must query DB for embeddings of returned ids
            ids = res.get("ids", [])
            if ids:
                from cb_app.models import Ticket
                rows = list(Ticket.objects.filter(id__in=ids).values_list("id", "embedding"))
                # rows: list of (id, emb)
                # For each active index, add the new vectors
                for ns in list(faiss_manager.indices.keys()):
                    if ns.startswith("tickets_session_"):
                        object_ids = [r[0] for r in rows if r[1]]
                        vectors = [r[1] for r in rows if r[1]]
                        if object_ids and vectors:
                            faiss_manager.add(ns, object_ids, vectors)

            if request.content_type == "application/json" or request.headers.get("x-requested-with") == "XMLHttpRequest":
                return JsonResponse({"message": message})
        except Exception as e:
            message = f"❌ Error: {e}"
            if request.content_type == "application/json":
                return JsonResponse({"error": message}, status=500)
            messages.error(request, message)

    return render(request, "cb_app/upload.html", {"message": message})
