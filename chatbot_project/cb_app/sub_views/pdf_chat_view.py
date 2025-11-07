from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from ollama import Client
from pgvector.django import CosineDistance
from ..models import PDFChunk

ollama_client = Client()

@login_required
def pdf_chat_view(request):
    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if not query:
            return render(request, "cb_app/pdf_chat.html", {"error": "Please enter a question."})

        try:
            emb = ollama_client.embeddings(model="nomic-embed-text", prompt="search_query: " + query)
            q_vec = emb.get("embedding")
        except Exception as e:
            return render(request, "cb_app/pdf_chat.html", {"error": f"Embedding failed: {e}"})

        # 🔍 Retrieve top chunks using cosine similarity only
        top_chunks = PDFChunk.objects.annotate(
            distance=CosineDistance("embedding", q_vec)
        ).order_by("distance")[:3]

        if not top_chunks:
            return render(request, "cb_app/pdf_chat.html", {"error": "No relevant information found."})

        # Combine top retrieved chunks for display
        results = []
        for i, c in enumerate(top_chunks, start=1):
            results.append({
                "rank": i,
                "text": c.text,
                "similarity": round(1 - float(c.distance), 4)
            })

        # Instead of generating an AI answer, return retrieved chunks directly
        return render(request, "cb_app/pdf_chat.html", {
            "query": query,
            "results": results,
            "context_text": "\n\n---\n\n".join([r["text"] for r in results])
        })

    return render(request, "cb_app/pdf_chat.html")
