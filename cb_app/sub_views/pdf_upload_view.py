# cb_app/sub_views/pdf_upload_view.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from cb_app.logic.pdf_core import extract_text_from_pdf, chunk_text, embed_texts
from cb_app.models import PDFDocument, PDFChunk
from cb_app.logic.index_manager import faiss_manager

NAMESPACE_PDF_BASE = "pdf_session_"

@login_required
def upload_pdf_view(request):
    if request.method == "POST":
        pdf_file = request.FILES.get("pdf")
        if not pdf_file:
            messages.error(request, "Please upload a valid PDF.")
            return redirect("cb_app:upload_pdf")

        pdf_doc = PDFDocument.objects.create(uploaded_by=request.user, title=pdf_file.name, pdf_file=pdf_file)
        # extract & chunk
        text = extract_text_from_pdf(pdf_doc.pdf_file.path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)

        new_ids = []
        new_vectors = []
        for t, e in zip(chunks, embeddings):
            chunk = PDFChunk.objects.create(document=pdf_doc, text=t, embedding=e)
            new_ids.append(chunk.id)
            new_vectors.append(e)

        # Add to any active pdf session indices
        for ns in list(faiss_manager.indices.keys()):
            if ns.startswith(NAMESPACE_PDF_BASE):
                faiss_manager.add(ns, new_ids, new_vectors)

        messages.success(request, f"✅ {pdf_file.name} uploaded and processed successfully!")
        return redirect("cb_app:pdf_chat")

    return render(request, "cb_app/upload_pdf.html")
