import fitz
import re
from ollama import Client
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from ..models import PDFDocument, PDFChunk

ollama_client = Client()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_blocks = []
    for page in doc:
        txt = re.sub(r"\s+", " ", page.get_text("text")).strip()
        if txt:
            text_blocks.append(txt)
    return "\n\n".join(text_blocks)

def chunk_text(text, chunk_size=800, overlap=100):
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    return [c for c in chunks if c]

def embed_texts(texts, model="nomic-embed-text"):
    embeddings = []
    for txt in texts:
        try:
            resp = ollama_client.embeddings(model=model, prompt="search_document: " + txt)
            embeddings.append(resp.get("embedding"))
        except Exception as e:
            print(f"[WARN] Embedding failed: {e}")
            embeddings.append([0.0] * 768)
    return embeddings

@login_required
def upload_pdf_view(request):
    if request.method == "POST":
        pdf_file = request.FILES.get("pdf")
        if not pdf_file:
            messages.error(request, "Please upload a valid PDF.")
            return redirect("cb_app:upload_pdf")

        pdf_doc = PDFDocument.objects.create(
            uploaded_by=request.user, title=pdf_file.name, pdf_file=pdf_file
        )
        text = extract_text_from_pdf(pdf_doc.pdf_file.path)
        chunks = chunk_text(text)
        embeddings = embed_texts(chunks)

        for t, e in zip(chunks, embeddings):
            PDFChunk.objects.create(document=pdf_doc, text=t, embedding=e)

        messages.success(request, f"✅ {pdf_file.name} uploaded and processed successfully!")
        return redirect("cb_app:pdf_chat")

    return render(request, "cb_app/upload_pdf.html")
