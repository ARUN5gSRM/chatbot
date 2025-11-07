from django.db import models
from django.contrib.auth.models import User
from pgvector.django import VectorField
import datetime

# ===================================================================
# 1️⃣ MAIN KNOWLEDGE BASE TABLE — USED FOR CHATBOT
# ===================================================================
class Ticket(models.Model):
    id = models.AutoField(primary_key=True)
    short_description = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    keywords = models.TextField(null=True, blank=True)
    solution = models.TextField(null=True, blank=True)
    category = models.TextField(null=True, blank=True)
    issue = models.TextField(null=True, blank=True)
    rca = models.TextField(null=True, blank=True)
    uploaded_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    embedding = VectorField(dimensions=768, null=True, blank=True)  # pgvector column

    class Meta:
        db_table = "tickets_final"

    def __str__(self):
        return self.short_description or f"Ticket {self.id}"

    def combined(self):
        return " ".join(filter(None, [self.short_description, self.description, self.keywords]))


# ===================================================================
# 2️⃣ AUTO-GENERATED SUPPORT TICKET
# ===================================================================
class AutoTicket(models.Model):
    ticket_id = models.CharField(max_length=30, unique=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    user_name = models.CharField(max_length=100, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    # Chatbot Context
    query_text = models.TextField()
    last_bot_responses = models.JSONField(default=dict)
    retrieved_results = models.JSONField(default=dict)
    similarity_threshold_used = models.FloatField(default=0.8)
    chat_history = models.JSONField(default=dict)

    # Support Metadata
    assigned_to = models.CharField(max_length=100, default="Tech Support")
    priority = models.CharField(max_length=20, default="MEDIUM")

    class Meta:
        db_table = "auto_tickets"

    def __str__(self):
        return f"{self.ticket_id} - {self.user_name or 'Anonymous'}"

    @staticmethod
    def generate_ticket_id():
        """Generate ticket ID like TCKT-2025-00001"""
        now = datetime.datetime.now()
        prefix = "TCKT"
        year = now.year
        base = AutoTicket.objects.count() + 1
        return f"{prefix}-{year}-{base:05d}"


# ===================================================================
# 3️⃣ PDF RAG MODELS
# ===================================================================
class PDFDocument(models.Model):
    id = models.AutoField(primary_key=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=512)
    pdf_file = models.FileField(upload_to="pdfs/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "pdf_documents"

    def __str__(self):
        return f"{self.title} (by {self.uploaded_by})"


class PDFChunk(models.Model):
    document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name="chunks")
    text = models.TextField()
    embedding = VectorField(dimensions=768, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    token_start = models.IntegerField(null=True, blank=True)
    token_end = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "pdf_chunks"

    def __str__(self):
        return f"Chunk {self.id} of {self.document.title}"
