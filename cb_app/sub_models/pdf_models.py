# cb_app/models/pdf_models.py
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

class PDFDocument(models.Model):
    id = models.AutoField(primary_key=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    title = models.CharField(max_length=512)
    pdf_file = models.FileField(upload_to="pdfs/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "pdf_documents"

    def __str__(self):
        return self.title


class PDFChunk(models.Model):
    document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE, related_name="chunks")
    text = models.TextField()
    embedding = ArrayField(models.FloatField(), size=768, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    token_start = models.IntegerField(null=True, blank=True)
    token_end = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "pdf_chunks"

    def __str__(self):
        return f"Chunk {self.id} ({self.document.title})"
