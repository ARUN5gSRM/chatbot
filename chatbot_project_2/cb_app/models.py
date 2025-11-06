from django.db import models
from django.contrib.auth.models import User
from pgvector.django import VectorField
import uuid

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
class RegisteredTicket(models.Model):
    ticket_id = models.CharField(max_length=20, unique=True, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="tickets")
    subject = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    description = models.TextField()
    priority = models.CharField(
        max_length=20,
        choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')],
        default='Medium'
    )
    status = models.CharField(max_length=50, default='Open')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "cb_app_registeredticket"
        managed = False  # ✅ Do NOT let Django manage this

    def save(self, *args, **kwargs):
        if not self.ticket_id:
            self.ticket_id = f"TCKT-{uuid.uuid4().hex[:8].upper()}"
        super().save(*args, **kwargs)