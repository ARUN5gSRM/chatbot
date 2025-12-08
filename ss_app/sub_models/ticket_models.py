# ss_app/models/ticket_models.py
from django.db import models
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

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
    embedding = ArrayField(models.FloatField(), size=768, null=True, blank=True)

    class Meta:
        db_table = "tickets_final"

    def __str__(self):
        return f"Ticket {self.id}"
