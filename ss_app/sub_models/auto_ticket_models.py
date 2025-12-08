from django.db import models
from django.contrib.auth.models import User

class AutoTicket(models.Model):
    ticket_id = models.CharField(max_length=30, unique=True)
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    user_name = models.CharField(max_length=100, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    query_text = models.TextField()
    last_bot_responses = models.JSONField(default=dict)
    retrieved_results = models.JSONField(default=dict)
    similarity_threshold_used = models.FloatField(default=0.8)
    chat_history = models.JSONField(default=dict)

    assigned_to = models.CharField(max_length=100, default="Tech Support")
    priority = models.CharField(max_length=20, default="MEDIUM")

    class Meta:
        db_table = "auto_tickets"

    def __str__(self):
        return self.ticket_id
