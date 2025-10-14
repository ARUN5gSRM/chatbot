from django.db import models
from django.contrib.auth.models import User

class Ticket(models.Model):
    short_description = models.TextField(blank=True,null=True)
    description = models.TextField(blank=True,null=True)
    keywords = models.TextField(blank=True,null=True)
    solution = models.TextField(blank=True,null=True)
    category = models.CharField(max_length=255, blank=True)
    issue = models.TextField(blank=True)
    rca = models.TextField(blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def combined(self):
        # helper for search corpus
        return " ".join(filter(None, [self.short_description, self.description, self.keywords]))
