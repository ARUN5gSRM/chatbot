
from django.contrib import admin
from .models import Ticket

@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):
    list_display = ("short_description", "category", "uploaded_by", "uploaded_at")
    search_fields = ("short_description", "description", "keywords", "solution")

