from django.contrib import admin
from .models import Ticket

@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):
    list_display = ("short_description", "title_or_description", "uploaded_by", "uploaded_at")
    search_fields = ("short_description", "description", "keywords", "solution")
    list_filter = ("category", "issue", "rca")

    def title_or_description(self, obj):
        return obj.short_description or obj.description
    title_or_description.short_description = "Description"
