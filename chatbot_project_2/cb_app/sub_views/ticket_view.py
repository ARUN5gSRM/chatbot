from django.shortcuts import render, get_object_or_404
from ..models import RegisteredTicket

def register_ticket_success_view(request, ticket_id):
    ticket = get_object_or_404(RegisteredTicket, ticket_id=ticket_id)
    return render(request, "register_ticket_success.html", {"ticket": ticket})
