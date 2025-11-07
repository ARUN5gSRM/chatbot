import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from cb_app.models import AutoTicket


@login_required
def generate_ticket_from_chat(request):
    """Generate a support ticket using the most recent chatbot interaction."""
    # ✅ Retrieve session-stored chat context
    chat_history = request.session.get("chat_history", [])
    last_query = request.session.get("last_query", "")
    last_threshold = request.session.get("last_threshold", 0.8)
    last_results = request.session.get("last_results", [])
    last_bot_response = request.session.get("last_bot_response", "")
    user = request.user

    # Extract last few assistant responses
    last_bot_responses = [m["content"] for m in chat_history if m["role"] == "assistant"][-5:]

    # Priority logic
    priority = "HIGH" if any(
        k in last_query.lower() for k in ["urgent", "critical", "fail", "error"]
    ) else "MEDIUM"

    # ✅ Create and save AutoTicket entry
    ticket = AutoTicket.objects.create(
        ticket_id=AutoTicket.generate_ticket_id(),
        user=user,
        user_name=user.username,
        timestamp=timezone.now(),
        query_text=last_query or "No query captured",
        last_bot_responses={"responses": last_bot_responses or [last_bot_response]},
        retrieved_results={"ids": last_results},
        similarity_threshold_used=last_threshold,
        chat_history={"conversation": chat_history[-10:]},
        assigned_to="Tech Support",
        priority=priority,
    )

    # Redirect to summary
    return redirect("cb_app:auto_ticket_summary", ticket_id=ticket.ticket_id)


@login_required
def auto_ticket_summary_view(request, ticket_id):
    """Display ticket summary after automatic generation."""
    ticket = get_object_or_404(AutoTicket, ticket_id=ticket_id)
    return render(request, "cb_app/auto_ticket_summary.html", {"ticket": ticket})
