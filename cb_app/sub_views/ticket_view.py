# cb_app/sub_views/ticket_view.py
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.contrib import messages

from cb_app.models import AutoTicket

def _fallback_ticket_id():
    """
    Create a fallback ticket id similar to TCKT-<YEAR>-<zero5count>.
    Uses the current number of AutoTicket rows to make a reasonably unique id.
    """
    now = timezone.now()
    year = now.year
    try:
        base = AutoTicket.objects.count() + 1
    except Exception:
        base = 1
    return f"TCKT-{year}-{base:05d}"


@login_required
def generate_ticket_from_chat(request):
    """
    Generate a support ticket using the most recent chatbot interaction saved in session.
    Accepts GET or POST, but preferable to use POST for a destructive/create action.
    """
    # prefer POST, but allow GET in this app pattern
    # Retrieve session-stored chat context
    chat_history = request.session.get("chat_history", [])
    last_query = request.session.get("last_query", "") or ""
    last_threshold = request.session.get("last_threshold", 0.8)
    last_results = request.session.get("last_results", []) or []
    last_bot_response = request.session.get("last_bot_response", "")

    user = request.user

    # Extract last few assistant responses
    last_bot_responses = [m["content"] for m in chat_history if m.get("role") == "assistant"][-5:]
    if not last_bot_responses and last_bot_response:
        last_bot_responses = [last_bot_response]

    # Priority logic (simple)
    priority = "HIGH" if any(
        k in last_query.lower() for k in ["urgent", "critical", "fail", "error"]
    ) else "MEDIUM"

    # Generate ticket id — use model method if present, otherwise fallback
    try:
        ticket_id = AutoTicket.generate_ticket_id()
    except Exception:
        ticket_id = _fallback_ticket_id()

    # Create and save AutoTicket entry
    try:
        ticket = AutoTicket.objects.create(
            ticket_id=ticket_id,
            user=user,
            user_name=getattr(user, "username", "") or "Anonymous",
            timestamp=timezone.now(),
            query_text=last_query or "No query captured",
            last_bot_responses={"responses": last_bot_responses or [last_bot_response or ""]},
            retrieved_results={"ids": last_results},
            similarity_threshold_used=float(last_threshold or 0.8),
            chat_history={"conversation": chat_history[-10:] if chat_history else []},
            assigned_to="Tech Support",
            priority=priority,
        )
    except Exception as e:
        # log or bubble up; for now surface a friendly message and redirect back
        messages.error(request, f"Failed to create ticket: {e}")
        return redirect("cb_app:chatbot")

    messages.success(request, f"Ticket {ticket.ticket_id} created.")
    return redirect("cb_app:auto_ticket_summary", ticket_id=ticket.ticket_id)


@login_required
def auto_ticket_summary_view(request, ticket_id):
    """Display ticket summary after automatic generation."""
    ticket = get_object_or_404(AutoTicket, ticket_id=ticket_id)
    return render(request, "cb_app/auto_ticket_summary.html", {"ticket": ticket})
