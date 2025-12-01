# cb_app/views.py
"""
Central import hub for view callables.

This file re-exports view functions from the sub_views package so you can
`from cb_app import views` or wire them in urls.py easily.

Only imports (no logic).
"""
from .sub_views.home_view import index_view

from .sub_views.auth_view import signup_view, login_view, logout_view

from .sub_views.chatbot_view import chatbot_view, clear_chat_view

from .sub_views.upload_view import upload_view

from .sub_views.pdf_upload_view import upload_pdf_view

from .sub_views.pdf_chat_view import pdf_chat_view

from .sub_views.ticket_view import generate_ticket_from_chat, auto_ticket_summary_view

# Optional API views — import if present (fail gracefully if not)
try:
    from .sub_views.api_chat_view import api_chat
except Exception:
    api_chat = None

try:
    from .sub_views.api_pdf_view import api_pdf_search
except Exception:
    api_pdf_search = None


__all__ = [
    "index_view",
    "signup_view",
    "login_view",
    "logout_view",
    "chatbot_view",
    "clear_chat_view",
    "upload_view",
    "upload_pdf_view",
    "pdf_chat_view",
    "generate_ticket_from_chat",
    "auto_ticket_summary_view",
]

# include optional API exports only if they were imported successfully
if api_chat is not None:
    __all__.append("api_chat")
if api_pdf_search is not None:
    __all__.append("api_pdf_search")
