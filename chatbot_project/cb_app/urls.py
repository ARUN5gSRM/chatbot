from django.urls import path
from .sub_views.home_view import (
    signup_view, login_view, logout_view,
    index_view, upload_view,
)
from .sub_views.chatbot_view import chatbot_view, clear_chat_view
from .sub_views import ticket_view  # new auto ticket generator view
from .sub_views.pdf_upload_view import upload_pdf_view
from .sub_views.pdf_chat_view import pdf_chat_view
app_name = "cb_app"

urlpatterns = [
    # 🏠 Home and Authentication
    path("", index_view, name="index"),
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),

    # 📤 Upload Page
    path("upload/", upload_view, name="upload"),

    # 💬 Chatbot Interface
    path("chatbot/", chatbot_view, name="chatbot"),
    path("chat/clear-history/", clear_chat_view, name="chat_clear"),

    # 🎫 Auto Ticket Generation
    path("auto-ticket/", ticket_view.generate_ticket_from_chat, name="auto_ticket"),
    path("auto-ticket/<str:ticket_id>/", ticket_view.auto_ticket_summary_view, name="auto_ticket_summary"),

    # pdf
    path("upload-pdf/", upload_pdf_view, name="upload_pdf"),
    path("pdf-chat/", pdf_chat_view, name="pdf_chat"),
]
