# cb_app/urls.py
from django.urls import path
from django.urls import include

# import required view functions from sub_views
from .sub_views.home_view import index_view
from .sub_views.auth_view import signup_view, login_view, logout_view
from .sub_views.chatbot_view import chatbot_view, clear_chat_view
from .sub_views.upload_view import upload_view
from .sub_views.pdf_upload_view import upload_pdf_view
from .sub_views.pdf_chat_view import pdf_chat_view
from .sub_views.ticket_view import generate_ticket_from_chat, auto_ticket_summary_view
from cb_app.sub_views.crawl_view import crawl_site_view
from cb_app.sub_views.webchat_view import webchat_view
# optional API views (import safely)
try:
    from .sub_views.api_chat_view import api_chat
except Exception:
    api_chat = None

try:
    from .sub_views.api_pdf_view import api_pdf_search
except Exception:
    api_pdf_search = None

app_name = "cb_app"

urlpatterns = [
    # Home + Auth
    path("", index_view, name="index"),
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),

    # Uploads
    path("upload/", upload_view, name="upload"),
    path("upload-pdf/", upload_pdf_view, name="upload_pdf"),

    # Chat UI
    path("chatbot/", chatbot_view, name="chatbot"),
    path("chat/clear-history/", clear_chat_view, name="chat_clear"),

    # PDF chat
    path("pdf-chat/", pdf_chat_view, name="pdf_chat"),

    # Auto-ticket
    path("auto-ticket/", generate_ticket_from_chat, name="auto_ticket"),
    path("auto-ticket/<str:ticket_id>/", auto_ticket_summary_view, name="auto_ticket_summary"),
    # NEW — Web Chat (fixed name)
    path("web-chat/", webchat_view, name="web_chat"),

    # NEW — Crawl Site
    path("crawl-site/", crawl_site_view, name="crawl_site"),

]

# Add optional API routes if modules are present
if api_chat is not None:
    urlpatterns += [
        path("api/chat/", api_chat, name="api_chat"),
    ]

if api_pdf_search is not None:
    urlpatterns += [
        path("api/pdf/", api_pdf_search, name="api_pdf_search"),
    ]
