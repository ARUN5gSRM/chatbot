from django.urls import path
from .sub_views.home_view import (
    signup_view, login_view, logout_view,
    index_view, upload_view, register_ticket_view,
)
from .sub_views.chatbot_view import chatbot_view
from .sub_views.ticket_view import register_ticket_success_view  # ✅ add this import

app_name = "cb_app"

urlpatterns = [
    path("", index_view, name="index"),
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("upload/", upload_view, name="upload"),
    path("chat/", chatbot_view, name="chatbot"),
    path("register-ticket/", register_ticket_view, name="register_ticket"),
    path("register-ticket/success/<str:ticket_id>/", register_ticket_success_view, name="register_ticket_success"),


]
