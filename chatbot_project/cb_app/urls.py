from django.urls import path
from .sub_views.home_view import signup_view, login_view, logout_view, index_view, upload_view
from .sub_views.chatbot_view import chatbot_view

app_name = "cb_app"

urlpatterns = [
    path("", index_view, name="index"),
    path("signup/", signup_view, name="signup"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("upload/", upload_view, name="upload"),
    path("chat/", chatbot_view, name="chatbot"),
]

