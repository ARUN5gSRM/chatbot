# ss_app/sub_views/auth_view.py
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required

from ss_app.logic.index_manager import faiss_manager

def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Account created. Please log in.")
            return redirect("ss_app:login")
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = UserCreationForm()
    # templates is under ss_app/templates/ss_app/signup.html
    return render(request, "ss_app/signup.html", {"form": form})

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome {user.username}!")
            return redirect("ss_app:index")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    # templates is under ss_app/templates/ss_app/login.html
    return render(request, "ss_app/login.html", {"form": form})

@login_required
def logout_view(request):
    # On logout, clear any session-scoped FAISS indices for this session
    session_key = request.session.session_key
    if session_key:
        tickets_ns = f"tickets_session_{session_key}"
        pdf_ns = f"pdf_session_{session_key}"
        # clear and remove indices if present
        if tickets_ns in faiss_manager.indices:
            faiss_manager.indices.pop(tickets_ns, None)
        if pdf_ns in faiss_manager.indices:
            faiss_manager.indices.pop(pdf_ns, None)
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect("ss_app:login")
