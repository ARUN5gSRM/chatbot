# cb_app/sub_views/home_view.py
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def index_view(request):
    return render(request, "cb_app/index.html")
