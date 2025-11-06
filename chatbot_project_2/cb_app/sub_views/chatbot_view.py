from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from ..sub_models.chatbot import chatbot_search

@login_required
def chatbot_view(request):
    results = {}
    query = ""
    if request.method == "POST":
        query = request.POST.get("query", "")
        if query.strip():
            results = chatbot_search(query, top_k=3)

    return render(request, "chatbot.html", {"results": results, "query": query})
