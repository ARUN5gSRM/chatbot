# cb_app/sub_views/chatbot_view.py
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from ..sub_models.chatbot import chatbot_search, get_recent_conversation
from ..sub_models.ollama_helper import OllamaEmbeddingModel


@login_required
def chatbot_view(request):
    """Main chatbot page — handles query submission and displays results."""
    query = ""
    results = {}

    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if query:
            embedder = OllamaEmbeddingModel()
            results = chatbot_search(request, query, embedder)
    else:
        results = {"chat_history": get_recent_conversation(request)}

    return render(request, "cb_app/chatbot.html", {"results": results, "query": query})


@login_required
def clear_chat_view(request):
    """Clear chat session."""
    if "chat_history" in request.session:
        del request.session["chat_history"]
    return redirect("cb_app:chatbot")
