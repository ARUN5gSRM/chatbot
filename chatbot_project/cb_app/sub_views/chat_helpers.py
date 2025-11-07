# cb_app/sub_views/chat_helpers.py
CONVERSATION_HISTORY_KEY = "chat_history"
MAX_CONVERSATION_MESSAGES = 8


def init_session_history_if_needed(request):
    """Ensure session has a conversation history list."""
    if CONVERSATION_HISTORY_KEY not in request.session:
        request.session[CONVERSATION_HISTORY_KEY] = []
        request.session.modified = True


def append_user_message(request, text: str):
    """Add user message to session conversation."""
    init_session_history_if_needed(request)
    history = request.session[CONVERSATION_HISTORY_KEY]
    history.append({"role": "user", "content": text})
    request.session[CONVERSATION_HISTORY_KEY] = history[-MAX_CONVERSATION_MESSAGES:]
    request.session.modified = True


def append_assistant_message(request, text: str):
    """Add assistant message to session conversation."""
    init_session_history_if_needed(request)
    history = request.session[CONVERSATION_HISTORY_KEY]
    history.append({"role": "assistant", "content": text})
    request.session[CONVERSATION_HISTORY_KEY] = history[-MAX_CONVERSATION_MESSAGES:]
    request.session.modified = True


def get_recent_conversation(request):
    """Return recent conversation history."""
    init_session_history_if_needed(request)
    return request.session[CONVERSATION_HISTORY_KEY][-MAX_CONVERSATION_MESSAGES:]


def clear_conversation(request):
    """Clear chat session."""
    if CONVERSATION_HISTORY_KEY in request.session:
        del request.session[CONVERSATION_HISTORY_KEY]
        request.session.modified = True
