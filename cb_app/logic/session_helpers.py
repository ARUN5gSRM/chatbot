# cb_app/sub_models/session_helpers.py
"""
Session helpers for chat conversation stored in Django sessions.
"""
from typing import List
CONVERSATION_HISTORY_KEY = "chat_history"
MAX_CONVERSATION_MESSAGES = 8
STATE_KEY = "chat_state"

def init_session_history_if_needed(request):
    if CONVERSATION_HISTORY_KEY not in request.session:
        request.session[CONVERSATION_HISTORY_KEY] = []
    if STATE_KEY not in request.session:
        request.session[STATE_KEY] = "IDLE"
    request.session.modified = True

def append_user_message(request, text: str):
    init_session_history_if_needed(request)
    history = request.session[CONVERSATION_HISTORY_KEY]
    history.append({"role": "user", "content": text})
    request.session[CONVERSATION_HISTORY_KEY] = history[-MAX_CONVERSATION_MESSAGES:]
    request.session["last_query"] = text
    request.session.modified = True

def append_assistant_message(request, text: str):
    init_session_history_if_needed(request)
    history = request.session[CONVERSATION_HISTORY_KEY]
    history.append({"role": "assistant", "content": text})
    request.session[CONVERSATION_HISTORY_KEY] = history[-MAX_CONVERSATION_MESSAGES:]
    request.session["last_bot_response"] = text
    request.session.modified = True

def get_recent_conversation(request):
    init_session_history_if_needed(request)
    return request.session[CONVERSATION_HISTORY_KEY][-MAX_CONVERSATION_MESSAGES:]

def clear_conversation(request):
    if CONVERSATION_HISTORY_KEY in request.session:
        del request.session[CONVERSATION_HISTORY_KEY]
    request.session[STATE_KEY] = "IDLE"
    request.session.modified = True
    return {"status": "cleared"}
