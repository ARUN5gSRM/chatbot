// Simple progressive enhancement for chat + typing indicator + autoscroll.
// Works with server-side POST fallback.
document.addEventListener("DOMContentLoaded", function() {
    const chatbox = document.getElementById("chatbox");
    const chatForm = document.querySelector("form[method='post']");
    const input = chatForm ? chatForm.querySelector("input[name='query'], textarea[name='query']") : null;
    const typingId = "typing-indicator";

    function scrollToBottom() {
        if (!chatbox) return;
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    function setTyping(on=true) {
        if (!chatbox) return;
        let el = document.getElementById(typingId);
        if (on) {
            if (!el) {
                el = document.createElement("div");
                el.id = typingId;
                el.className = "msg assistant typing";
                el.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
                chatbox.appendChild(el);
            }
            scrollToBottom();
        } else {
            if (el) el.remove();
        }
    }

    // initial autoscroll
    scrollToBottom();

    // Enhance form with AJAX JSON post if server supports it.
    if (chatForm && input) {
        chatForm.addEventListener("submit", function(e) {
            // fallback: allow normal POST if fetch/AJAX not desired by server
            if (!window.fetch) return;

            e.preventDefault();
            const text = (input.value || "").trim();
            if (!text) return;

            // append user bubble quickly
            const userMsg = document.createElement("div");
            userMsg.className = "msg user";
            userMsg.textContent = text;
            chatbox.appendChild(userMsg);
            scrollToBottom();
            input.value = "";

            setTyping(true);

            // send JSON to the current URL; backend should accept JSON and reply JSON
            fetch(window.location.href, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-Requested-With": "XMLHttpRequest",
                    "Accept": "application/json",
                    "X-CSRFToken": (document.cookie.match(/csrftoken=([^;]+)/) || [])[1] || ""
                },
                body: JSON.stringify({ query: text })
            }).then(async resp => {
                setTyping(false);
                if (!resp.ok) {
                    const txt = await resp.text();
                    const err = document.createElement("div");
                    err.className = "msg assistant";
                    err.textContent = "Error: " + (txt || resp.statusText);
                    chatbox.appendChild(err);
                    scrollToBottom();
                    return;
                }
                const data = await resp.json();
                const assistant = document.createElement("div");
                assistant.className = "msg assistant";
                assistant.textContent = data.message || data.semantic?.length ? (data.message || "Response") : "No response.";
                chatbox.appendChild(assistant);
                scrollToBottom();
            }).catch(err => {
                setTyping(false);
                const eEl = document.createElement("div");
                eEl.className = "msg assistant";
                eEl.textContent = "Network error: " + err.message;
                chatbox.appendChild(eEl);
                scrollToBottom();
            });
        });
    }

    // If upload form exists, keep the jQuery progress code; otherwise nothing to do here.
});
