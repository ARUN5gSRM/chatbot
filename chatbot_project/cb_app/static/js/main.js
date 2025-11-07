
// Auto-scroll chatbox to bottom
document.addEventListener("DOMContentLoaded", function() {
    const chatbox = document.getElementById("chatbox");
    if (chatbox) {
        chatbox.scrollTop = chatbox.scrollHeight;
    }
});