(() => {
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatLog = document.getElementById("chat-log");
    const sendBtn = document.getElementById("chat-send-btn");
    const resetBtn = document.getElementById("chat-reset-btn");

    if (!chatForm || !chatInput || !chatLog || !sendBtn || !resetBtn) {
        return;
    }

    const sessionStorageKey = "stock_it_up_chat_session";
    let sessionId = localStorage.getItem(sessionStorageKey);
    if (!sessionId) {
        sessionId = `sess_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
        localStorage.setItem(sessionStorageKey, sessionId);
    }

    function addBubble(role, text) {
        const div = document.createElement("div");
        div.className = `chat-bubble ${role === "user" ? "chat-bubble-user" : "chat-bubble-bot"}`;
        div.textContent = text;
        chatLog.appendChild(div);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function callApi(path, payload) {
        const response = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json", "Accept": "application/json" },
            body: JSON.stringify(payload),
        });
        const body = await response.json();
        if (!response.ok) {
            throw new Error(body.error || "Request failed");
        }
        return body;
    }

    async function startChat() {
        chatLog.innerHTML = "";
        sendBtn.disabled = true;
        try {
            const body = await callApi("/api/chat/start", { session_id: sessionId });
            addBubble("bot", body.reply || "Chat started.");
        } catch (error) {
            addBubble("bot", "Failed to start chat session.");
        } finally {
            sendBtn.disabled = false;
            chatInput.focus();
        }
    }

    chatForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        addBubble("user", text);
        chatInput.value = "";
        sendBtn.disabled = true;

        try {
            const body = await callApi("/api/chat/message", {
                session_id: sessionId,
                message: text,
            });
            addBubble("bot", body.reply || "No response.");
        } catch (error) {
            addBubble("bot", "Unable to process message right now.");
        } finally {
            sendBtn.disabled = false;
            chatInput.focus();
        }
    });

    resetBtn.addEventListener("click", async () => {
        sendBtn.disabled = true;
        try {
            const body = await callApi("/api/chat/reset", { session_id: sessionId });
            chatLog.innerHTML = "";
            addBubble("bot", body.reply || "Chat reset.");
        } catch (error) {
            addBubble("bot", "Unable to reset chat.");
        } finally {
            sendBtn.disabled = false;
            chatInput.focus();
        }
    });

    startChat();
})();

