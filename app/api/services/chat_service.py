from __future__ import annotations

import logging
import threading
import time

from chatbot.conversation_manager import ConversationManager

logger = logging.getLogger("stock_it_up.chat")


class ChatService:
    def __init__(self) -> None:
        self._sessions: dict[str, ConversationManager] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._max_sessions = 200

    def _touch(self, session_id: str) -> None:
        self._last_seen[session_id] = time.time()

    def _evict_if_needed(self) -> None:
        if len(self._sessions) < self._max_sessions:
            return
        oldest_session = min(self._last_seen.items(), key=lambda item: item[1])[0]
        self._sessions.pop(oldest_session, None)
        self._last_seen.pop(oldest_session, None)
        logger.info("chat.session_evicted session_id=%s", oldest_session)

    def start(self, session_id: str) -> dict:
        with self._lock:
            manager = ConversationManager()
            opening = manager.start()
            self._evict_if_needed()
            self._sessions[session_id] = manager
            self._touch(session_id)
            logger.info("chat.start session_id=%s", session_id)
            return {
                "session_id": session_id,
                "state": manager.context.state.name,
                "reply": opening,
                "done": manager.context.is_complete(),
            }

    def message(self, session_id: str, text: str) -> dict:
        with self._lock:
            manager = self._sessions.get(session_id)
            if manager is None:
                manager = ConversationManager()
                opening = manager.start()
                self._evict_if_needed()
                self._sessions[session_id] = manager
                logger.info("chat.implicit_start session_id=%s", session_id)
                reply = f"{opening}\n\n{manager.handle_message(text)}"
            else:
                reply = manager.handle_message(text)

            self._touch(session_id)
            logger.info(
                "chat.message session_id=%s state=%s done=%s text_len=%s",
                session_id,
                manager.context.state.name,
                manager.context.is_complete(),
                len(text),
            )
            return {
                "session_id": session_id,
                "state": manager.context.state.name,
                "reply": reply,
                "done": manager.context.is_complete(),
            }

    def reset(self, session_id: str) -> dict:
        return self.start(session_id)


chat_service = ChatService()

