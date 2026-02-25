from chatbot.conversation_manager import ConversationManager


class Chatbot:
    """
    Thin facade around :class:`ConversationManager` for use in tests
    or alternative front-ends.
    """

    def __init__(self):
        self._manager = ConversationManager()

    def start(self) -> str:
        """Return the opening message."""
        return self._manager.start()

    def send(self, message: str) -> str:
        """Send a user message and get a reply."""
        return self._manager.handle_message(message)

    @property
    def is_done(self) -> bool:
        return self._manager.context.is_complete()