from __future__ import annotations

from chatbot.conversation_manager import ConversationManager

try:
    from app.main import app
except Exception:  # pragma: no cover - web stack may be unavailable in CLI-only env
    app = None


def run_cli() -> None:
    manager = ConversationManager()

    print("Welcome to Stock It Up ChatBot")
    print("Type 'exit' to quit.\n")
    print("Bot:", manager.start())

    while True:
        user_input = input("\nYou: ")
        response = manager.handle_message(user_input)
        print("Bot:", response)

        if manager.context.is_complete():
            break


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

