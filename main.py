import sys
from chatbot.conversation_manager import ConversationManager


def main():
    manager = ConversationManager()

    print("Welcome to Stock It Up ChatBot")
    print("Type 'exit' to quit.\n")

    # Print the opening prompt without waiting for user input
    print("Bot:", manager.start())

    while True:
        user_input = input("\nYou: ")

        response = manager.handle_message(user_input)
        print("Bot:", response)

        if manager.context.is_complete():
            break


if __name__ == "__main__":
    main()