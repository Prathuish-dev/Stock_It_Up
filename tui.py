import sys
import contextlib
from io import StringIO
from textual.app import App, ComposeResult
from textual.widgets import Header, RichLog, Input, Footer
from textual.containers import Container
from textual import work

from chatbot.conversation_manager import ConversationManager


class StockApp(App):
    """A Textual app wrapper for Stock It Up ChatBot."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }
    #chat-container {
        height: 1fr;
        border: round $primary;
        margin: 1 2;
        background: $panel;
    }
    RichLog {
        height: 1fr;
        padding: 1 2;
        background: $panel;
        color: $text;
    }
    Input {
        dock: bottom;
        margin: 1 2;
        border: tall $secondary;
        background: $surface;
    }
    Input:focus {
        border: tall $primary;
    }
    """

    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.manager = ConversationManager()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        with Container(id="chat-container"):
            yield RichLog(id="chat_log", highlight=True, markup=True)
        yield Input(placeholder="Type your message here...", id="user_input")
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        self.title = "Stock It Up ChatBot"
        self.sub_title = "The TUI dashboard for stock screening"
        self.theme = "textual-dark"

        chat_log = self.query_one(RichLog)
        
        # Display the opening prompt
        opening = self.manager.start()
        chat_log.write("[bold green]Welcome to Stock It Up ChatBot[/]")
        chat_log.write("Type 'exit' to quit.\n")
        chat_log.write(f"[bold cyan]Bot:[/] {opening}")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Called when the user presses Enter in the Input widget."""
        user_input = event.value.strip()
        if not user_input:
            return

        # Clear the input
        input_widget = self.query_one(Input)
        input_widget.value = ""

        # Write user's message to log
        chat_log = self.query_one(RichLog)
        chat_log.write(f"\n[bold magenta]You:[/] {user_input}")

        # Check for quit early to avoid heavy processing
        if user_input.lower() == "exit":
            self.exit()
            return

        # Disable input while waiting
        input_widget.disabled = True

        # Run chatbot logic asynchronously
        self.handle_chatbot_query(user_input)

    @work(thread=True)
    def handle_chatbot_query(self, user_input: str) -> None:
        """Run the chatbot turn in a background thread."""
        output_buffer = StringIO()
        
        # Capture stdout so prints from cache/screening appear in UI
        with contextlib.redirect_stdout(output_buffer):
            response = self.manager.handle_message(user_input)

        # Get collected standard prints (if any)
        system_output = output_buffer.getvalue().strip()

        # Update the UI on the main thread
        self.call_from_thread(self._update_log, system_output, response)

    def _update_log(self, system_output: str, response: str) -> None:
        """Write the results back to the RichLog."""
        chat_log = self.query_one(RichLog)
        
        if system_output:
            chat_log.write(f"[yellow]{system_output}[/]")
            chat_log.write("") # spacer
            
        chat_log.write(f"[bold cyan]Bot:[/] {response}")

        # Re-enable the input
        input_widget = self.query_one(Input)
        input_widget.disabled = False
        input_widget.focus()

        # Exit app if the conversation state is done (quit intent)
        if self.manager.context.is_complete():
            self.exit()

if __name__ == "__main__":
    app = StockApp()
    app.run()
