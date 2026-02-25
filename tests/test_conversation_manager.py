import unittest
from chatbot.conversation_manager import ConversationManager
from chatbot.enums import ConversationState


class TestConversationManager(unittest.TestCase):

    def setUp(self):
        self.manager = ConversationManager()

    def test_start_returns_greeting(self):
        msg = self.manager.start()
        self.assertIn("Stock It Up", msg)

    def test_full_happy_path(self):
        self.manager.start()
        self.manager.handle_message("50000")           # budget
        self.manager.handle_message("medium")          # risk
        self.manager.handle_message("long")            # horizon
        self.manager.handle_message("yes")             # accept default weights
        response = self.manager.handle_message("TCS INFY")  # stocks
        self.assertIn("RANK", response)
        self.assertEqual(self.manager.context.state, ConversationState.SHOW_RESULTS)

    def test_quit_ends_session(self):
        self.manager.start()
        self.manager.handle_message("exit")
        self.assertTrue(self.manager.context.is_complete())

    def test_restart_resets_state(self):
        self.manager.start()
        self.manager.handle_message("50000")
        self.manager.handle_message("restart")
        self.assertEqual(self.manager.context.state, ConversationState.COLLECT_BUDGET)
        self.assertIsNone(self.manager.context.budget)

    def test_invalid_budget_reprompts(self):
        self.manager.start()
        response = self.manager.handle_message("abc")
        self.assertIn("valid budget", response.lower())

    def test_invalid_risk_reprompts(self):
        self.manager.start()
        self.manager.handle_message("50000")
        response = self.manager.handle_message("banana")
        self.assertIn("low", response.lower())

    def test_explain_command(self):
        self.manager.start()
        self.manager.handle_message("50000")
        self.manager.handle_message("medium")
        self.manager.handle_message("long")
        self.manager.handle_message("yes")
        self.manager.handle_message("TCS INFY")
        response = self.manager.handle_message("explain TCS")
        self.assertIn("TCS", response)


if __name__ == "__main__":
    unittest.main()