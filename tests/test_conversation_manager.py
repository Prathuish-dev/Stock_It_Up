import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from chatbot.conversation_manager import ConversationManager
from chatbot.enums import ConversationState, Exchange


# ---------------------------------------------------------------------------
# Helpers: synthetic DataFrame that satisfies MetricsEngine constraints
# ---------------------------------------------------------------------------

def _make_df(n_rows: int = 300, start_price: float = 1000.0, growth: float = 0.0001):
    """Return a synthetic price DataFrame with n_rows trading days."""
    dates = pd.date_range(end=datetime.today(), periods=n_rows, freq="B")
    prices = [start_price * ((1 + growth) ** i) for i in range(n_rows)]
    return pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": [p * 1.01 for p in prices],
        "Low":  [p * 0.99 for p in prices],
        "Close": prices,
        "Adj Close": prices,
        "Volume": [500_000] * n_rows,
    })


_SYNTHETIC_TCS  = _make_df(600, start_price=2000.0, growth=0.00015)
_SYNTHETIC_INFY = _make_df(600, start_price=1500.0, growth=0.00012)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

@patch("chatbot.response_generator.DataLoader.load_stock")
class TestConversationManager(unittest.TestCase):
    """
    All tests patch DataLoader.load_stock so no disk I/O occurs.
    The patch is applied at class level via decorator.
    """

    def _drive(self, mock_load, tickers=None):
        """Helper: walk through the full happy path and return the manager."""
        if tickers is None:
            tickers = ["TCS", "INFY"]

        def side_effect(exchange, ticker):
            return _SYNTHETIC_TCS if ticker == "TCS" else _SYNTHETIC_INFY

        mock_load.side_effect = side_effect

        m = ConversationManager()
        m.start()
        m.handle_message("NSE")
        m.handle_message("50000")
        m.handle_message("medium")
        m.handle_message("long")
        m.handle_message("yes")
        m.handle_message(" ".join(tickers))
        return m

    # -- Greeting / start --------------------------------------------------

    def test_start_returns_greeting(self, mock_load):
        m = ConversationManager()
        msg = m.start()
        self.assertIn("Stock It Up", msg)
        self.assertEqual(m.context.state, ConversationState.COLLECT_EXCHANGE)

    # -- Exchange step (now first) -----------------------------------------

    def test_exchange_nse_recognised(self, mock_load):
        m = ConversationManager()
        m.start()
        resp = m.handle_message("NSE")
        self.assertEqual(m.context.exchange, Exchange.NSE)
        self.assertEqual(m.context.state, ConversationState.COLLECT_BUDGET)
        self.assertIn("NSE", resp)

    def test_exchange_bse_recognised(self, mock_load):
        m = ConversationManager()
        m.start()
        resp = m.handle_message("BSE")
        self.assertEqual(m.context.exchange, Exchange.BSE)

    def test_invalid_exchange_reprompts(self, mock_load):
        m = ConversationManager()
        m.start()
        resp = m.handle_message("London")
        self.assertEqual(m.context.state, ConversationState.COLLECT_EXCHANGE)
        self.assertIn("NSE", resp)

    # -- Full happy path ---------------------------------------------------

    def test_full_happy_path(self, mock_load):
        m = self._drive(mock_load)
        self.assertIsNotNone(m.context.results)
        self.assertEqual(m.context.state, ConversationState.SHOW_RESULTS)
        self.assertEqual(m.context.exchange, Exchange.NSE)

    def test_results_table_contains_rank(self, mock_load):
        m = self._drive(mock_load)
        # Re-trigger analysis display by checking results directly
        self.assertTrue(len(m.context.results) >= 1)
        self.assertIn("total_score", m.context.results[0])

    # -- Quit / restart ---------------------------------------------------

    def test_quit_ends_session(self, mock_load):
        m = ConversationManager()
        m.start()
        m.handle_message("exit")
        self.assertTrue(m.context.is_complete())

    def test_restart_resets_state(self, mock_load):
        m = ConversationManager()
        m.start()
        m.handle_message("NSE")
        m.handle_message("restart")
        self.assertEqual(m.context.state, ConversationState.COLLECT_EXCHANGE)
        self.assertIsNone(m.context.exchange)
        self.assertIsNone(m.context.budget)

    # -- Input validation -------------------------------------------------

    def test_invalid_budget_reprompts(self, mock_load):
        m = ConversationManager()
        m.start()
        m.handle_message("NSE")
        resp = m.handle_message("banana budget")
        self.assertIn("valid budget", resp.lower())
        self.assertEqual(m.context.state, ConversationState.COLLECT_BUDGET)

    def test_invalid_risk_reprompts(self, mock_load):
        m = ConversationManager()
        m.start()
        m.handle_message("NSE")
        m.handle_message("50000")
        resp = m.handle_message("extreme turbo risk")
        self.assertIn("low", resp.lower())

    def test_invalid_horizon_reprompts(self, mock_load):
        m = ConversationManager()
        m.start()
        m.handle_message("NSE")
        m.handle_message("50000")
        m.handle_message("medium")
        resp = m.handle_message("forever")
        self.assertIn("short", resp.lower())

    # -- Explain ----------------------------------------------------------

    def test_explain_command(self, mock_load):
        m = self._drive(mock_load)
        resp = m.handle_message("explain TCS")
        self.assertIn("TCS", resp)
        self.assertIn("CAGR", resp)

    def test_explain_unknown_ticker(self, mock_load):
        m = self._drive(mock_load)
        resp = m.handle_message("explain ZZZUNKNOWN")
        self.assertIn("no result found", resp.lower())

    # -- Position sizing in results ---------------------------------------

    def test_results_contain_position_sizing(self, mock_load):
        m = ConversationManager()

        def side_effect(exchange, ticker):
            return _SYNTHETIC_TCS if ticker == "TCS" else _SYNTHETIC_INFY

        mock_load.side_effect = side_effect

        m.start()
        m.handle_message("NSE")
        m.handle_message("50000")   # ₹50,000 budget
        m.handle_message("medium")
        m.handle_message("long")
        m.handle_message("yes")
        resp = m.handle_message("TCS INFY")
        # Budget line should mention "shares"
        self.assertIn("shares", resp)


if __name__ == "__main__":
    unittest.main()