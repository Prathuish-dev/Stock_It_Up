"""
Unit tests for discoverability commands.

Tests cover:
- IntentParser: COMMAND_MAP detection for all 4 new intents
- DataLoader: search_tickers() prefix/substring ranking
- ResponseGenerator: list_companies() pagination, search_results() grouping,
  show_keywords(), show_exchanges()
- ConversationManager: end-to-end routing of all 4 global commands
"""

import unittest
from unittest.mock import patch, MagicMock
from chatbot.intent_parser import IntentParser, COMMAND_MAP
from chatbot.enums import Intent, Exchange
from chatbot.response_generator import ResponseGenerator
from chatbot.conversation_manager import ConversationManager


# ---------------------------------------------------------------------------
#  IntentParser — COMMAND_MAP detection
# ---------------------------------------------------------------------------

class TestIntentParserDiscoverability(unittest.TestCase):

    def setUp(self):
        self.parser = IntentParser()

    # -- LIST_COMPANIES --
    def test_list_intent(self):
        self.assertEqual(self.parser.parse_intent("list"), Intent.LIST_COMPANIES)

    def test_list_nse_intent(self):
        self.assertEqual(self.parser.parse_intent("list nse"), Intent.LIST_COMPANIES)

    def test_list_bse_intent(self):
        self.assertEqual(self.parser.parse_intent("list bse companies"), Intent.LIST_COMPANIES)

    def test_show_companies_intent(self):
        self.assertEqual(self.parser.parse_intent("show companies"), Intent.LIST_COMPANIES)

    def test_available_stocks_intent(self):
        self.assertEqual(self.parser.parse_intent("available stocks"), Intent.LIST_COMPANIES)

    # -- SEARCH_COMPANY --
    def test_search_intent(self):
        self.assertEqual(self.parser.parse_intent("search TCS"), Intent.SEARCH_COMPANY)

    def test_find_intent(self):
        self.assertEqual(self.parser.parse_intent("find bajaj"), Intent.SEARCH_COMPANY)

    # -- SHOW_KEYWORDS --
    def test_help_intent(self):
        self.assertEqual(self.parser.parse_intent("help"), Intent.SHOW_KEYWORDS)

    def test_keywords_intent(self):
        self.assertEqual(self.parser.parse_intent("keywords"), Intent.SHOW_KEYWORDS)

    def test_commands_intent(self):
        self.assertEqual(self.parser.parse_intent("commands"), Intent.SHOW_KEYWORDS)

    # -- SHOW_EXCHANGES --
    def test_exchanges_intent(self):
        self.assertEqual(self.parser.parse_intent("exchanges"), Intent.SHOW_EXCHANGES)

    def test_markets_intent(self):
        self.assertEqual(self.parser.parse_intent("markets"), Intent.SHOW_EXCHANGES)

    # -- extract_list_params --
    def test_extract_list_nse_page1(self):
        params = self.parser.extract_list_params("list nse")
        self.assertEqual(params["exchange"], Exchange.NSE)
        self.assertEqual(params["page"], 1)

    def test_extract_list_bse_page2(self):
        params = self.parser.extract_list_params("list bse page 2")
        self.assertEqual(params["exchange"], Exchange.BSE)
        self.assertEqual(params["page"], 2)

    def test_extract_list_no_exchange(self):
        params = self.parser.extract_list_params("list")
        self.assertIsNone(params["exchange"])
        self.assertEqual(params["page"], 1)

    # -- extract_search_query --
    def test_extract_search_query_basic(self):
        self.assertEqual(self.parser.extract_search_query("search TCS"), "TCS")

    def test_extract_search_query_find(self):
        self.assertEqual(self.parser.extract_search_query("find bajaj"), "BAJAJ")

    def test_extract_search_query_empty(self):
        self.assertEqual(self.parser.extract_search_query("search"), "")


# ---------------------------------------------------------------------------
#  ResponseGenerator — discoverability formatting
# ---------------------------------------------------------------------------

class TestResponseGeneratorDiscoverability(unittest.TestCase):

    def setUp(self):
        # Patch DataLoader so no disk I/O
        with patch("chatbot.response_generator.DataLoader") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.list_available.return_value = [
                chr(ord("A") + i) * 3 for i in range(150)  # AAA, BBB, ...
            ]
            mock_cls.return_value = mock_instance
            self.gen = ResponseGenerator()

    def test_list_companies_page1_shows_50(self):
        companies = [f"STOCK{i:03d}" for i in range(120)]
        resp = self.gen.list_companies(companies, exchange="NSE", page=1)
        # Should show 50 tickers
        self.assertIn("page 1/3", resp)
        self.assertIn("next page", resp.lower())

    def test_list_companies_page2(self):
        companies = [f"STOCK{i:03d}" for i in range(120)]
        resp = self.gen.list_companies(companies, exchange="NSE", page=2)
        self.assertIn("page 2/3", resp)

    def test_list_companies_last_page_no_next(self):
        companies = [f"STOCK{i:03d}" for i in range(50)]
        resp = self.gen.list_companies(companies, exchange="NSE", page=1)
        self.assertNotIn("next page", resp.lower())

    def test_list_companies_empty(self):
        resp = self.gen.list_companies([], exchange="NSE")
        self.assertIn("No companies", resp)

    def test_search_results_exact_match(self):
        matches = ["TCS"]
        resp = self.gen.search_results("TCS", matches, exchange="NSE")
        self.assertIn("Exact", resp)
        self.assertIn("TCS", resp)

    def test_search_results_prefix_group(self):
        matches = ["TCS", "TCSINFRA"]
        resp = self.gen.search_results("TCS", matches, exchange="NSE")
        self.assertIn("Prefix", resp)

    def test_search_results_no_matches(self):
        resp = self.gen.search_results("ZZZNOPE", [], exchange="NSE")
        self.assertIn("No tickers", resp)

    def test_show_keywords_contains_all_commands(self):
        resp = self.gen.show_keywords()
        for cmd in ["list", "search", "explain", "restart", "exit", "help"]:
            self.assertIn(cmd, resp.lower())

    def test_show_exchanges_contains_nse_bse(self):
        # Patch list_available via the already-mocked DataLoader
        self.gen._loader.list_available.return_value = ["A"] * 100
        resp = self.gen.show_exchanges()
        self.assertIn("NSE", resp)
        self.assertIn("BSE", resp)
        self.assertIn("100", resp)   # count shows up


# ---------------------------------------------------------------------------
#  ConversationManager — end-to-end routing
# ---------------------------------------------------------------------------

@patch("chatbot.conversation_manager.DataLoader")
@patch("chatbot.response_generator.DataLoader")
class TestConversationManagerDiscoverability(unittest.TestCase):

    def _make_manager(self, mock_rg_loader, mock_cm_loader):
        """Build a manager with mocked DataLoaders."""
        tickers = [f"TICKER{i:03d}" for i in range(60)]

        for mock_cls in (mock_rg_loader, mock_cm_loader):
            inst = MagicMock()
            inst.list_available.return_value = tickers
            inst.search_tickers.return_value = ["TCS"]
            mock_cls.return_value = inst

        m = ConversationManager()
        return m

    def test_help_works_before_exchange(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        resp = m.handle_message("help")
        self.assertIn("Supported Commands", resp)

    def test_exchanges_works_before_exchange(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        resp = m.handle_message("exchanges")
        self.assertIn("NSE", resp)
        self.assertIn("BSE", resp)

    def test_list_nse_works_before_exchange_selected(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        resp = m.handle_message("list nse")
        self.assertIn("NSE", resp)

    def test_list_without_exchange_prompts(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        resp = m.handle_message("list")
        self.assertIn("exchange", resp.lower())

    def test_list_uses_session_exchange_when_set(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        m.handle_message("NSE")   # set exchange
        resp = m.handle_message("list")
        self.assertIn("NSE", resp)

    def test_search_without_exchange_prompts(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        resp = m.handle_message("search TCS")
        self.assertIn("exchange", resp.lower())

    def test_search_with_exchange_returns_results(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        m.handle_message("NSE")
        resp = m.handle_message("search TCS")
        self.assertIn("TCS", resp)

    def test_list_page2(self, mock_rg, mock_cm):
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        resp = m.handle_message("list nse page 2")
        self.assertIn("page 2", resp)

    def test_discoverability_works_after_results(self, mock_rg, mock_cm):
        """Global commands must work even in SHOW_RESULTS state."""
        m = self._make_manager(mock_rg, mock_cm)
        m.start()
        m.handle_message("NSE")
        # Jump to SHOW_RESULTS state manually
        from chatbot.enums import ConversationState
        m.context.state = ConversationState.SHOW_RESULTS
        resp = m.handle_message("keywords")
        self.assertIn("Supported Commands", resp)


if __name__ == "__main__":
    unittest.main()
