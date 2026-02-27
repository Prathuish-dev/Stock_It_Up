"""
tests/test_parser_stress.py
---------------------------
PHASE 5 — Parser robustness and case-insensitivity stress tests.

Tests verify that IntentParser is resilient to:
    - Mixed-case input (e.g. 'ToP 20 NsE bY CaGr')
    - Noisy natural-language phrasing with particles and filler words
    - Edge spacing / punctuation
    - Correct parameter extraction from impure inputs
"""

import unittest
from chatbot.intent_parser import IntentParser
from chatbot.enums import Intent, Exchange


class TestParserCaseInsensitivity(unittest.TestCase):
    """Parser must detect intent regardless of character casing."""

    def setUp(self):
        self.parser = IntentParser()

    def test_screen_top_all_caps(self):
        """'TOP 10 NSE BY CAGR' → SCREEN_TOP."""
        self.assertEqual(
            self.parser.parse_intent("TOP 10 NSE BY CAGR"),
            Intent.SCREEN_TOP,
        )

    def test_screen_top_mixed_case(self):
        """'ToP 20 NsE bY CaGr' → SCREEN_TOP."""
        self.assertEqual(
            self.parser.parse_intent("ToP 20 NsE bY CaGr"),
            Intent.SCREEN_TOP,
        )

    def test_screen_top_all_lower(self):
        """'top 5 nse by volatility' → SCREEN_TOP."""
        self.assertEqual(
            self.parser.parse_intent("top 5 nse by volatility"),
            Intent.SCREEN_TOP,
        )

    def test_screener_params_case_insensitive_limit(self):
        """Limit is extracted correctly regardless of case."""
        params = self.parser.extract_screener_params("ToP 20 NsE bY CaGr")
        self.assertEqual(params["limit"], 20)

    def test_screener_params_case_insensitive_exchange(self):
        """NSE/BSE detection works even from mixed-case text."""
        params = self.parser.extract_screener_params("ToP 20 NsE bY CaGr")
        self.assertEqual(params["exchange"], Exchange.NSE)

    def test_screener_params_case_insensitive_metric(self):
        """Metric alias 'cagr' detected from 'CaGr'."""
        params = self.parser.extract_screener_params("ToP 20 NsE bY CaGr")
        self.assertEqual(params["metric"], "cagr")

    def test_screener_params_metric_alias_growth_caps(self):
        """'GROWTH' alias resolves to 'cagr'."""
        params = self.parser.extract_screener_params("TOP 10 NSE BY GROWTH")
        self.assertEqual(params["metric"], "cagr")

    def test_screener_params_direction_caps(self):
        """'LOWEST' triggers ascending direction."""
        params = self.parser.extract_screener_params("LOWEST 10 BSE BY VOLATILITY")
        self.assertEqual(params["direction"], "asc")

    def test_help_case_insensitive(self):
        """'HELP' and 'Help' must both fire SHOW_KEYWORDS."""
        self.assertEqual(self.parser.parse_intent("HELP"), Intent.SHOW_KEYWORDS)
        self.assertEqual(self.parser.parse_intent("Help"), Intent.SHOW_KEYWORDS)

    def test_quit_case_insensitive(self):
        """'EXIT' and 'Quit' must both fire QUIT."""
        self.assertEqual(self.parser.parse_intent("EXIT"), Intent.QUIT)
        self.assertEqual(self.parser.parse_intent("Quit"), Intent.QUIT)


class TestParserNoisyInput(unittest.TestCase):
    """Parser must detect intent from conversational, noisy phrasing."""

    def setUp(self):
        self.parser = IntentParser()

    def test_noisy_top_cagr(self):
        """Filler words around a screener command must not prevent detection."""
        self.assertEqual(
            self.parser.parse_intent(
                "hey can you maybe top 10 nse by cagr please?"
            ),
            Intent.SCREEN_TOP,
        )

    def test_noisy_top_score(self):
        """Noisy prefix before 'best' must still resolve to SCREEN_TOP."""
        self.assertEqual(
            self.parser.parse_intent("i would like to see the best 5 BSE stocks"),
            Intent.SCREEN_TOP,
        )

    def test_noisy_lowest_volatility(self):
        """'safest' embedded in a sentence must trigger SCREEN_TOP."""
        self.assertEqual(
            self.parser.parse_intent(
                "give me the safest 20 nse stocks by volatility"
            ),
            Intent.SCREEN_TOP,
        )

    def test_noisy_params_limit_extracted(self):
        """Limit must be extracted from a noisy sentence."""
        params = self.parser.extract_screener_params(
            "hey can you maybe top 10 nse by cagr please?"
        )
        self.assertEqual(params["limit"], 10)

    def test_noisy_params_exchange_extracted(self):
        """Exchange must be extracted from noisy sentence."""
        params = self.parser.extract_screener_params(
            "hey can you maybe top 10 nse by cagr please?"
        )
        self.assertEqual(params["exchange"], Exchange.NSE)

    def test_noisy_params_metric_extracted(self):
        """Metric keyword must survive noisy context."""
        params = self.parser.extract_screener_params(
            "hey can you maybe top 10 nse by cagr please?"
        )
        self.assertEqual(params["metric"], "cagr")

    def test_noisy_no_false_positive_on_screener(self):
        """
        Pure filler text with no screener keyword must NOT produce SCREEN_TOP.
        (Regression: ensure we never fire SCREEN_TOP on irrelevant noise.)
        """
        intent = self.parser.parse_intent("hey there how are you doing today")
        self.assertNotEqual(intent, Intent.SCREEN_TOP)

    def test_screener_with_trailing_question_mark(self):
        """Trailing punctuation must not prevent screener detection."""
        self.assertEqual(
            self.parser.parse_intent("top 5 NSE by cagr?"),
            Intent.SCREEN_TOP,
        )

    def test_screener_with_extra_spaces(self):
        """Extra whitespace must not confuse the parser."""
        self.assertEqual(
            self.parser.parse_intent("  top   10   NSE   by   cagr  "),
            Intent.SCREEN_TOP,
        )


class TestParserEdgeBoundary(unittest.TestCase):
    """Boundary / edge conditions for the parser."""

    def setUp(self):
        self.parser = IntentParser()

    def test_empty_string_does_not_raise(self):
        """Completely empty string must not raise an exception."""
        try:
            intent = self.parser.parse_intent("")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"parse_intent('') raised {exc!r} unexpectedly")

    def test_whitespace_only_does_not_raise(self):
        """Whitespace-only input must not raise."""
        try:
            self.parser.parse_intent("   \t\n")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"parse_intent(whitespace) raised {exc!r} unexpectedly")

    def test_non_ascii_chars_do_not_crash(self):
        """Unicode / emoji input must not raise (may return UNKNOWN)."""
        try:
            self.parser.parse_intent("📈 top 10 NSE by cagr 🚀")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"parse_intent(unicode) raised {exc!r} unexpectedly")

    def test_large_limit_number_parsed(self):
        """A 3-digit limit (200) must be parsed correctly."""
        params = self.parser.extract_screener_params("top 200 NSE by cagr")
        self.assertEqual(params["limit"], 200)

    def test_exchange_bse_mixed_case(self):
        """'bSE' must resolve to Exchange.BSE."""
        ex = self.parser.extract_exchange("I want bSE stocks")
        self.assertEqual(ex, Exchange.BSE)


if __name__ == "__main__":
    unittest.main()
