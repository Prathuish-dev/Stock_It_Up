"""
tests/test_screener.py
----------------------
Unit tests for Screener Mode.  All disk I/O is mocked — no CSVs are read.

Test coverage:
    Intent / parser layer  — intent detection, param extraction
    MetricsEngine          — compute_metric() selective dispatch
    ScreenerEngine         — metric mode, score mode, error skipping
    ConversationManager    — integration: full turn end-to-end
    ResponseGenerator      — format_screener_results() output shape
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

import pandas as pd
import numpy as np

from chatbot.enums import Intent, Exchange
from chatbot.intent_parser import IntentParser
from chatbot.metrics_engine import MetricsEngine
from chatbot.screener_engine import ScreenerEngine
from chatbot.response_generator import ResponseGenerator
from chatbot.conversation_manager import ConversationManager


# ---------------------------------------------------------------------------
# Shared synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_df(n_rows: int = 300, start_price: float = 1000.0, growth: float = 0.0002):
    """Return a synthetic price DataFrame with enough rows for MetricsEngine."""
    dates  = pd.date_range(end=datetime.today(), periods=n_rows, freq="B")
    prices = [start_price * ((1 + growth) ** i) for i in range(n_rows)]
    return pd.DataFrame({
        "Date":      dates,
        "Open":      prices,
        "High":      [p * 1.01 for p in prices],
        "Low":       [p * 0.99 for p in prices],
        "Close":     prices,
        "Adj Close": prices,
        "Volume":    [1_000_000] * n_rows,
    })


_DF_A = _make_df(600, start_price=2000.0, growth=0.00015)
_DF_B = _make_df(600, start_price=1500.0, growth=0.00012)
_DF_C = _make_df(600, start_price=1000.0, growth=0.00008)


# ===========================================================================
# 1. Intent + Parser
# ===========================================================================

class TestScreenerIntent(unittest.TestCase):

    def setUp(self):
        self.parser = IntentParser()

    # -- Intent detection --

    def test_screen_top_intent_basic(self):
        self.assertEqual(
            self.parser.parse_intent("top 10 NSE by cagr"),
            Intent.SCREEN_TOP,
        )

    def test_screen_top_intent_lowest(self):
        self.assertEqual(
            self.parser.parse_intent("lowest 5 BSE by volatility"),
            Intent.SCREEN_TOP,
        )

    def test_screen_top_intent_best(self):
        self.assertEqual(
            self.parser.parse_intent("best 20 NSE by score"),
            Intent.SCREEN_TOP,
        )

    def test_screen_top_does_not_conflict_with_list(self):
        """'list NSE' must not be confused with SCREEN_TOP."""
        self.assertEqual(
            self.parser.parse_intent("list NSE"),
            Intent.LIST_COMPANIES,
        )

    # -- Parameter extraction --

    def test_screener_params_limit(self):
        params = self.parser.extract_screener_params("top 5 NSE by cagr")
        self.assertEqual(params["limit"], 5)

    def test_screener_params_exchange_nse(self):
        params = self.parser.extract_screener_params("top 10 NSE by cagr")
        self.assertEqual(params["exchange"], Exchange.NSE)

    def test_screener_params_exchange_bse(self):
        params = self.parser.extract_screener_params("top 5 BSE by score")
        self.assertEqual(params["exchange"], Exchange.BSE)

    def test_screener_params_metric_cagr(self):
        params = self.parser.extract_screener_params("top 10 NSE by cagr")
        self.assertEqual(params["metric"], "cagr")

    def test_screener_params_metric_volatility(self):
        params = self.parser.extract_screener_params("top 10 NSE by volatility")
        self.assertEqual(params["metric"], "volatility")

    def test_screener_params_metric_score(self):
        params = self.parser.extract_screener_params("top 10 NSE by score")
        self.assertEqual(params["metric"], "score")

    def test_screener_params_metric_alias_safe(self):
        """'safe' should map to 'volatility' (lower = safer)."""
        params = self.parser.extract_screener_params("top 10 NSE safest stocks")
        self.assertEqual(params["metric"], "volatility")

    def test_screener_params_metric_alias_growth(self):
        params = self.parser.extract_screener_params("top 10 NSE by growth")
        self.assertEqual(params["metric"], "cagr")

    def test_screener_params_direction_desc_default(self):
        params = self.parser.extract_screener_params("top 10 NSE by cagr")
        self.assertEqual(params["direction"], "desc")

    def test_screener_params_direction_asc(self):
        params = self.parser.extract_screener_params("lowest 10 NSE by volatility")
        self.assertEqual(params["direction"], "asc")

    def test_screener_params_defaults(self):
        """No number or exchange → default limit=10, metric=cagr."""
        params = self.parser.extract_screener_params("top NSE")
        self.assertEqual(params["limit"], 10)
        self.assertEqual(params["metric"], "cagr")

    def test_screener_params_no_exchange(self):
        params = self.parser.extract_screener_params("top 10 by cagr")
        self.assertIsNone(params["exchange"])


# ===========================================================================
# 2. MetricsEngine.compute_metric()
# ===========================================================================

class TestComputeMetric(unittest.TestCase):

    def test_compute_metric_cagr_matches_compute_cagr(self):
        df = MetricsEngine.filter_by_horizon(_DF_A, 3)
        self.assertAlmostEqual(
            MetricsEngine.compute_metric(df, "cagr"),
            MetricsEngine.compute_cagr(df),
            places=10,
        )

    def test_compute_metric_volatility(self):
        df = MetricsEngine.filter_by_horizon(_DF_A, 3)
        self.assertAlmostEqual(
            MetricsEngine.compute_metric(df, "volatility"),
            MetricsEngine.compute_volatility(df),
            places=10,
        )

    def test_compute_metric_avg_volume(self):
        df = MetricsEngine.filter_by_horizon(_DF_A, 3)
        self.assertEqual(
            MetricsEngine.compute_metric(df, "avg_volume"),
            MetricsEngine.compute_avg_volume(df),
        )

    def test_compute_metric_unknown_raises(self):
        with self.assertRaises(ValueError):
            MetricsEngine.compute_metric(_DF_A, "banana_metric")


# ===========================================================================
# 3. ScreenerEngine
# ===========================================================================

class TestScreenerEngine(unittest.TestCase):

    def _mock_loader(self, tickers, dfs):
        """Return a MagicMock DataLoader wired with given tickers and DFs."""
        loader = MagicMock()
        loader.list_available.return_value = tickers
        df_map = dict(zip(tickers, dfs))
        loader.load_stock.side_effect = lambda ex, t: df_map[t]
        return loader

    def test_metric_mode_returns_sorted_desc(self):
        """Top by CAGR → highest first."""
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        self.assertEqual(len(results), 3)
        # Values should be descending
        values = [r["value"] for r in results]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_metric_mode_returns_sorted_asc(self):
        """Lowest by volatility → smallest value first."""
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="volatility", limit=3,
            horizon_years=3, direction="asc", data_loader=loader,
        )
        values = [r["value"] for r in results]
        self.assertEqual(values, sorted(values))

    def test_limit_respected(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=2,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        self.assertLessEqual(len(results), 2)

    def test_bad_ticker_skipped_silently(self):
        """A ticker that raises FileNotFoundError should be skipped."""
        loader = MagicMock()
        loader.list_available.return_value = ["GOOD", "BAD"]
        loader.load_stock.side_effect = lambda ex, t: (
            _DF_A if t == "GOOD" else (_ for _ in ()).throw(
                FileNotFoundError("no file")
            )
        )
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=5,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        tickers = [r["ticker"] for r in results]
        self.assertIn("GOOD", tickers)
        self.assertNotIn("BAD", tickers)

    def test_score_mode_returns_total_score_field(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="score", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        self.assertTrue(all("total_score" in r for r in results))

    def test_score_mode_without_risk_profile(self):
        """score mode with risk_profile=None must not raise."""
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="score", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
            risk_profile=None,
        )
        self.assertGreater(len(results), 0)

    def test_empty_exchange_returns_empty_list(self):
        loader = MagicMock()
        loader.list_available.return_value = []
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=10,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        self.assertEqual(results, [])

    def test_result_has_display_value(self):
        loader = self._mock_loader(["A"], [_DF_A])
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=1,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        self.assertIn("display_value", results[0])
        self.assertIn("%", results[0]["display_value"])


# ===========================================================================
# 4. ConversationManager integration
# ===========================================================================

@patch("chatbot.response_generator.DataLoader.load_stock")
class TestConversationManagerScreener(unittest.TestCase):

    def test_screen_top_no_exchange_prompts_user(self, mock_load):
        """Without an exchange in session or command, prompt the user."""
        m = ConversationManager()
        m.start()
        # Don't select exchange — state is COLLECT_EXCHANGE but no exchange set
        with patch("chatbot.screener_engine.ScreenerEngine.run") as mock_run:
            resp = m.handle_message("top 10 by cagr")
        self.assertIn("exchange", resp.lower())

    def test_screen_top_with_inline_exchange(self, mock_load):
        """Exchange embedded in command ('top 10 NSE by cagr') works without prior session."""
        m = ConversationManager()
        m.start()
        with patch.object(m._loader, "list_available", return_value=["A", "B", "C"]):
            with patch.object(m._loader, "load_stock", side_effect=lambda ex, t: _DF_A):
                resp = m.handle_message("top 3 NSE by cagr")
        # Should include the table header
        self.assertIn("CAGR", resp)

    def test_screen_top_session_exchange_fallback(self, mock_load):
        """When session exchange is set, screener uses it even without inline exchange."""
        m = ConversationManager()
        m.start()
        m.handle_message("NSE")      # sets session exchange
        with patch.object(m._loader, "list_available", return_value=["A"]):
            with patch.object(m._loader, "load_stock", return_value=_DF_A):
                resp = m.handle_message("top 1 by cagr")
        self.assertIn("CAGR", resp)


# ===========================================================================
# 5. ResponseGenerator.format_screener_results()
# ===========================================================================

class TestFormatScreenerResults(unittest.TestCase):

    def setUp(self):
        self.gen = ResponseGenerator.__new__(ResponseGenerator)
        # Avoid calling __init__ which would try to open files
        self.gen._loader = MagicMock()
        self.gen._engine = MagicMock()

    def _sample_results(self, n=3):
        return [
            {
                "ticker":        f"TICK{i}",
                "value":         0.15 - i * 0.01,
                "metric":        "cagr",
                "display_value": f"+{(15 - i):.2f}%",
                "display_name":  "CAGR",
            }
            for i in range(n)
        ]

    def test_output_contains_metric_name(self):
        out = self.gen.format_screener_results(
            self._sample_results(), metric="cagr",
            exchange="NSE", limit=3, horizon_years=3,
        )
        self.assertIn("CAGR", out)

    def test_output_contains_exchange(self):
        out = self.gen.format_screener_results(
            self._sample_results(), metric="cagr",
            exchange="NSE", limit=3, horizon_years=3,
        )
        self.assertIn("NSE", out)

    def test_output_has_ranked_rows(self):
        out = self.gen.format_screener_results(
            self._sample_results(3), metric="cagr",
            exchange="NSE", limit=3, horizon_years=3,
        )
        self.assertIn("TICK0", out)
        self.assertIn("TICK1", out)
        self.assertIn("TICK2", out)

    def test_empty_results_returns_graceful_message(self):
        out = self.gen.format_screener_results(
            [], metric="cagr", exchange="NSE", limit=10, horizon_years=3,
        )
        self.assertIn("No results", out)

    def test_lowest_direction_header(self):
        out = self.gen.format_screener_results(
            self._sample_results(), metric="volatility",
            exchange="BSE", limit=3, horizon_years=3, direction="asc",
        )
        self.assertIn("Lowest", out)

    def test_score_mode_footer_hint(self):
        results = [
            {
                "ticker": "A", "value": 0.9, "metric": "score",
                "display_value": "0.9000", "display_name": "Score",
                "total_score": 0.9, "component_scores": {}, "weights_used": {},
                "metrics": {},
            }
        ]
        out = self.gen.format_screener_results(
            results, metric="score", exchange="NSE", limit=1, horizon_years=3,
        )
        self.assertIn("cagr", out.lower())


if __name__ == "__main__":
    unittest.main()
