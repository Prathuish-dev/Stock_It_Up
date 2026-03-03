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
        """'best' routes to SCREEN_POSITION (single-result positional mode).
        Design: 'top N' → SCREEN_TOP list; 'best' → SCREEN_POSITION single card.
        """
        self.assertEqual(
            self.parser.parse_intent("best 20 NSE by score"),
            Intent.SCREEN_POSITION,
        )

    def test_screen_top_intent_top_keyword(self):
        """'top N ...' must still route to SCREEN_TOP (list mode)."""
        self.assertEqual(
            self.parser.parse_intent("top 20 NSE by score"),
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

    def test_screener_sharpe_alias_in_parser(self):
        """'top 10 NSE by sharpe' must extract metric='sharpe'."""
        params = self.parser.extract_screener_params("top 10 NSE by sharpe")
        self.assertEqual(params["metric"], "sharpe")

    def test_screener_risk_adjusted_alias_in_parser(self):
        """'top 10 NSE by risk-adjusted' must also map to metric='sharpe'."""
        params = self.parser.extract_screener_params("top 10 NSE by risk-adjusted return")
        self.assertEqual(params["metric"], "sharpe")

    def test_screener_risk_alias_maps_to_volatility(self):
        """'top 10 NSE by risk' must map to metric='volatility'."""
        params = self.parser.extract_screener_params("top 10 NSE by risk")
        self.assertEqual(params["metric"], "volatility")

    def test_screener_risky_alias_maps_to_volatility(self):
        """'top 10 BSE by risky' must map to metric='volatility'."""
        params = self.parser.extract_screener_params("top 10 BSE by risky")
        self.assertEqual(params["metric"], "volatility")

    def test_screener_volume_alias_maps_to_avg_volume(self):
        """'top 10 NSE by volume' must map to metric='avg_volume'."""
        params = self.parser.extract_screener_params("top 10 NSE by volume")
        self.assertEqual(params["metric"], "avg_volume")

    def test_screener_price_alias_maps_to_latest_price(self):
        """'top 10 NSE by price' must map to metric='latest_price'."""
        params = self.parser.extract_screener_params("top 10 NSE by price")
        self.assertEqual(params["metric"], "latest_price")

    def test_screener_drawdown_alias_maps_to_max_drawdown(self):
        """'top 10 NSE by drawdown' must map to metric='max_drawdown'."""
        params = self.parser.extract_screener_params("top 10 NSE by drawdown")
        self.assertEqual(params["metric"], "max_drawdown")


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

    def test_screener_large_limit(self):
        """
        200-ticker universe, limit=150: must return ≤150 results,
        no crash, and heap-based trimming must work correctly.
        """
        # Build 200 tickers each backed by a valid DataFrame
        tickers = [f"T{i:03d}" for i in range(200)]
        loader = MagicMock()
        loader.list_available.return_value = tickers
        # All tickers share the same DataFrame — fast to create
        loader.load_stock.side_effect = lambda ex, t: _DF_A

        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=150,
            horizon_years=3, direction="desc", data_loader=loader,
        )

        # Core invariants
        self.assertLessEqual(len(results), 150)
        self.assertGreater(len(results), 0)
        # Result must contain the required fields
        self.assertIn("ticker", results[0])
        self.assertIn("display_value", results[0])

    def test_screener_sharpe_metric_sorted_desc(self):
        """
        Phase 2 / Sharpe — screener with metric='sharpe' must:
        - Not crash
        - Return results sorted descending (highest Sharpe first)
        - Include 'display_value' in each result
        """
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="sharpe", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        self.assertEqual(len(results), 3)
        values = [r["value"] for r in results]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertIn("display_value", results[0])

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
@patch("chatbot.conversation_manager.MetricCache")
class TestConversationManagerScreener(unittest.TestCase):
    """
    Integration tests for screener commands through ConversationManager.

    MetricCache is stubbed at class level so no test ever touches the real
    comp_stock_data/.cache directory.  The mock cache always reports a miss
    (is_valid → False) and get_or_build returns {} so ScreenerEngine.run()
    falls through to the per-test _loader patches.
    """

    def _setup_cache_mock(self, mock_cache_cls):
        """Configure the MetricCache mock instance for all tests."""
        mc = mock_cache_cls.return_value
        mc.is_valid.return_value      = False  # always a miss → no stale output
        mc.get_or_build.return_value  = {}     # triggers live scan path in run()
        mc.invalidate.return_value    = None
        mc.build.return_value         = {}
        return mc

    def test_screen_top_no_exchange_prompts_user(self, mock_cache_cls, mock_load):
        """Without an exchange in session or command, prompt the user."""
        self._setup_cache_mock(mock_cache_cls)
        m = ConversationManager()
        m.start()
        # Don't select exchange — state is COLLECT_EXCHANGE but no exchange set
        with patch("chatbot.screener_engine.ScreenerEngine.run") as mock_run:
            resp = m.handle_message("top 10 by cagr")
        self.assertIn("exchange", resp.lower())

    def test_screen_top_with_inline_exchange(self, mock_cache_cls, mock_load):
        """Exchange embedded in command ('top 10 NSE by cagr') works without prior session."""
        self._setup_cache_mock(mock_cache_cls)
        m = ConversationManager()
        m.start()
        with patch.object(m._loader, "list_available", return_value=["A", "B", "C"]):
            with patch.object(m._loader, "load_stock", side_effect=lambda ex, t: _DF_A):
                resp = m.handle_message("top 3 NSE by cagr")
        # Should include the table header
        self.assertIn("CAGR", resp)

    def test_screen_top_session_exchange_fallback(self, mock_cache_cls, mock_load):
        """When session exchange is set, screener uses it even without inline exchange."""
        self._setup_cache_mock(mock_cache_cls)
        m = ConversationManager()
        m.start()
        m.handle_message("NSE")      # sets session exchange
        with patch.object(m._loader, "list_available", return_value=["A"]):
            with patch.object(m._loader, "load_stock", return_value=_DF_A):
                resp = m.handle_message("top 1 by cagr")
        self.assertIn("CAGR", resp)

    def test_screen_without_exchange(self, mock_cache_cls, mock_load):
        """
        Phase 2 — FSM must NOT crash when no exchange is available;
        it must instead prompt the user to supply one.
        """
        self._setup_cache_mock(mock_cache_cls)
        m = ConversationManager()
        m.start()
        # Do NOT provide an exchange — neither inline nor via session
        resp = m.handle_message("top 10 by cagr")
        # FSM must ask for the exchange, never raise an exception
        self.assertIn("exchange", resp.lower())


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
                "rank":          i + 1,
                "rank_label":    ["1st", "2nd", "3rd"][i],
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
                "rank": 1, "rank_label": "1st",
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

    def test_rank_label_in_output(self):
        """format_screener_results must render ordinal rank labels."""
        out = self.gen.format_screener_results(
            self._sample_results(3), metric="cagr",
            exchange="NSE", limit=3, horizon_years=3,
        )
        self.assertIn("1st", out)
        self.assertIn("2nd", out)
        self.assertIn("3rd", out)

    def test_best_pick_callout_in_desc(self):
        """Best-pick line must appear for direction='desc'."""
        out = self.gen.format_screener_results(
            self._sample_results(3), metric="cagr",
            exchange="NSE", limit=3, horizon_years=3, direction="desc",
        )
        self.assertIn("Best pick", out)
        self.assertIn("TICK0", out)   # first result = best

    def test_no_best_pick_callout_in_asc(self):
        """Best-pick callout must NOT appear for direction='asc' (lowest mode)."""
        out = self.gen.format_screener_results(
            self._sample_results(3), metric="volatility",
            exchange="NSE", limit=3, horizon_years=3, direction="asc",
        )
        self.assertNotIn("Best pick", out)


# ===========================================================================
# 6. ScreenerEngine rank fields
# ===========================================================================

class TestScreenerEngineRankFields(unittest.TestCase):
    """Verify that rank / rank_label are correctly attached to each result."""

    def _mock_loader(self, tickers, dfs):
        loader = MagicMock()
        loader.list_available.return_value = tickers
        df_map = dict(zip(tickers, dfs))
        loader.load_stock.side_effect = lambda ex, t: df_map[t]
        return loader

    def test_metric_mode_rank_field_present(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        for r in results:
            self.assertIn("rank", r)
            self.assertIn("rank_label", r)

    def test_metric_mode_rank_starts_at_one(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["rank_label"], "1st")

    def test_metric_mode_rank_sequential(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        ranks = [r["rank"] for r in results]
        self.assertEqual(ranks, list(range(1, len(results) + 1)))

    def test_score_mode_rank_field_present(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="score", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        for r in results:
            self.assertIn("rank", r)
            self.assertIn("rank_label", r)

    def test_score_mode_rank_labels_correct(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        results = ScreenerEngine.run(
            exchange="NSE", metric="score", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        expected_labels = ["1st", "2nd", "3rd"]
        for r, expected in zip(results, expected_labels):
            self.assertEqual(r["rank_label"], expected)

    def test_ordinal_11th_12th_13th_special_case(self):
        """11, 12, 13 must use 'th', not 'st'/'nd'/'rd'."""
        from chatbot.screener_engine import _ordinal
        self.assertEqual(_ordinal(11), "11th")
        self.assertEqual(_ordinal(12), "12th")
        self.assertEqual(_ordinal(13), "13th")
        self.assertEqual(_ordinal(21), "21st")
        self.assertEqual(_ordinal(22), "22nd")
        self.assertEqual(_ordinal(23), "23rd")
        self.assertEqual(_ordinal(111), "111th")


# ===========================================================================
# 7. Positional Queries  (SCREEN_POSITION intent + fetch_position)
# ===========================================================================

class TestPositionalQueries(unittest.TestCase):

    def setUp(self):
        self.parser = IntentParser()
        self.gen = ResponseGenerator.__new__(ResponseGenerator)
        self.gen._loader = MagicMock()
        self.gen._engine = MagicMock()

    def _mock_loader(self, tickers, dfs):
        loader = MagicMock()
        loader.list_available.return_value = tickers
        df_map = dict(zip(tickers, dfs))
        loader.load_stock.side_effect = lambda ex, t: df_map[t]
        return loader

    # --- Intent detection ---

    def test_worst_intent_is_screen_position(self):
        self.assertEqual(self.parser.parse_intent("worst NSE by cagr"), Intent.SCREEN_POSITION)

    def test_best_intent_is_screen_position(self):
        self.assertEqual(self.parser.parse_intent("best NSE by cagr"), Intent.SCREEN_POSITION)

    def test_last_intent_is_screen_position(self):
        self.assertEqual(self.parser.parse_intent("last BSE by score"), Intent.SCREEN_POSITION)

    def test_2nd_best_intent_is_screen_position(self):
        self.assertEqual(self.parser.parse_intent("2nd best NSE by cagr"), Intent.SCREEN_POSITION)

    def test_top_still_screen_top(self):
        """'top N ...' must NOT be confused with SCREEN_POSITION."""
        self.assertEqual(self.parser.parse_intent("top 5 NSE by cagr"), Intent.SCREEN_TOP)

    def test_lowest_still_screen_top(self):
        self.assertEqual(self.parser.parse_intent("lowest 10 NSE by cagr"), Intent.SCREEN_TOP)

    # --- extract_position_params ---

    def test_params_worst_cagr(self):
        p = self.parser.extract_position_params("worst NSE by cagr")
        self.assertEqual(p["position"], 1)
        self.assertTrue(p["from_end"])
        self.assertEqual(p["metric"], "cagr")
        self.assertEqual(p["exchange"], Exchange.NSE)

    def test_params_best_score(self):
        p = self.parser.extract_position_params("best BSE by score")
        self.assertEqual(p["position"], 1)
        self.assertFalse(p["from_end"])
        self.assertEqual(p["metric"], "score")

    def test_params_2nd_best_digit_ordinal(self):
        p = self.parser.extract_position_params("2nd best NSE by cagr")
        self.assertEqual(p["position"], 2)
        self.assertFalse(p["from_end"])

    def test_params_3rd_worst_digit_ordinal(self):
        p = self.parser.extract_position_params("3rd worst BSE by sharpe")
        self.assertEqual(p["position"], 3)
        self.assertTrue(p["from_end"])

    def test_params_second_last_word_ordinal(self):
        p = self.parser.extract_position_params("second last NSE by cagr")
        self.assertEqual(p["position"], 2)
        self.assertTrue(p["from_end"])

    def test_params_fifth_best_word_ordinal(self):
        p = self.parser.extract_position_params("fifth best NSE by volatility")
        self.assertEqual(p["position"], 5)
        self.assertFalse(p["from_end"])

    def test_params_no_exchange_returns_none(self):
        p = self.parser.extract_position_params("worst by cagr")
        self.assertIsNone(p["exchange"])

    def test_params_default_metric_is_cagr(self):
        p = self.parser.extract_position_params("best NSE")
        self.assertEqual(p["metric"], "cagr")

    # --- fetch_position ---

    def test_fetch_position_best_cagr_returns_highest(self):
        """best by CAGR should return the stock with the highest CAGR."""
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        result = ScreenerEngine.fetch_position(
            exchange="NSE", metric="cagr", position=1, from_end=False,
            horizon_years=3, data_loader=loader,
        )
        self.assertIsNotNone(result)
        # _DF_A has highest growth rate — should be ranked 1st best
        self.assertEqual(result["ticker"], "A")
        self.assertFalse(result["from_end"])
        self.assertEqual(result["direction_label"], "from the top")

    def test_fetch_position_worst_cagr_returns_lowest(self):
        """worst by CAGR should return the stock with the lowest CAGR."""
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        result = ScreenerEngine.fetch_position(
            exchange="NSE", metric="cagr", position=1, from_end=True,
            horizon_years=3, data_loader=loader,
        )
        self.assertIsNotNone(result)
        # _DF_C has the lowest growth → worst CAGR
        self.assertEqual(result["ticker"], "C")
        self.assertTrue(result["from_end"])
        self.assertEqual(result["direction_label"], "from the bottom")

    def test_fetch_position_worst_volatility_returns_highest_vol(self):
        """
        ⚠️ Critical: worst by volatility = HIGHEST volatility
        (volatility is lower_is_better, so worst = highest value).
        All three test DFs have near-identical tiny volatility (growth=constant),
        so we build one high-vol DF to confirm the correct stock is returned.
        """
        import numpy as np
        # High-volatility DF: prices jump up and down wildly
        n = 300
        dates  = pd.date_range(end=datetime.today(), periods=n, freq="B")
        prices = [1000 + 500 * ((-1) ** i) for i in range(n)]   # alternating 500/1500
        df_high_vol = pd.DataFrame({
            "Date": dates, "Open": prices, "High": prices,
            "Low": prices, "Close": prices, "Adj Close": prices,
            "Volume": [1_000_000] * n,
        })

        loader = self._mock_loader(["CALM", "WILD"], [_DF_A, df_high_vol])
        result = ScreenerEngine.fetch_position(
            exchange="NSE", metric="volatility", position=1, from_end=True,
            horizon_years=3, data_loader=loader,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["ticker"], "WILD",
                         "worst by volatility must return the most volatile stock")

    def test_fetch_position_2nd_best_not_first(self):
        """2nd best must differ from 1st best."""
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        best   = ScreenerEngine.fetch_position("NSE", "cagr", 1, False, 3, loader)
        second = ScreenerEngine.fetch_position("NSE", "cagr", 2, False, 3, loader)
        self.assertIsNotNone(best)
        self.assertIsNotNone(second)
        self.assertNotEqual(best["ticker"], second["ticker"])

    def test_fetch_position_out_of_range_returns_none(self):
        loader = self._mock_loader(["A"], [_DF_A])
        result = ScreenerEngine.fetch_position("NSE", "cagr", 100, False, 3, loader)
        self.assertIsNone(result)

    def test_fetch_position_rank_label_correct(self):
        loader = self._mock_loader(["A", "B", "C"], [_DF_A, _DF_B, _DF_C])
        result = ScreenerEngine.fetch_position("NSE", "cagr", 3, False, 3, loader)
        self.assertIsNotNone(result)
        self.assertEqual(result["rank"], 3)
        self.assertEqual(result["rank_label"], "3rd")

    def test_deterministic_tie_breaking(self):
        """Identical metric values must always produce alphabetical ordering."""
        # All three DFs are the same → same CAGR → tie
        loader = self._mock_loader(["ZZZ", "AAA", "MMM"], [_DF_A, _DF_A, _DF_A])
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=3,
            horizon_years=3, direction="desc", data_loader=loader,
        )
        tickers = [r["ticker"] for r in results]
        self.assertEqual(tickers, sorted(tickers),
                         "Tied values must produce alphabetical ticker order")

    # --- format_position_result ---

    def _make_position_result(self):
        return {
            "rank": 2, "rank_label": "2nd",
            "ticker": "RELIANCE", "value": 0.18,
            "metric": "cagr", "display_value": "+18.00%",
            "display_name": "CAGR",
            "from_end": False,
            "direction_label": "from the top",
        }

    def test_format_position_result_contains_ticker(self):
        out = self.gen.format_position_result(
            self._make_position_result(),
            position=2, from_end=False, metric="cagr",
            exchange="NSE", horizon_years=3,
        )
        self.assertIn("RELIANCE", out)

    def test_format_position_result_contains_rank_label(self):
        out = self.gen.format_position_result(
            self._make_position_result(),
            position=2, from_end=False, metric="cagr",
            exchange="NSE", horizon_years=3,
        )
        self.assertIn("2nd", out)
        self.assertIn("from the top", out)

    def test_format_position_result_none_graceful(self):
        out = self.gen.format_position_result(
            None, position=100, from_end=False, metric="cagr",
            exchange="NSE", horizon_years=3,
        )
        self.assertIn("Not enough data", out)
        self.assertIn("100th", out)

    def test_format_position_result_worst_label(self):
        result = self._make_position_result()
        result["from_end"] = True
        result["direction_label"] = "from the bottom"
        out = self.gen.format_position_result(
            result, position=2, from_end=True, metric="cagr",
            exchange="NSE", horizon_years=3,
        )
        self.assertIn("Worst", out)
        self.assertIn("from the bottom", out)


# ===========================================================================
# 8. Sort / Filter for Analysis Results
# ===========================================================================

class TestSortResults(unittest.TestCase):

    def setUp(self):
        self.parser = IntentParser()
        self.gen = ResponseGenerator.__new__(ResponseGenerator)
        self.gen._loader = MagicMock()
        self.gen._engine = MagicMock()

    # ---- Intent detection ----

    def test_sort_by_risk_intent(self):
        self.assertEqual(self.parser.parse_intent("sort by risk"), Intent.SORT_RESULTS)

    def test_sort_by_returns_intent(self):
        self.assertEqual(self.parser.parse_intent("sort by returns"), Intent.SORT_RESULTS)

    def test_worst_first_intent(self):
        self.assertEqual(self.parser.parse_intent("worst first"), Intent.SORT_RESULTS)

    def test_order_by_volume_intent(self):
        self.assertEqual(self.parser.parse_intent("order by volume"), Intent.SORT_RESULTS)

    def test_ascending_keyword_intent(self):
        self.assertEqual(self.parser.parse_intent("ascending"), Intent.SORT_RESULTS)

    # ---- extract_sort_params ----

    def test_params_sort_by_returns(self):
        p = self.parser.extract_sort_params("sort by returns")
        self.assertEqual(p["field"], "cagr")
        self.assertEqual(p["direction"], "desc")   # higher_is_better → desc default

    def test_params_sort_by_risk_defaults_asc(self):
        """sort by risk → volatility asc (safest first — lower_is_better)."""
        p = self.parser.extract_sort_params("sort by risk")
        self.assertEqual(p["field"], "volatility")
        self.assertEqual(p["direction"], "asc")    # lower_is_better → asc default

    def test_params_worst_first_direction(self):
        p = self.parser.extract_sort_params("worst first")
        self.assertEqual(p["direction"], "asc")

    def test_params_explicit_asc_override(self):
        """Explicit 'asc' keyword overrides the lower-is-better default for any field."""
        p = self.parser.extract_sort_params("sort by cagr asc")
        self.assertEqual(p["field"], "cagr")
        self.assertEqual(p["direction"], "asc")

    def test_params_explicit_desc_override(self):
        p = self.parser.extract_sort_params("sort by risk desc")
        self.assertEqual(p["field"], "volatility")
        self.assertEqual(p["direction"], "desc")   # explicit desc overrides default

    def test_params_sort_by_volume(self):
        p = self.parser.extract_sort_params("sort by volume")
        self.assertEqual(p["field"], "avg_volume")
        self.assertEqual(p["direction"], "desc")

    def test_params_default_field_score(self):
        """No field keyword → default to score."""
        p = self.parser.extract_sort_params("best first")
        self.assertEqual(p["field"], "score")
        self.assertEqual(p["direction"], "desc")

    # ---- Sorting logic ----

    def _make_results(self):
        """Three analysis result dicts with different CAGR and volatility."""
        def _r(ticker, score, cagr, vol, price=1000.0):
            return {
                "ticker": ticker,
                "total_score": score,
                "component_scores": {},
                "weights_used": {},
                "metrics": {
                    "cagr": cagr,
                    "volatility": vol,
                    "avg_volume": 1_000_000.0,
                    "latest_price": price,
                },
                "rank": 1,
                "rank_label": "1st",
            }
        return [
            _r("A", 0.90, 0.20, 0.10),  # best score, high cagr, low vol
            _r("B", 0.70, 0.15, 0.30),  # mid score
            _r("C", 0.50, 0.05, 0.05),  # low score, low cagr, safest
        ]

    def test_sort_by_cagr_desc_order(self):
        results = self._make_results()
        field, direction = "cagr", "desc"
        sorted_r = sorted(results,
                          key=lambda r: r["metrics"].get(field, 0.0),
                          reverse=(direction == "desc"))
        tickers = [r["ticker"] for r in sorted_r]
        self.assertEqual(tickers, ["A", "B", "C"])

    def test_sort_by_volatility_asc_safest_first(self):
        results = self._make_results()
        sorted_r = sorted(results,
                          key=lambda r: r["metrics"].get("volatility", 0.0),
                          reverse=False)
        tickers = [r["ticker"] for r in sorted_r]
        # C has lowest vol (0.05), A has 0.10, B has 0.30
        self.assertEqual(tickers, ["C", "A", "B"])

    def test_context_results_not_mutated(self):
        """Sorting must not change the original context.results order."""
        original = self._make_results()
        import copy
        original_copy = copy.deepcopy(original)
        _ = sorted(original, key=lambda r: r["metrics"]["volatility"])
        # original itself must be unchanged
        for orig, copy_ in zip(original, original_copy):
            self.assertEqual(orig["ticker"], copy_["ticker"])

    # ---- format_sorted_table ----

    def test_format_sorted_table_has_re_sorted_header(self):
        results = self._make_results()
        out = self.gen.format_sorted_table(results, "cagr", "desc", budget=None)
        self.assertIn("Re-sorted by", out)
        self.assertIn("CAGR", out)

    def test_format_sorted_table_re_numbers_ranks(self):
        results = self._make_results()
        out = self.gen.format_sorted_table(results, "cagr", "desc", budget=None)
        self.assertIn("1st", out)
        self.assertIn("2nd", out)
        self.assertIn("3rd", out)

    def test_format_sorted_table_shows_score_leader(self):
        results = self._make_results()
        out = self.gen.format_sorted_table(results, "volatility", "asc", budget=None)
        self.assertIn("Best pick (by score)", out)   # score leader A mentioned
        self.assertIn("A", out)


if __name__ == "__main__":
    unittest.main()
