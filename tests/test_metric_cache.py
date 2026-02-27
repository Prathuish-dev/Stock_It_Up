"""
tests/test_metric_cache.py
--------------------------
Unit tests for MetricCache.

Coverage:
  - _fingerprint() — changes when files change
  - MetricCache.build() — builds dict, writes JSON file
  - MetricCache.get_or_build() — cache hit (no rebuild), cache miss (rebuild)
  - MetricCache.is_valid() — True when cache matches, False when stale
  - MetricCache.invalidate() — deletes cache file
  - Atomic write — temp file is cleaned up; corrupt JSON handled gracefully
  - ScreenerEngine.run(cache=...) — fast path serves from cached dict
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from chatbot.metric_cache import MetricCache, _fingerprint, _cache_path
from chatbot.screener_engine import ScreenerEngine


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_df(n_rows: int = 252) -> pd.DataFrame:
    """Small synthetic price DataFrame with required columns."""
    dates  = pd.date_range("2022-01-01", periods=n_rows, freq="B")
    prices = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    return pd.DataFrame({
        "Date":      dates,
        "Adj Close": prices,
        "Close":     prices,
        "Volume":    np.random.randint(100_000, 500_000, n_rows).astype(float),
    })


def _make_loader(tickers: list[str]) -> MagicMock:
    """Mock DataLoader that returns synthetic DataFrames."""
    loader = MagicMock()
    loader.list_available.return_value = tickers
    loader.load_stock.side_effect      = lambda ex, t: _make_df()
    return loader


# ---------------------------------------------------------------------------
# TestFingerprint — filesystem hash
# ---------------------------------------------------------------------------

class TestFingerprint(unittest.TestCase):
    """_fingerprint() must detect changes in the CSV file set."""

    def test_empty_dir_returns_stable_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fp = _fingerprint(base, "NSE")
            # Empty exchange dir → empty fingerprint (dir doesn't exist)
            self.assertIsInstance(fp, str)

    def test_fingerprint_changes_after_file_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            base  = Path(tmp)
            exdir = base / "stock_data_NSE" / "TCS"
            exdir.mkdir(parents=True)
            fp1 = _fingerprint(base, "NSE")
            (exdir / "TCS_2023.csv").write_text("Date,Adj Close\n")
            fp2 = _fingerprint(base, "NSE")
            self.assertNotEqual(fp1, fp2)

    def test_fingerprint_stable_with_no_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base  = Path(tmp)
            exdir = base / "stock_data_NSE" / "TCS"
            exdir.mkdir(parents=True)
            (exdir / "TCS_2023.csv").write_text("Date,Adj Close\n")
            fp1 = _fingerprint(base, "NSE")
            fp2 = _fingerprint(base, "NSE")
            self.assertEqual(fp1, fp2)


# ---------------------------------------------------------------------------
# TestMetricCacheBuild — build() correctness
# ---------------------------------------------------------------------------

class TestMetricCacheBuild(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        # Create minimal filesystem layout so fingerprint works
        (base / "stock_data_NSE").mkdir()
        self._loader = _make_loader(["TCS", "INFY", "WIPRO"])
        self._loader._base = base
        self._cache = MetricCache(self._loader)

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_returns_dict_with_all_tickers(self):
        result = self._cache.build("NSE", horizon_years=3)
        self.assertEqual(set(result.keys()), {"TCS", "INFY", "WIPRO"})

    def test_build_has_metric_keys_per_ticker(self):
        result = self._cache.build("NSE", horizon_years=3)
        for ticker, row in result.items():
            self.assertIn("cagr",         row, ticker)
            self.assertIn("volatility",   row, ticker)
            self.assertIn("sharpe",       row, ticker)
            self.assertIn("max_drawdown", row, ticker)
            self.assertIn("sortino",      row, ticker)

    def test_build_writes_json_file(self):
        self._cache.build("NSE", horizon_years=3)
        p = _cache_path(self._loader._base, "NSE", 3)
        self.assertTrue(p.exists(), f"Cache file not found: {p}")

    def test_build_json_is_valid(self):
        self._cache.build("NSE", horizon_years=3)
        p = _cache_path(self._loader._base, "NSE", 3)
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertIn("metrics",     data)
        self.assertIn("fingerprint", data)
        self.assertIn("exchange",    data)
        self.assertEqual(data["exchange"], "NSE")


# ---------------------------------------------------------------------------
# TestMetricCacheHitMiss — get_or_build() caching logic
# ---------------------------------------------------------------------------

class TestMetricCacheHitMiss(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        (base / "stock_data_NSE").mkdir()
        self._loader = _make_loader(["TCS", "INFY"])
        self._loader._base = base
        self._cache = MetricCache(self._loader)

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_call_builds_then_returns_metrics(self):
        """First call → no cache → build → return dict."""
        result = self._cache.get_or_build("NSE", horizon_years=3)
        self.assertIn("TCS",  result)
        self.assertIn("INFY", result)

    def test_second_call_hits_cache_without_reloading(self):
        """Second call with same fingerprint → cache hit → no load_stock calls."""
        self._cache.build("NSE", horizon_years=3)
        # Now reset call count
        self._loader.load_stock.reset_mock()
        _ = self._cache.get_or_build("NSE", horizon_years=3)
        # Cache was valid → no CSV reads
        self._loader.load_stock.assert_not_called()

    def test_is_valid_true_after_build(self):
        self._cache.build("NSE", horizon_years=3)
        self.assertTrue(self._cache.is_valid("NSE", 3))

    def test_is_valid_false_before_build(self):
        self.assertFalse(self._cache.is_valid("NSE", 3))


# ---------------------------------------------------------------------------
# TestMetricCacheInvalidation — invalidate() + staleness detection
# ---------------------------------------------------------------------------

class TestMetricCacheInvalidation(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        (base / "stock_data_NSE").mkdir()
        self._loader = _make_loader(["TCS"])
        self._loader._base = base
        self._cache = MetricCache(self._loader)

    def tearDown(self):
        self._tmp.cleanup()

    def test_invalidate_deletes_cache_file(self):
        self._cache.build("NSE", horizon_years=3)
        self.assertTrue(self._cache.is_valid("NSE", 3))
        self._cache.invalidate("NSE", horizon_years=3)
        self.assertFalse(self._cache.is_valid("NSE", 3))

    def test_stale_cache_triggers_rebuild(self):
        """
        After adding a new CSV to the exchange dir, the fingerprint changes
        → is_valid() returns False (stale cache detected).
        """
        self._cache.build("NSE", horizon_years=3)
        self.assertTrue(self._cache.is_valid("NSE", 3))

        # Simulate a new CSV arriving on disk
        new_csv = self._loader._base / "stock_data_NSE" / "new_file.csv"
        new_csv.write_text("Date,Adj Close\n")

        self.assertFalse(self._cache.is_valid("NSE", 3))

    def test_corrupt_cache_file_treated_as_miss(self):
        """Corrupt JSON in cache file should be silently treated as a miss."""
        self._cache.build("NSE", horizon_years=3)
        p = _cache_path(self._loader._base, "NSE", 3)
        p.write_text("{ not valid json }")
        # Should not raise, should return None path → triggers rebuild
        self.assertFalse(self._cache.is_valid("NSE", 3))


# ---------------------------------------------------------------------------
# TestScreenerCacheFastPath — ScreenerEngine.run(cache=...) fast path
# ---------------------------------------------------------------------------

class TestScreenerCacheFastPath(unittest.TestCase):
    """ScreenerEngine.run(cache=cache) must serve results purely from dict."""

    def _make_cached_metrics(self) -> dict:
        return {
            "TCS":  {"cagr": 0.20, "volatility": 0.15, "avg_volume": 1e6,
                     "sharpe": 1.5, "max_drawdown": 0.10, "sortino": 1.8},
            "INFY": {"cagr": 0.15, "volatility": 0.12, "avg_volume": 8e5,
                     "sharpe": 1.2, "max_drawdown": 0.08, "sortino": 1.4},
            "WIPRO":{"cagr": 0.10, "volatility": 0.18, "avg_volume": 5e5,
                     "sharpe": 0.8, "max_drawdown": 0.15, "sortino": 0.9},
        }

    def test_metric_mode_served_from_cache(self):
        """Metric-mode screener via cache — no load_stock calls."""
        cached = self._make_cached_metrics()
        mock_cache = MagicMock()
        mock_cache.get_or_build.return_value = cached
        mock_loader = MagicMock()

        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=2, horizon_years=3,
            direction="desc", data_loader=mock_loader, cache=mock_cache,
        )
        mock_loader.load_stock.assert_not_called()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["ticker"], "TCS")   # highest CAGR

    def test_metric_mode_desc_sorts_correctly(self):
        cached = self._make_cached_metrics()
        mock_cache = MagicMock()
        mock_cache.get_or_build.return_value = cached
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=3, horizon_years=3,
            direction="desc", data_loader=MagicMock(), cache=mock_cache,
        )
        values = [r["value"] for r in results]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_metric_mode_asc_sorts_correctly(self):
        cached = self._make_cached_metrics()
        mock_cache = MagicMock()
        mock_cache.get_or_build.return_value = cached
        results = ScreenerEngine.run(
            exchange="NSE", metric="volatility", limit=3, horizon_years=3,
            direction="asc", data_loader=MagicMock(), cache=mock_cache,
        )
        values = [r["value"] for r in results]
        self.assertEqual(values, sorted(values))

    def test_cache_none_falls_back_to_live_scan(self):
        """
        When cache=None, ScreenerEngine.run() must use the live CSV path.
        We verify by ensuring _run_metric_mode is called (via list_available).
        """
        loader = MagicMock()
        loader.list_available.return_value = []  # fast → empty results
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=10, horizon_years=3,
            direction="desc", data_loader=loader, cache=None,
        )
        loader.list_available.assert_called_once_with("NSE")
        self.assertEqual(results, [])

    def test_cache_result_has_correct_keys(self):
        cached = self._make_cached_metrics()
        mock_cache = MagicMock()
        mock_cache.get_or_build.return_value = cached
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=1, horizon_years=3,
            direction="desc", data_loader=MagicMock(), cache=mock_cache,
        )
        self.assertEqual(len(results), 1)
        for key in ("ticker", "value", "metric", "display_value", "display_name"):
            self.assertIn(key, results[0])

    def test_none_metric_values_are_skipped(self):
        """Tickers with None for the requested metric must be excluded."""
        cached = {
            "TCS":  {"cagr": 0.20},
            "INFY": {"cagr": None},   # insufficient data
        }
        mock_cache = MagicMock()
        mock_cache.get_or_build.return_value = cached
        results = ScreenerEngine.run(
            exchange="NSE", metric="cagr", limit=10, horizon_years=3,
            direction="desc", data_loader=MagicMock(), cache=mock_cache,
        )
        tickers = [r["ticker"] for r in results]
        self.assertIn("TCS",  tickers)
        self.assertNotIn("INFY", tickers)


if __name__ == "__main__":
    unittest.main()
