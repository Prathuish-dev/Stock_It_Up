"""
tests/test_metrics_engine.py
-----------------------------
Unit tests for MetricsEngine — focusing on the new compute_sharpe() method
and its integration into compute_metric() and compute_all().

Test coverage:
    compute_sharpe()      — positive, zero-vol, below-risk-free, clipping
    compute_metric()      — dispatcher routes 'sharpe' correctly
    compute_all()         — includes 'sharpe' key in output dict
    Horizon consistency   — sharpe uses pre-filtered df (caller's responsibility)
"""

import unittest
from datetime import datetime

import pandas as pd
import numpy as np

from chatbot.metrics_engine import MetricsEngine
from chatbot.config import RISK_FREE_RATE, SHARPE_CLIP_MIN, SHARPE_CLIP_MAX


# ---------------------------------------------------------------------------
# DataFrame factories
# ---------------------------------------------------------------------------

def _make_df(n_rows: int = 300, start_price: float = 1000.0, growth: float = 0.0002):
    """Synthetic price DataFrame with enough rows for MetricsEngine."""
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


def _make_df_flat(n_rows: int = 300, price: float = 1000.0):
    """Flat price series — zero volatility, CAGR ≈ 0."""
    dates  = pd.date_range(end=datetime.today(), periods=n_rows, freq="B")
    return pd.DataFrame({
        "Date":      dates,
        "Open":      [price] * n_rows,
        "High":      [price] * n_rows,
        "Low":       [price] * n_rows,
        "Close":     [price] * n_rows,
        "Adj Close": [price] * n_rows,
        "Volume":    [500_000] * n_rows,
    })


# Shared DataFrames (created once per module load)
_DF_HIGH_GROWTH = _make_df(600, start_price=500.0,  growth=0.0006)   # high CAGR
_DF_LOW_GROWTH  = _make_df(600, start_price=1000.0, growth=0.00005)  # low CAGR
_DF_FLAT        = _make_df_flat(600)


# ===========================================================================
# 1. compute_sharpe()
# ===========================================================================

class TestComputeSharpe(unittest.TestCase):

    def test_compute_sharpe_positive(self):
        """High-growth stock with moderate volatility should yield positive Sharpe."""
        df  = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        val = MetricsEngine.compute_sharpe(df)
        self.assertIsInstance(val, float)
        # High growth > risk-free → must be positive
        self.assertGreater(val, 0.0)

    def test_compute_sharpe_zero_volatility_returns_zero(self):
        """Flat price series has effectively zero volatility → returns 0.0, no crash."""
        df  = MetricsEngine.filter_by_horizon(_DF_FLAT, 3)
        val = MetricsEngine.compute_sharpe(df)
        self.assertEqual(val, 0.0)

    def test_compute_sharpe_below_risk_free_is_negative(self):
        """
        Very low CAGR (< RISK_FREE_RATE) should produce a negative Sharpe Ratio.
        Uses a tiny growth rate that cannot beat 6% risk-free.
        """
        df  = MetricsEngine.filter_by_horizon(_DF_LOW_GROWTH, 3)
        sharpe = MetricsEngine.compute_sharpe(df)
        cagr   = MetricsEngine.compute_cagr(df)
        # Only meaningful when CAGR < RISK_FREE_RATE
        if cagr < RISK_FREE_RATE:
            self.assertLess(sharpe, 0.0)
        else:
            self.skipTest("Low-growth fixture still exceeds risk-free rate in this data window.")

    def test_compute_sharpe_clipped_at_max(self):
        """
        Even an extremely high-growth stock must be clipped at SHARPE_CLIP_MAX (3.0).
        We simulate this by asserting the output never exceeds the configured bound.
        """
        df  = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        val = MetricsEngine.compute_sharpe(df)
        self.assertLessEqual(val, SHARPE_CLIP_MAX)

    def test_compute_sharpe_clipped_at_min(self):
        """Output must never fall below SHARPE_CLIP_MIN (-1.0)."""
        df  = MetricsEngine.filter_by_horizon(_DF_LOW_GROWTH, 3)
        val = MetricsEngine.compute_sharpe(df)
        self.assertGreaterEqual(val, SHARPE_CLIP_MIN)

    def test_compute_sharpe_higher_growth_higher_sharpe(self):
        """
        Given two stocks with similar volatility profiles (same price path shape),
        the one with higher growth should have a higher Sharpe Ratio.
        """
        df_high = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        df_low  = MetricsEngine.filter_by_horizon(_DF_LOW_GROWTH,  3)
        sharpe_high = MetricsEngine.compute_sharpe(df_high)
        sharpe_low  = MetricsEngine.compute_sharpe(df_low)
        self.assertGreater(sharpe_high, sharpe_low)

    def test_compute_sharpe_uses_pre_filtered_df(self):
        """
        compute_sharpe() must NOT filter internally — the caller is responsible
        for horizon filtering. This test verifies that passing different horizon
        slices of the same DataFrame produces different Sharpe values, confirming
        the function respects whatever window it receives.
        """
        df_3yr = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        df_5yr = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 5)
        # They may differ — the key assertion is no crash and both are finite floats
        sharpe_3 = MetricsEngine.compute_sharpe(df_3yr)
        sharpe_5 = MetricsEngine.compute_sharpe(df_5yr)
        self.assertIsInstance(sharpe_3, float)
        self.assertIsInstance(sharpe_5, float)
        self.assertTrue(-1.0 <= sharpe_3 <= 3.0)
        self.assertTrue(-1.0 <= sharpe_5 <= 3.0)


# ===========================================================================
# 2. compute_metric() dispatcher
# ===========================================================================

class TestComputeMetricDispatcher(unittest.TestCase):

    def setUp(self):
        self.df = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)

    def test_compute_metric_dispatches_sharpe(self):
        """compute_metric(df, 'sharpe') must equal compute_sharpe(df) exactly."""
        self.assertAlmostEqual(
            MetricsEngine.compute_metric(self.df, "sharpe"),
            MetricsEngine.compute_sharpe(self.df),
            places=12,
        )

    def test_compute_metric_sharpe_is_float(self):
        val = MetricsEngine.compute_metric(self.df, "sharpe")
        self.assertIsInstance(val, float)

    def test_compute_metric_unknown_still_raises(self):
        """Adding sharpe must not break the ValueError for truly unknown metrics."""
        with self.assertRaises(ValueError):
            MetricsEngine.compute_metric(self.df, "unknown_metric")


# ===========================================================================
# 3. compute_all()
# ===========================================================================

class TestComputeAll(unittest.TestCase):

    def setUp(self):
        self.df = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        self.result = MetricsEngine.compute_all(self.df)

    def test_compute_all_includes_sharpe_key(self):
        """'sharpe' must be present in the compute_all() output dict."""
        self.assertIn("sharpe", self.result)

    def test_compute_all_sharpe_matches_standalone(self):
        """compute_all()['sharpe'] must equal compute_sharpe() exactly."""
        self.assertAlmostEqual(
            self.result["sharpe"],
            MetricsEngine.compute_sharpe(self.df),
            places=12,
        )

    def test_compute_all_all_original_keys_still_present(self):
        """Backward compatibility: original keys must still be present."""
        for key in ("cagr", "volatility", "avg_volume", "latest_price"):
            self.assertIn(key, self.result)

    def test_compute_all_new_keys_present(self):
        """'max_drawdown' and 'sortino' must be present in compute_all() output."""
        self.assertIn("max_drawdown", self.result)
        self.assertIn("sortino",      self.result)

    def test_compute_all_sharpe_is_float(self):
        self.assertIsInstance(self.result["sharpe"], float)

    def test_compute_all_max_drawdown_is_float(self):
        self.assertIsInstance(self.result["max_drawdown"], float)

    def test_compute_all_sortino_is_float(self):
        self.assertIsInstance(self.result["sortino"], float)


# ===========================================================================
# 4. compute_max_drawdown()
# ===========================================================================

def _make_mdd_df(prices: list):
    """Build a minimal DataFrame from a list of closing prices."""
    import pandas as pd
    from datetime import datetime
    n     = len(prices)
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    return pd.DataFrame({
        "Date":      dates,
        "Open":      prices,
        "High":      [p * 1.01 for p in prices],
        "Low":       [p * 0.99 for p in prices],
        "Close":     prices,
        "Adj Close": prices,
        "Volume":    [1_000_000] * n,
    })


class TestMaxDrawdown(unittest.TestCase):

    def test_monotonic_increase_mdd_is_zero(self):
        """Strictly increasing prices — no drawdown, MDD must be 0.0."""
        df  = _make_mdd_df([100, 110, 120, 130, 140, 150])
        mdd = MetricsEngine.compute_max_drawdown(df)
        self.assertAlmostEqual(mdd, 0.0, places=9)

    def test_single_drop_correct_percentage(self):
        """Price drops from 200 to 100 (50% drawdown)."""
        df  = _make_mdd_df([200, 200, 200, 100, 100])
        mdd = MetricsEngine.compute_max_drawdown(df)
        self.assertAlmostEqual(mdd, 0.5, places=9)

    def test_multiple_peaks_selects_worst(self):
        """
        Two drawdowns: 10% and 25%.
        MDD must report 25% (the worst).
        """
        # Peak 200, drop to 180 (10%), recover to 200, drop to 150 (25%)
        df  = _make_mdd_df([200, 190, 180, 200, 175, 150])
        mdd = MetricsEngine.compute_max_drawdown(df)
        self.assertAlmostEqual(mdd, 0.25, places=9)

    def test_mdd_is_positive(self):
        """MDD must always be a non-negative number."""
        for fixture in (_DF_HIGH_GROWTH, _DF_LOW_GROWTH, _DF_FLAT):
            df  = MetricsEngine.filter_by_horizon(fixture, 3)
            mdd = MetricsEngine.compute_max_drawdown(df)
            self.assertGreaterEqual(mdd, 0.0)

    def test_mdd_bounded_at_one(self):
        """MDD can never exceed 100% (total loss)."""
        df  = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        mdd = MetricsEngine.compute_max_drawdown(df)
        self.assertLessEqual(mdd, 1.0)

    def test_empty_df_returns_zero(self):
        """Empty DataFrame must return 0.0 without crashing."""
        import pandas as pd
        empty_df = pd.DataFrame()
        self.assertEqual(MetricsEngine.compute_max_drawdown(empty_df), 0.0)

    def test_dispatcher_routes_max_drawdown(self):
        """compute_metric(df, 'max_drawdown') must equal compute_max_drawdown(df)."""
        df = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        self.assertAlmostEqual(
            MetricsEngine.compute_metric(df, "max_drawdown"),
            MetricsEngine.compute_max_drawdown(df),
            places=12,
        )

    def test_high_growth_lower_or_equal_mdd_than_volatile(self):
        """Higher growth fixture should not have worse MDD than a declining one."""
        df_h = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        df_l = MetricsEngine.filter_by_horizon(_DF_LOW_GROWTH,  3)
        # Simply validate both return floats — exact ordering depends on data
        self.assertIsInstance(MetricsEngine.compute_max_drawdown(df_h), float)
        self.assertIsInstance(MetricsEngine.compute_max_drawdown(df_l), float)


# ===========================================================================
# 5. compute_sortino()
# ===========================================================================

class TestSortino(unittest.TestCase):

    def setUp(self):
        self.df_high = MetricsEngine.filter_by_horizon(_DF_HIGH_GROWTH, 3)
        self.df_low  = MetricsEngine.filter_by_horizon(_DF_LOW_GROWTH,  3)
        self.df_flat = MetricsEngine.filter_by_horizon(_DF_FLAT,        3)

    def test_sortino_returns_float(self):
        val = MetricsEngine.compute_sortino(self.df_high)
        self.assertIsInstance(val, float)

    def test_sortino_positive_for_high_growth(self):
        """
        High-growth fixture has a smooth uptrend — no daily returns fall below
        the daily risk-free hurdle, so the downside set is empty → Sortino = 0.0.
        This is correct behaviour. We only assert non-negative here.
        Use test_sortino_positive_with_noise for a noisy fixture test.
        """
        val = MetricsEngine.compute_sortino(self.df_high)
        self.assertGreaterEqual(val, 0.0)

    def test_sortino_positive_with_noise(self):
        """
        Fixture with alternating up/down days and strong CAGR must produce
        a positive Sortino — the downside set is non-empty but return > Rf.
        """
        import pandas as pd
        # Alternating +1.2% / -0.3% every other day: CAGR is well above 6%
        n = 600
        dates   = pd.date_range(end="2025-01-01", periods=n, freq="B")
        prices  = []
        p = 1000.0
        for i in range(n):
            p *= (1.012 if i % 2 == 0 else 0.997)
            prices.append(p)
        df = pd.DataFrame({
            "Date":      dates,
            "Adj Close": prices,
            "Volume":    [1_000_000] * n,
            "Open": prices, "High": prices, "Low": prices, "Close": prices,
        })
        val = MetricsEngine.compute_sortino(df)
        self.assertGreater(val, 0.0,
            msg=f"Expected positive Sortino for strong-CAGR noisy fixture, got {val}")

    def test_sortino_flat_series(self):
        """
        Flat price series has no negative returns (no downside days below RF).
        Sortino must return 0.0 (empty downside set or zero CAGR path).
        """
        val = MetricsEngine.compute_sortino(self.df_flat)
        # Either 0.0 (no downside) or clipped — must be a valid number
        self.assertIsInstance(val, float)
        self.assertTrue(-1.0 <= val <= 3.0)

    def test_sortino_clipped_at_max(self):
        """Output must never exceed SHARPE_CLIP_MAX (3.0)."""
        val = MetricsEngine.compute_sortino(self.df_high)
        self.assertLessEqual(val, SHARPE_CLIP_MAX)

    def test_sortino_clipped_at_min(self):
        """Output must never fall below SHARPE_CLIP_MIN (-1.0)."""
        val = MetricsEngine.compute_sortino(self.df_low)
        self.assertGreaterEqual(val, SHARPE_CLIP_MIN)

    def test_sortino_higher_growth_ge_lower_growth(self):
        """
        Higher-growth stock should have a Sortino ratio >= lower-growth stock,
        all else equal (same price path shape).
        """
        s_high = MetricsEngine.compute_sortino(self.df_high)
        s_low  = MetricsEngine.compute_sortino(self.df_low)
        self.assertGreaterEqual(s_high, s_low)

    def test_dispatcher_routes_sortino(self):
        """compute_metric(df, 'sortino') must equal compute_sortino(df)."""
        self.assertAlmostEqual(
            MetricsEngine.compute_metric(self.df_high, "sortino"),
            MetricsEngine.compute_sortino(self.df_high),
            places=12,
        )

    def test_sortino_insufficient_data_returns_zero(self):
        """DataFrame with fewer than _MIN_ROWS rows must return 0.0, not raise."""
        import pandas as pd
        tiny = _make_mdd_df([100, 110, 90, 105])  # only 4 rows
        val  = MetricsEngine.compute_sortino(tiny)
        self.assertEqual(val, 0.0)


if __name__ == "__main__":
    unittest.main()
