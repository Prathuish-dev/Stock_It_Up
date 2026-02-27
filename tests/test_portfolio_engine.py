"""
tests/test_portfolio_engine.py
-------------------------------
Unit tests for PortfolioEngine.

Test coverage:
    Empty / single-stock edge cases
    Proportional allocation correctness
    Softmax allocation properties
    Risk-adjusted allocation direction
    Constraint application (cap / floor)
    Normalisation helper
    Error handling (missing score, bad method)
"""

import math
import unittest

from chatbot.portfolio_engine import PortfolioEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alloc(ticker_scores: dict, **kwargs) -> dict:
    """Return {ticker: allocation} for easy assertion."""
    data = [{"ticker": t, "total_score": s} for t, s in ticker_scores.items()]
    result = PortfolioEngine.allocate(data, **kwargs)
    return {r["ticker"]: r["allocation"] for r in result}


def _sum_alloc(result: list) -> float:
    return sum(r["allocation"] for r in result)


# ===========================================================================
# 1. Edge Cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        self.assertEqual(PortfolioEngine.allocate([]), [])

    def test_single_stock_gets_full_allocation(self):
        result = PortfolioEngine.allocate([{"ticker": "TCS", "total_score": 0.8}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["allocation"], 1.0)
        self.assertEqual(result[0]["ticker"], "TCS")

    def test_single_stock_total_score_preserved(self):
        result = PortfolioEngine.allocate([{"ticker": "TCS", "total_score": 0.4}])
        self.assertEqual(result[0]["total_score"], 0.4)

    def test_unknown_method_raises(self):
        data = [{"ticker": "A", "total_score": 0.5},
                {"ticker": "B", "total_score": 0.5}]
        with self.assertRaises(ValueError):
            PortfolioEngine.allocate(data, method="magic")

    def test_missing_total_score_raises(self):
        data = [{"ticker": "A"}, {"ticker": "B", "total_score": 0.5}]
        with self.assertRaises(ValueError):
            PortfolioEngine.allocate(data)


# ===========================================================================
# 2. Proportional Allocation
# ===========================================================================

class TestProportional(unittest.TestCase):

    def test_two_stocks_ratio(self):
        # Score 2:1 → allocation 2/3 : 1/3
        result = _alloc({"A": 2.0, "B": 1.0})
        self.assertAlmostEqual(result["A"], 2 / 3, places=4)
        self.assertAlmostEqual(result["B"], 1 / 3, places=4)

    def test_allocations_sum_to_one(self):
        data = [{"ticker": t, "total_score": s}
                for t, s in [("A", 0.82), ("B", 0.65), ("C", 0.50), ("D", 0.33)]]
        result = PortfolioEngine.allocate(data)
        self.assertAlmostEqual(_sum_alloc(result), 1.0, places=4)

    def test_equal_scores_equal_allocation(self):
        result = _alloc({"X": 0.5, "Y": 0.5})
        self.assertAlmostEqual(result["X"], 0.5, places=4)
        self.assertAlmostEqual(result["Y"], 0.5, places=4)

    def test_all_zero_scores_equal_fallback(self):
        result = _alloc({"A": 0.0, "B": 0.0, "C": 0.0})
        for v in result.values():
            self.assertAlmostEqual(v, 1 / 3, places=4)

    def test_higher_score_gets_higher_allocation(self):
        result = _alloc({"BEST": 0.9, "MID": 0.5, "WORST": 0.1})
        self.assertGreater(result["BEST"], result["MID"])
        self.assertGreater(result["MID"], result["WORST"])

    def test_output_contains_ticker_and_total_score(self):
        data = [{"ticker": "TCS", "total_score": 0.8},
                {"ticker": "INFY", "total_score": 0.6}]
        result = PortfolioEngine.allocate(data)
        keys = set(result[0].keys())
        self.assertIn("ticker", keys)
        self.assertIn("allocation", keys)
        self.assertIn("total_score", keys)

    def test_does_not_mutate_input(self):
        original = [{"ticker": "A", "total_score": 0.7},
                    {"ticker": "B", "total_score": 0.3}]
        import copy
        snapshot = copy.deepcopy(original)
        PortfolioEngine.allocate(original)
        self.assertEqual(original, snapshot)


# ===========================================================================
# 3. Softmax Allocation
# ===========================================================================

class TestSoftmax(unittest.TestCase):

    def test_allocations_sum_to_one(self):
        data = [{"ticker": t, "total_score": s}
                for t, s in [("A", 0.9), ("B", 0.5), ("C", 0.1)]]
        result = PortfolioEngine.allocate(data, method="softmax")
        self.assertAlmostEqual(_sum_alloc(result), 1.0, places=6)

    def test_higher_score_gets_higher_allocation(self):
        result = _alloc({"HIGH": 0.9, "LOW": 0.1}, method="softmax")
        self.assertGreater(result["HIGH"], result["LOW"])

    def test_all_positive_allocations(self):
        data = [{"ticker": t, "total_score": s}
                for t, s in [("A", 0.9), ("B", 0.5), ("C", 0.1)]]
        result = PortfolioEngine.allocate(data, method="softmax")
        for r in result:
            self.assertGreater(r["allocation"], 0)

    def test_equal_scores_near_equal_allocation(self):
        result = _alloc({"X": 0.5, "Y": 0.5}, method="softmax")
        self.assertAlmostEqual(result["X"], 0.5, places=4)

    def test_softmax_more_concentrated_than_proportional(self):
        """Softmax boosts the winner when score spread is large."""
        # Use scores far apart so exp() strongly amplifies the difference
        data = [{"ticker": t, "total_score": s}
                for t, s in [("A", 5.0), ("B", 1.0)]]
        prop = PortfolioEngine.allocate(data, method="proportional")
        soft = PortfolioEngine.allocate(data, method="softmax")
        prop_top = next(r["allocation"] for r in prop if r["ticker"] == "A")
        soft_top = next(r["allocation"] for r in soft if r["ticker"] == "A")
        self.assertGreater(soft_top, prop_top)


# ===========================================================================
# 4. Risk-Adjusted Allocation
# ===========================================================================

class TestRiskAdjusted(unittest.TestCase):

    def _ra(self, stocks, profile):
        data = [
            {"ticker": t, "total_score": s, "volatility": v}
            for t, s, v in stocks
        ]
        result = PortfolioEngine.allocate(data, method="risk_adjusted",
                                          risk_profile=profile)
        return {r["ticker"]: r["allocation"] for r in result}

    def test_conservative_penalises_high_volatility(self):
        # STABLE has lower volatility: LOW profile should prefer it
        result = self._ra(
            [("STABLE", 0.7, 0.10), ("RISKY", 0.7, 0.50)],
            "LOW",
        )
        self.assertGreater(result["STABLE"], result["RISKY"])

    def test_aggressive_rewards_high_volatility(self):
        result = self._ra(
            [("STABLE", 0.7, 0.10), ("RISKY", 0.7, 0.50)],
            "HIGH",
        )
        self.assertGreater(result["RISKY"], result["STABLE"])

    def test_medium_profile_uses_raw_scores(self):
        # With equal raw scores, MEDIUM → equal allocation regardless of vol
        result = self._ra(
            [("A", 0.5, 0.10), ("B", 0.5, 0.90)],
            "MEDIUM",
        )
        self.assertAlmostEqual(result["A"], result["B"], places=4)

    def test_none_profile_uses_raw_scores(self):
        result = self._ra(
            [("A", 0.6, 0.10), ("B", 0.4, 0.90)],
            None,
        )
        self.assertGreater(result["A"], result["B"])

    def test_allocations_sum_to_one(self):
        data = [
            {"ticker": "A", "total_score": 0.8, "volatility": 0.2},
            {"ticker": "B", "total_score": 0.5, "volatility": 0.4},
            {"ticker": "C", "total_score": 0.3, "volatility": 0.6},
        ]
        result = PortfolioEngine.allocate(data, method="risk_adjusted",
                                          risk_profile="LOW")
        self.assertAlmostEqual(_sum_alloc(result), 1.0, places=3)

    def test_missing_volatility_defaults_to_zero(self):
        """Stocks without 'volatility' key should not crash."""
        data = [{"ticker": "A", "total_score": 0.8},
                {"ticker": "B", "total_score": 0.4}]
        result = PortfolioEngine.allocate(data, method="risk_adjusted",
                                          risk_profile="LOW")
        self.assertAlmostEqual(_sum_alloc(result), 1.0, places=4)


# ===========================================================================
# 5. Constraint Application
# ===========================================================================

class TestConstraints(unittest.TestCase):

    def test_max_cap_respected(self):
        data = [{"ticker": "A", "total_score": 0.9},
                {"ticker": "B", "total_score": 0.1}]
        result = PortfolioEngine.allocate(data, config={"max_cap": 0.60})
        for r in result:
            self.assertLessEqual(r["allocation"], 0.60 + 1e-4)

    def test_min_floor_respected(self):
        data = [{"ticker": "A", "total_score": 0.95},
                {"ticker": "B", "total_score": 0.05}]
        result = PortfolioEngine.allocate(data, config={"min_floor": 0.10})
        for r in result:
            self.assertGreaterEqual(r["allocation"], 0.10 - 1e-4)

    def test_allocations_still_sum_to_one_after_cap(self):
        data = [{"ticker": t, "total_score": s}
                for t, s in [("A", 0.90), ("B", 0.05), ("C", 0.05)]]
        result = PortfolioEngine.allocate(data, config={"max_cap": 0.50})
        self.assertAlmostEqual(_sum_alloc(result), 1.0, places=3)

    def test_no_config_applies_no_constraints(self):
        # No cap → dominant stock can exceed 0.60
        data = [{"ticker": "A", "total_score": 1.0},
                {"ticker": "B", "total_score": 0.01}]
        result = PortfolioEngine.allocate(data, config=None)
        top = next(r["allocation"] for r in result if r["ticker"] == "A")
        self.assertGreater(top, 0.60)


# ===========================================================================
# 6. Normalise Helper
# ===========================================================================

class TestNormalize(unittest.TestCase):

    def test_normalise_basic(self):
        result = PortfolioEngine._normalize([1.0, 3.0])
        self.assertAlmostEqual(result[0], 0.25)
        self.assertAlmostEqual(result[1], 0.75)

    def test_normalise_all_zeros(self):
        result = PortfolioEngine._normalize([0.0, 0.0, 0.0])
        for v in result:
            self.assertAlmostEqual(v, 1 / 3, places=6)

    def test_normalise_single(self):
        self.assertEqual(PortfolioEngine._normalize([5.0]), [1.0])


# ===========================================================================
# 7. Allocation rounded to 4 decimal places
# ===========================================================================

class TestOutputFormat(unittest.TestCase):

    def test_allocation_rounded_to_4dp(self):
        data = [{"ticker": "A", "total_score": 2},
                {"ticker": "B", "total_score": 1}]
        result = PortfolioEngine.allocate(data)
        for r in result:
            # Must be expressible with ≤ 4 decimal digits
            self.assertEqual(r["allocation"], round(r["allocation"], 4))

    def test_proportional_classic_example(self):
        """score 2:1 → allocations ≈ 0.6667 and 0.3333"""
        data = [{"ticker": "A", "total_score": 2},
                {"ticker": "B", "total_score": 1}]
        result = PortfolioEngine.allocate(data)
        by_ticker = {r["ticker"]: r["allocation"] for r in result}
        self.assertAlmostEqual(by_ticker["A"], 2 / 3, places=4)
        self.assertAlmostEqual(by_ticker["B"], 1 / 3, places=4)


# ===========================================================================
# 8. Risk Decomposition
# ===========================================================================

class TestRiskDecomposition(unittest.TestCase):

    def test_adds_risk_keys(self):
        allocs = [{"ticker": "A", "allocation": 0.5, "volatility": 0.2}]
        result = PortfolioEngine.compute_risk_decomposition(allocs)
        self.assertIn("risk_contribution", result[0])
        self.assertIn("risk_share", result[0])

    def test_risk_share_sums_to_one(self):
        allocs = [
            {"ticker": "A", "allocation": 0.5, "volatility": 0.2},
            {"ticker": "B", "allocation": 0.5, "volatility": 0.4},
        ]
        result = PortfolioEngine.compute_risk_decomposition(allocs)
        total_share = sum(r["risk_share"] for r in result)
        self.assertAlmostEqual(total_share, 1.0)

    def test_higher_vol_higher_share_given_equal_allocation(self):
        allocs = [
            {"ticker": "A", "allocation": 0.5, "volatility": 0.2},
            {"ticker": "B", "allocation": 0.5, "volatility": 0.4},
        ]
        result = PortfolioEngine.compute_risk_decomposition(allocs)
        self.assertGreater(result[1]["risk_share"], result[0]["risk_share"])

    def test_handles_zero_total_risk(self):
        allocs = [
            {"ticker": "A", "allocation": 0.5, "volatility": 0.0},
            {"ticker": "B", "allocation": 0.5},
        ]
        result = PortfolioEngine.compute_risk_decomposition(allocs)
        self.assertNotIn("risk_share", result[0])


# ===========================================================================
# 9. Capital Allocation
# ===========================================================================

class TestCapitalAllocation(unittest.TestCase):

    def test_allocates_capital_correctly(self):
        allocs = [
            {"ticker": "A", "allocation": 0.41},
            {"ticker": "B", "allocation": 0.59},
        ]
        result = PortfolioEngine.allocate_capital(allocs, total_capital=100000)
        self.assertEqual(result[0]["capital_amount"], 41000.0)
        self.assertEqual(result[1]["capital_amount"], 59000.0)

    def test_handles_zero_capital(self):
        allocs = [{"ticker": "A", "allocation": 1.0}]
        result = PortfolioEngine.allocate_capital(allocs, total_capital=0)
        self.assertEqual(result[0]["capital_amount"], 0.0)


# ===========================================================================
# PHASE 1 — Mathematical Integrity (must NEVER fail)
# ===========================================================================

class TestMathematicalIntegrity(unittest.TestCase):
    """Invariant tests — any failure here means core math is broken."""

    def test_allocation_sums_to_one(self):
        """Proportional allocation: sum of weights must equal 1.0 within 1e-9."""
        data = [
            {"ticker": "A", "total_score": 0.8},
            {"ticker": "B", "total_score": 0.2},
        ]
        allocations = PortfolioEngine.allocate(data, method="proportional")
        total = sum(a["allocation"] for a in allocations)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_allocation_sums_to_one_softmax(self):
        """Softmax allocation must also sum to 1.0 within 1e-9."""
        data = [
            {"ticker": "A", "total_score": 0.9},
            {"ticker": "B", "total_score": 0.5},
            {"ticker": "C", "total_score": 0.1},
        ]
        allocations = PortfolioEngine.allocate(data, method="softmax")
        total = sum(a["allocation"] for a in allocations)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_allocation_sums_to_one_risk_adjusted(self):
        """Risk-adjusted allocation must sum to 1.0 within 1e-9."""
        data = [
            {"ticker": "A", "total_score": 0.7, "volatility": 0.15},
            {"ticker": "B", "total_score": 0.3, "volatility": 0.40},
        ]
        allocations = PortfolioEngine.allocate(
            data, method="risk_adjusted", risk_profile="LOW"
        )
        total = sum(a["allocation"] for a in allocations)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_risk_shares_sum_to_one(self):
        """Risk decomposition: sum of risk_share must equal 1.0 within 1e-9."""
        allocations = [
            {"ticker": "A", "allocation": 0.6, "volatility": 0.2},
            {"ticker": "B", "allocation": 0.4, "volatility": 0.1},
        ]
        result = PortfolioEngine.compute_risk_decomposition(allocations)
        total = sum(a["risk_share"] for a in result)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_capital_distribution_matches_budget(self):
        """Capital amounts must sum to exactly the budget (remainder absorption)."""
        allocations = [
            {"ticker": "A", "allocation": 0.5},
            {"ticker": "B", "allocation": 0.5},
        ]
        result = PortfolioEngine.allocate_capital(allocations, 100_000)
        total_capital = sum(a["capital_amount"] for a in result)
        self.assertAlmostEqual(total_capital, 100_000, delta=1e-6)

    def test_capital_distribution_uneven_split(self):
        """Non-trivial allocation ratios must still sum exactly to budget."""
        allocations = [
            {"ticker": "A", "allocation": 1/3},
            {"ticker": "B", "allocation": 1/3},
            {"ticker": "C", "allocation": 1/3},
        ]
        result = PortfolioEngine.allocate_capital(allocations, 100_000)
        total_capital = sum(a["capital_amount"] for a in result)
        # 1e-2 tolerance: last element absorbs ±1¢ rounding residual
        self.assertAlmostEqual(total_capital, 100_000, delta=1e-2)


# ===========================================================================
# PHASE 3 — Edge Case Stress
# ===========================================================================

class TestEdgeCaseStress(unittest.TestCase):

    def test_equal_scores_equal_allocation(self):
        """Two stocks with identical scores must receive equal allocations."""
        allocations = PortfolioEngine.allocate(
            [
                {"ticker": "A", "total_score": 1.0},
                {"ticker": "B", "total_score": 1.0},
            ]
        )
        self.assertEqual(allocations[0]["allocation"], allocations[1]["allocation"])

    def test_single_stock_allocation_is_one(self):
        """A single-stock universe must always get 100% allocation."""
        allocations = PortfolioEngine.allocate(
            [{"ticker": "A", "total_score": 1.0}]
        )
        self.assertEqual(allocations[0]["allocation"], 1.0)

    def test_zero_volatility_risk_decomposition_sums_to_one(self):
        """
        When one stock has zero volatility, the remaining non-zero vol stock
        must carry 100% of risk_share — still sums to 1.0.
        """
        allocations = [
            {"ticker": "A", "allocation": 0.5, "volatility": 0.0},
            {"ticker": "B", "allocation": 0.5, "volatility": 0.2},
        ]
        result = PortfolioEngine.compute_risk_decomposition(allocations)
        total = sum(a["risk_share"] for a in result)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_all_zero_volatility_does_not_crash(self):
        """
        All-zero volatility means total_risk=0; engine must return without
        adding risk_share keys (existing fallback path).
        """
        allocations = [
            {"ticker": "A", "allocation": 0.5, "volatility": 0.0},
            {"ticker": "B", "allocation": 0.5, "volatility": 0.0},
        ]
        result = PortfolioEngine.compute_risk_decomposition(allocations)
        # Should not raise and should return the list unmodified
        self.assertEqual(len(result), 2)
        self.assertNotIn("risk_share", result[0])

    def test_many_equal_scores_sum_to_one(self):
        """10 stocks with equal scores — allocation sum stays exactly 1."""
        data = [{"ticker": f"T{i}", "total_score": 1.0} for i in range(10)]
        allocations = PortfolioEngine.allocate(data)
        total = sum(a["allocation"] for a in allocations)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)


# ===========================================================================
# PHASE 4 — Extreme Risk Profiles
# ===========================================================================

class TestExtremeProfiles(unittest.TestCase):
    """
    Verify that risk_adjusted method changes allocations in the
    expected direction for CONSERVATIVE and AGGRESSIVE profiles.
    """

    def _allocate_ra(self, stocks, profile):
        data = [
            {"ticker": t, "total_score": score, "volatility": vol}
            for t, score, vol in stocks
        ]
        result = PortfolioEngine.allocate(
            data, method="risk_adjusted", risk_profile=profile
        )
        return {r["ticker"]: r["allocation"] for r in result}

    def test_conservative_reduces_high_volatility_weight(self):
        """
        CONSERVATIVE (LOW) profile: the low-volatility stock must receive
        a strictly larger allocation than the high-volatility stock,
        even when both have the same raw score.
        """
        result = self._allocate_ra(
            [("STABLE", 0.7, 0.05), ("RISKY", 0.7, 0.60)],
            "LOW",
        )
        self.assertGreater(result["STABLE"], result["RISKY"])

    def test_aggressive_increases_high_volatility_weight(self):
        """
        AGGRESSIVE (HIGH) profile: the high-volatility stock must receive
        a strictly larger allocation — growth potential is rewarded.
        """
        result = self._allocate_ra(
            [("STABLE", 0.7, 0.05), ("RISKY", 0.7, 0.60)],
            "HIGH",
        )
        self.assertGreater(result["RISKY"], result["STABLE"])

    def test_conservative_allocation_still_sums_to_one(self):
        data = [
            {"ticker": "A", "total_score": 0.8, "volatility": 0.10},
            {"ticker": "B", "total_score": 0.6, "volatility": 0.30},
            {"ticker": "C", "total_score": 0.4, "volatility": 0.55},
        ]
        result = PortfolioEngine.allocate(
            data, method="risk_adjusted", risk_profile="LOW"
        )
        total = sum(r["allocation"] for r in result)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_aggressive_allocation_still_sums_to_one(self):
        data = [
            {"ticker": "A", "total_score": 0.8, "volatility": 0.10},
            {"ticker": "B", "total_score": 0.6, "volatility": 0.30},
            {"ticker": "C", "total_score": 0.4, "volatility": 0.55},
        ]
        result = PortfolioEngine.allocate(
            data, method="risk_adjusted", risk_profile="HIGH"
        )
        total = sum(r["allocation"] for r in result)
        self.assertAlmostEqual(total, 1.0, delta=1e-9)


# ===========================================================================
# PHASE 4 (Extreme Budget) — Large Capital Scale Test
# ===========================================================================

class TestLargeBudget(unittest.TestCase):
    """Ensure capital distribution survives large budgets without precision collapse."""

    def test_large_budget_distribution(self):
        """₹10 billion budget must be distributed exactly — no rounding overflow."""
        allocations = [
            {"ticker": "A", "allocation": 0.33},
            {"ticker": "B", "allocation": 0.33},
            {"ticker": "C", "allocation": 0.34},
        ]
        result = PortfolioEngine.allocate_capital(allocations, 10_000_000_000)
        total = sum(a["capital_amount"] for a in result)
        self.assertEqual(total, 10_000_000_000)

    def test_very_large_budget_uneven_allocation(self):
        """Uneven 7-stock split over ₹1 crore: capital still sums exactly."""
        # Weights deliberately non-trivial
        weights = [0.25, 0.20, 0.18, 0.15, 0.10, 0.07, 0.05]
        allocations = [
            {"ticker": f"T{i}", "allocation": w}
            for i, w in enumerate(weights)
        ]
        result = PortfolioEngine.allocate_capital(allocations, 10_000_000)
        total = sum(a["capital_amount"] for a in result)
        # Last element absorbs ±1¢ residual; total must be within 1¢ of budget
        self.assertAlmostEqual(total, 10_000_000, delta=0.01)



if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# Portfolio-Level Sharpe
# ===========================================================================

class TestPortfolioSharpe(unittest.TestCase):
    """Tests for PortfolioEngine.compute_portfolio_sharpe() and portfolio_summary()."""

    def _allocs(self, specs: list) -> list:
        """
        Build allocation dicts from (allocation, cagr, volatility) tuples.
        Ticker names are auto-assigned A, B, C ...
        """
        return [
            {
                "ticker": chr(65 + i),
                "allocation": w,
                "total_score": 1.0,
                "cagr": c,
                "volatility": v,
            }
            for i, (w, c, v) in enumerate(specs)
        ]

    # ------------------------------------------------------------------ #
    #  Basic correctness
    # ------------------------------------------------------------------ #

    def test_portfolio_sharpe_positive(self):
        """Two stocks with CAGR well above risk-free should yield positive Sharpe."""
        allocs = self._allocs([
            (0.5, 0.20, 0.20),   # 20% CAGR, 20% vol
            (0.5, 0.15, 0.10),   # 15% CAGR, 10% vol
        ])
        sharpe = PortfolioEngine.compute_portfolio_sharpe(allocs)
        self.assertGreater(sharpe, 0.0)

    def test_portfolio_sharpe_zero_volatility_returns_zero(self):
        """Zero volatility → no division; must return 0.0 without crashing."""
        allocs = self._allocs([(1.0, 0.10, 0.0)])
        self.assertEqual(PortfolioEngine.compute_portfolio_sharpe(allocs), 0.0)

    def test_portfolio_sharpe_returns_zero_on_empty(self):
        """Empty list must return 0.0, not crash."""
        self.assertEqual(PortfolioEngine.compute_portfolio_sharpe([]), 0.0)

    def test_portfolio_sharpe_invalid_allocation_raises(self):
        """Allocations not summing to 1.0 must raise AssertionError."""
        allocs = self._allocs([
            (0.6, 0.10, 0.20),
            (0.6, 0.10, 0.20),   # sum = 1.2 → invalid
        ])
        with self.assertRaises(AssertionError):
            PortfolioEngine.compute_portfolio_sharpe(allocs)

    def test_portfolio_sharpe_below_risk_free_is_negative(self):
        """CAGR below RISK_FREE_RATE → Sharpe must be negative."""
        from chatbot.config import RISK_FREE_RATE
        low_cagr = RISK_FREE_RATE * 0.5   # always below Rf
        allocs = self._allocs([(1.0, low_cagr, 0.15)])
        sharpe = PortfolioEngine.compute_portfolio_sharpe(allocs)
        self.assertLess(sharpe, 0.0)

    # ------------------------------------------------------------------ #
    #  Formula verification
    # ------------------------------------------------------------------ #

    def test_portfolio_sharpe_single_stock_matches_formula(self):
        """
        For a single stock the formula collapses to:
            Sharpe = (CAGR - Rf) / vol
        which is identical to stock-level Sharpe (no clipping here).
        """
        from chatbot.config import RISK_FREE_RATE
        cagr, vol = 0.18, 0.15
        allocs = self._allocs([(1.0, cagr, vol)])
        expected = (cagr - RISK_FREE_RATE) / vol
        self.assertAlmostEqual(
            PortfolioEngine.compute_portfolio_sharpe(allocs),
            expected,
            places=10,
        )

    def test_portfolio_sharpe_two_stock_formula(self):
        """Verify the weighted-return and independent-vol formula result by hand."""
        import math
        from chatbot.config import RISK_FREE_RATE
        w1, w2         = 0.6, 0.4
        cagr1, cagr2   = 0.20, 0.10
        vol1, vol2     = 0.25, 0.15

        expected_return = w1 * cagr1 + w2 * cagr2
        expected_vol    = math.sqrt(w1**2 * vol1**2 + w2**2 * vol2**2)
        expected_sharpe = (expected_return - RISK_FREE_RATE) / expected_vol

        allocs = self._allocs([(w1, cagr1, vol1), (w2, cagr2, vol2)])
        self.assertAlmostEqual(
            PortfolioEngine.compute_portfolio_sharpe(allocs),
            expected_sharpe,
            places=10,
        )

    # ------------------------------------------------------------------ #
    #  portfolio_summary() output dict
    # ------------------------------------------------------------------ #

    def test_portfolio_summary_returns_all_keys(self):
        """portfolio_summary() must return all four expected keys."""
        allocs = self._allocs([(0.5, 0.15, 0.2), (0.5, 0.12, 0.1)])
        summary = PortfolioEngine.portfolio_summary(allocs)
        for key in ("portfolio_return", "portfolio_volatility",
                    "portfolio_sharpe", "allocations"):
            self.assertIn(key, summary)

    def test_portfolio_summary_return_matches_formula(self):
        """portfolio_return must equal the weighted sum of CAGRs."""
        allocs = self._allocs([(0.7, 0.20, 0.15), (0.3, 0.10, 0.10)])
        summary = PortfolioEngine.portfolio_summary(allocs)
        expected_return = 0.7 * 0.20 + 0.3 * 0.10
        self.assertAlmostEqual(summary["portfolio_return"], expected_return, places=10)

    def test_portfolio_summary_sharpe_consistent_with_method(self):
        """portfolio_summary Sharpe must equal compute_portfolio_sharpe() exactly."""
        allocs = self._allocs([(0.6, 0.18, 0.22), (0.4, 0.12, 0.14)])
        summary = PortfolioEngine.portfolio_summary(allocs)
        self.assertAlmostEqual(
            summary["portfolio_sharpe"],
            PortfolioEngine.compute_portfolio_sharpe(allocs),
            places=12,
        )

    def test_portfolio_summary_empty_returns_zeros(self):
        """Empty allocations must return zero-valued summary, no crash."""
        summary = PortfolioEngine.portfolio_summary([])
        self.assertEqual(summary["portfolio_return"],     0.0)
        self.assertEqual(summary["portfolio_volatility"], 0.0)
        self.assertEqual(summary["portfolio_sharpe"],     0.0)

    def test_portfolio_summary_includes_mdd_and_sortino_keys(self):
        """portfolio_summary() must include portfolio_mdd and portfolio_sortino keys."""
        allocs = self._allocs([(0.5, 0.15, 0.2), (0.5, 0.12, 0.1)])
        summary = PortfolioEngine.portfolio_summary(allocs)
        self.assertIn("portfolio_mdd",     summary)
        self.assertIn("portfolio_sortino", summary)


# ===========================================================================
# Portfolio-Level MDD and Sortino
# ===========================================================================

class TestPortfolioMDDSortino(unittest.TestCase):
    """Tests for compute_portfolio_mdd() and compute_portfolio_sortino()."""

    def _allocs_with_returns(self, specs: list, n: int = 756) -> list:
        """
        Build allocation dicts that include ``daily_returns`` pd.Series.
        specs: list of (allocation, cagr, volatility, daily_growth)
        """
        import pandas as pd, numpy as np
        result = []
        for i, (w, c, v, g) in enumerate(specs):
            prices  = [1000 * ((1 + g) ** d) for d in range(n)]
            returns = pd.Series(
                [(prices[d + 1] - prices[d]) / prices[d] for d in range(n - 1)]
            )
            result.append({
                "ticker":        chr(65 + i),
                "allocation":    w,
                "total_score":   1.0,
                "cagr":          c,
                "volatility":    v,
                "daily_returns": returns,
            })
        return result

    def _allocs_simple(self, specs):
        """Allocation dicts without daily_returns — tests fallback paths."""
        return [
            {"ticker": chr(65+i), "allocation": w, "total_score": 1.0,
             "cagr": c, "volatility": v, "max_drawdown": mdd}
            for i, (w, c, v, mdd) in enumerate(specs)
        ]

    # ------------------------------------------------------------------ #
    #  compute_portfolio_mdd()
    # ------------------------------------------------------------------ #

    def test_portfolio_mdd_monotonic_portfolio_is_zero(self):
        """
        Strictly increasing portfolio price index → MDD must be 0.0.
        Use a positive daily growth rate for all stocks.
        """
        allocs = self._allocs_with_returns([(0.5, 0.12, 0.0, 0.0005),
                                            (0.5, 0.10, 0.0, 0.0003)])
        mdd = PortfolioEngine.compute_portfolio_mdd(allocs)
        self.assertAlmostEqual(mdd, 0.0, places=9)

    def test_portfolio_mdd_fallback_weighted_average(self):
        """
        Without daily_returns, MDD falls back to weighted average of per-stock MDD.
        Verify the formula: sum(wi * mdd_i).
        """
        allocs = self._allocs_simple([(0.6, 0.12, 0.2, 0.15),
                                      (0.4, 0.10, 0.1, 0.10)])
        expected = 0.6 * 0.15 + 0.4 * 0.10
        mdd = PortfolioEngine.compute_portfolio_mdd(allocs)
        self.assertAlmostEqual(mdd, expected, places=10)

    def test_portfolio_mdd_empty_returns_zero(self):
        self.assertEqual(PortfolioEngine.compute_portfolio_mdd([]), 0.0)

    def test_portfolio_mdd_non_negative(self):
        """MDD must always be >= 0."""
        allocs = self._allocs_with_returns([(0.5, 0.15, 0.2, 0.0006),
                                            (0.5, 0.12, 0.1, 0.0004)])
        mdd = PortfolioEngine.compute_portfolio_mdd(allocs)
        self.assertGreaterEqual(mdd, 0.0)

    def test_portfolio_mdd_bounded_at_one(self):
        """MDD must never exceed 1.0."""
        allocs = self._allocs_with_returns([(1.0, 0.10, 0.2, 0.0003)])
        mdd = PortfolioEngine.compute_portfolio_mdd(allocs)
        self.assertLessEqual(mdd, 1.0)

    # ------------------------------------------------------------------ #
    #  compute_portfolio_sortino()
    # ------------------------------------------------------------------ #

    def test_portfolio_sortino_empty_returns_zero(self):
        self.assertEqual(PortfolioEngine.compute_portfolio_sortino([]), 0.0)

    def test_portfolio_sortino_positive_for_growing_portfolio(self):
        """Positive-return portfolio should have positive Sortino (non-zero)."""
        allocs = self._allocs_with_returns([(0.6, 0.18, 0.2, 0.0007),
                                            (0.4, 0.14, 0.1, 0.0005)])
        sortino = PortfolioEngine.compute_portfolio_sortino(allocs)
        self.assertIsInstance(sortino, float)

    def test_portfolio_sortino_fallback_matches_sharpe(self):
        """
        Without daily_returns, sortino falls back to portfolio Sharpe.
        Verify they are equal in that case.
        """
        allocs = [{"ticker": "A", "allocation": 0.6, "total_score": 1.0,
                   "cagr": 0.15, "volatility": 0.2},
                  {"ticker": "B", "allocation": 0.4, "total_score": 1.0,
                   "cagr": 0.10, "volatility": 0.1}]
        sortino = PortfolioEngine.compute_portfolio_sortino(allocs)
        sharpe  = PortfolioEngine.compute_portfolio_sharpe(allocs)
        self.assertAlmostEqual(sortino, sharpe, places=10)

    # ------------------------------------------------------------------ #
    #  portfolio_summary() with new keys
    # ------------------------------------------------------------------ #

    def test_summary_has_mdd_and_sortino(self):
        allocs = self._allocs_simple([(0.5, 0.12, 0.2, 0.15),
                                      (0.5, 0.10, 0.1, 0.10)])
        summary = PortfolioEngine.portfolio_summary(allocs)
        self.assertIn("portfolio_mdd",     summary)
        self.assertIn("portfolio_sortino", summary)

    def test_summary_mdd_matches_standalone(self):
        allocs = self._allocs_simple([(0.7, 0.15, 0.25, 0.20),
                                      (0.3, 0.10, 0.10, 0.08)])
        summary = PortfolioEngine.portfolio_summary(allocs)
        self.assertAlmostEqual(
            summary["portfolio_mdd"],
            PortfolioEngine.compute_portfolio_mdd(allocs),
            places=12,
        )


# ===========================================================================
# Stage 2 — Covariance-Aware Portfolio Volatility  (σp² = wᵀΣw)
# ===========================================================================

class TestCovarianceVolatility(unittest.TestCase):
    """
    Tests for compute_portfolio_volatility_covariance(),
    compute_portfolio_sharpe_covariance(), and portfolio_summary_covariance().
    """

    def _alloc(self, weight: float, expected_return: float = 0.12) -> dict:
        return {"allocation": weight, "expected_return": expected_return,
                "total_score": 1.0, "ticker": "X"}

    def _allocs(self, specs: list) -> list:
        """(weight, expected_return) tuples → allocation dicts."""
        return [
            {"allocation": w, "expected_return": r,
             "total_score": 1.0, "ticker": chr(65 + i)}
            for i, (w, r) in enumerate(specs)
        ]

    # ------------------------------------------------------------------ #
    #  Volatility — formula correctness
    # ------------------------------------------------------------------ #

    def test_single_asset_formula(self):
        """
        Single asset with variance 0.04 → volatility = sqrt(0.04) = 0.2.
        """
        allocs = [self._alloc(1.0)]
        cov    = [[0.04]]
        vol    = PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov)
        self.assertAlmostEqual(vol, 0.2, places=10)

    def test_two_asset_manual_arithmetic(self):
        """
        w = [0.5, 0.5], cov = [[0.04, 0.02], [0.02, 0.09]]

        σp² = 0.5²×0.04 + 0.5²×0.09 + 2×0.5×0.5×0.02
            = 0.01 + 0.0225 + 0.01
            = 0.0425
        σp  = √0.0425
        """
        import math
        allocs = self._allocs([(0.5, 0.15), (0.5, 0.10)])
        cov    = [[0.04, 0.02], [0.02, 0.09]]
        vol    = PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov)
        expected = math.sqrt(0.5**2 * 0.04 + 0.5**2 * 0.09 + 2 * 0.5 * 0.5 * 0.02)
        self.assertAlmostEqual(vol, expected, places=12)

    def test_zero_off_diagonal_matches_stage1(self):
        """
        When covariance matrix is diagonal, Stage 2 should equal Stage 1
        (independent-stock formula).

        σp² = Σ wᵢ² σᵢ²   (no cross terms)
        """
        import math
        w1, w2 = 0.6, 0.4
        v1, v2 = 0.20, 0.15   # vols
        allocs = self._allocs([(w1, 0.18), (w2, 0.12)])
        # Attach volatility for Stage 1 reference
        allocs[0]["volatility"] = v1
        allocs[1]["volatility"] = v2

        cov_diag = [[v1**2, 0.0], [0.0, v2**2]]
        vol_stage2 = PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov_diag)
        vol_stage1 = math.sqrt(w1**2 * v1**2 + w2**2 * v2**2)
        self.assertAlmostEqual(vol_stage2, vol_stage1, places=12)

    def test_perfect_correlation_approaches_weighted_sum_of_vols(self):
        """
        When cov[i][j] = σi × σj (ρ=1), portfolio volatility equals Σ wᵢ σᵢ.

        This is the upper bound on diversification — no benefit from mixing.
        """
        v1, v2 = 0.25, 0.15
        w1, w2 = 0.6, 0.4
        allocs  = self._allocs([(w1, 0.20), (w2, 0.10)])
        cov_rho1 = [[v1 * v1, v1 * v2],
                    [v2 * v1, v2 * v2]]
        vol = PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov_rho1)
        expected = w1 * v1 + w2 * v2
        self.assertAlmostEqual(vol, expected, places=10)

    def test_positive_correlation_higher_than_independent(self):
        """
        Positive off-diagonal covariance increases portfolio volatility
        compared to the zero-correlation (Stage 1) case.
        """
        import math
        w1, w2 = 0.5, 0.5
        v1, v2 = 0.20, 0.20

        # Stage 1: independent (diagonal only)
        vol_stage1 = math.sqrt(w1**2 * v1**2 + w2**2 * v2**2)

        # Stage 2: with positive correlation
        cov_positive = [[v1**2, 0.02], [0.02, v2**2]]
        allocs = self._allocs([(w1, 0.15), (w2, 0.15)])
        vol_stage2 = PortfolioEngine.compute_portfolio_volatility_covariance(
            allocs, cov_positive
        )
        self.assertGreater(vol_stage2, vol_stage1)

    def test_negative_correlation_lower_than_independent(self):
        """
        Negative off-diagonal covariance reduces portfolio volatility —
        the key benefit of diversification.
        """
        import math
        w1, w2 = 0.5, 0.5
        v1, v2 = 0.20, 0.20

        vol_stage1 = math.sqrt(w1**2 * v1**2 + w2**2 * v2**2)

        cov_negative = [[v1**2, -0.02], [-0.02, v2**2]]
        allocs = self._allocs([(w1, 0.15), (w2, 0.15)])
        vol_stage2 = PortfolioEngine.compute_portfolio_volatility_covariance(
            allocs, cov_negative
        )
        self.assertLess(vol_stage2, vol_stage1)

    # ------------------------------------------------------------------ #
    #  Guard conditions
    # ------------------------------------------------------------------ #

    def test_empty_allocations_returns_zero(self):
        vol = PortfolioEngine.compute_portfolio_volatility_covariance([], [])
        self.assertEqual(vol, 0.0)

    def test_dimension_mismatch_raises(self):
        """Row count mismatch between allocations and matrix → AssertionError."""
        allocs = self._allocs([(0.5, 0.12), (0.5, 0.10)])
        cov_wrong = [[0.04]]   # 1×1 for 2 allocations
        with self.assertRaises(AssertionError):
            PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov_wrong)

    def test_non_square_matrix_raises(self):
        """Non-square row in matrix → AssertionError."""
        allocs = self._allocs([(0.5, 0.12), (0.5, 0.10)])
        cov_bad = [[0.04, 0.01, 0.99], [0.01, 0.09]]  # row 0 has 3 cols
        with self.assertRaises(AssertionError):
            PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov_bad)

    def test_bad_weights_raises(self):
        """Weights not summing to 1.0 → AssertionError."""
        allocs = self._allocs([(0.6, 0.12), (0.6, 0.10)])   # sum = 1.2
        cov = [[0.04, 0.01], [0.01, 0.09]]
        with self.assertRaises(AssertionError):
            PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov)

    def test_zero_variance_matrix_returns_zero(self):
        """All-zero covariance matrix → vol = 0.0 without crashing."""
        allocs = self._allocs([(0.5, 0.10), (0.5, 0.08)])
        cov_zero = [[0.0, 0.0], [0.0, 0.0]]
        vol = PortfolioEngine.compute_portfolio_volatility_covariance(allocs, cov_zero)
        self.assertEqual(vol, 0.0)

    # ------------------------------------------------------------------ #
    #  Sharpe — covariance-aware
    # ------------------------------------------------------------------ #

    def test_sharpe_zero_volatility_returns_zero(self):
        allocs = self._allocs([(0.5, 0.10), (0.5, 0.08)])
        cov_zero = [[0.0, 0.0], [0.0, 0.0]]
        sharpe = PortfolioEngine.compute_portfolio_sharpe_covariance(allocs, cov_zero)
        self.assertEqual(sharpe, 0.0)

    def test_sharpe_positive_excess_return(self):
        """Expected return > risk-free with nonzero vol → positive Sharpe."""
        from chatbot.config import RISK_FREE_RATE
        r = RISK_FREE_RATE + 0.08   # comfortably above Rf
        allocs = self._allocs([(0.5, r), (0.5, r)])
        cov    = [[0.04, 0.01], [0.01, 0.04]]
        sharpe = PortfolioEngine.compute_portfolio_sharpe_covariance(allocs, cov)
        self.assertGreater(sharpe, 0.0)

    def test_sharpe_empty_returns_zero(self):
        sharpe = PortfolioEngine.compute_portfolio_sharpe_covariance([], [])
        self.assertEqual(sharpe, 0.0)

    # ------------------------------------------------------------------ #
    #  portfolio_summary_covariance()
    # ------------------------------------------------------------------ #

    def test_summary_has_correct_keys(self):
        allocs = self._allocs([(0.5, 0.15), (0.5, 0.10)])
        cov    = [[0.04, 0.01], [0.01, 0.09]]
        summary = PortfolioEngine.portfolio_summary_covariance(allocs, cov)
        for key in ("portfolio_return", "portfolio_volatility",
                    "portfolio_sharpe", "model", "allocations"):
            self.assertIn(key, summary)

    def test_summary_model_label_is_covariance_aware(self):
        allocs = self._allocs([(0.5, 0.15), (0.5, 0.10)])
        cov    = [[0.04, 0.01], [0.01, 0.09]]
        summary = PortfolioEngine.portfolio_summary_covariance(allocs, cov)
        self.assertEqual(summary["model"], "covariance-aware")

    def test_summary_sharpe_matches_standalone(self):
        """Summary Sharpe must equal compute_portfolio_sharpe_covariance() exactly."""
        allocs = self._allocs([(0.6, 0.18), (0.4, 0.12)])
        cov    = [[0.04, 0.015], [0.015, 0.09]]
        summary = PortfolioEngine.portfolio_summary_covariance(allocs, cov)
        standalone = PortfolioEngine.compute_portfolio_sharpe_covariance(allocs, cov)
        self.assertAlmostEqual(summary["portfolio_sharpe"], standalone, places=12)

    def test_summary_empty_returns_zero_dict_with_model_label(self):
        summary = PortfolioEngine.portfolio_summary_covariance([], [])
        self.assertEqual(summary["portfolio_return"],     0.0)
        self.assertEqual(summary["portfolio_volatility"], 0.0)
        self.assertEqual(summary["portfolio_sharpe"],     0.0)
        self.assertEqual(summary["model"],                "covariance-aware")

    def test_summary_return_matches_weighted_expected_return(self):
        allocs = self._allocs([(0.7, 0.20), (0.3, 0.10)])
        cov    = [[0.04, 0.0], [0.0, 0.09]]
        summary = PortfolioEngine.portfolio_summary_covariance(allocs, cov)
        expected_return = 0.7 * 0.20 + 0.3 * 0.10
        self.assertAlmostEqual(summary["portfolio_return"], expected_return, places=12)


# ===========================================================================
# Stage 4 — Monte Carlo Simulation
# ===========================================================================

class TestMonteCarlo(unittest.TestCase):
    """
    Tests for PortfolioEngine._cholesky_decomposition(),
    PortfolioEngine.simulate_portfolio_monte_carlo(), and
    AllocationExplanationEngine._monte_carlo_section().
    """

    def _allocs(self, specs: list) -> list:
        """(weight, expected_return) → allocation dicts."""
        return [
            {"allocation": w, "expected_return": r,
             "total_score": 1.0, "ticker": chr(65 + i)}
            for i, (w, r) in enumerate(specs)
        ]

    def _run(self, specs, cov, n=1000, seed=42):
        """Helper: run simulation with fixed seed."""
        return PortfolioEngine.simulate_portfolio_monte_carlo(
            self._allocs(specs), cov, num_simulations=n, seed=seed
        )

    # ------------------------------------------------------------------ #
    #  Cholesky decomposition
    # ------------------------------------------------------------------ #

    def test_cholesky_identity_matrix(self):
        """Cholesky of I should return I (lower triangle)."""
        L = PortfolioEngine._cholesky_decomposition([[1, 0], [0, 1]])
        self.assertAlmostEqual(L[0][0], 1.0)
        self.assertAlmostEqual(L[1][1], 1.0)
        self.assertAlmostEqual(L[1][0], 0.0)

    def test_cholesky_2x2_correctness(self):
        """
        For cov = [[4, 2], [2, 3]], verify L satisfies L @ Lᵀ == cov.
        """
        import math
        cov = [[4.0, 2.0], [2.0, 3.0]]
        L   = PortfolioEngine._cholesky_decomposition(cov)
        # Reconstruct L @ Lᵀ
        n = len(cov)
        reconstructed = [[sum(L[i][k] * L[j][k] for k in range(n))
                          for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(reconstructed[i][j], cov[i][j], places=10)

    def test_cholesky_non_positive_definite_raises(self):
        """Singular or non-PSD matrix must raise ValueError."""
        with self.assertRaises(ValueError):
            PortfolioEngine._cholesky_decomposition([[0.0, 0.0], [0.0, 1.0]])

    # ------------------------------------------------------------------ #
    #  Simulation — output structure
    # ------------------------------------------------------------------ #

    def test_returns_all_keys(self):
        """simulate_portfolio_monte_carlo() must return all six expected keys."""
        result = self._run([(1.0, 0.10)], [[0.04]])
        for key in ("mean_return", "std_dev", "var_95", "cvar_95",
                    "probability_of_loss", "simulated_returns"):
            self.assertIn(key, result)

    def test_simulated_returns_length(self):
        """simulated_returns list must have exactly num_simulations entries."""
        result = self._run([(1.0, 0.10)], [[0.04]], n=500)
        self.assertEqual(len(result["simulated_returns"]), 500)

    def test_empty_allocations_returns_empty_dict(self):
        result = PortfolioEngine.simulate_portfolio_monte_carlo([], [])
        self.assertEqual(result, {})

    def test_bad_weights_raise(self):
        """Weights summing to != 1.0 must raise AssertionError."""
        allocs = self._allocs([(0.6, 0.10), (0.6, 0.12)])  # sum = 1.2
        with self.assertRaises(AssertionError):
            PortfolioEngine.simulate_portfolio_monte_carlo(allocs, [[0.04, 0], [0, 0.09]])

    # ------------------------------------------------------------------ #
    #  Seed reproducibility
    # ------------------------------------------------------------------ #

    def test_same_seed_identical_results(self):
        """Same random seed must produce bitwise-identical results."""
        r1 = self._run([(0.5, 0.15), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]], seed=7)
        r2 = self._run([(0.5, 0.15), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]], seed=7)
        self.assertEqual(r1["simulated_returns"], r2["simulated_returns"])
        self.assertEqual(r1["mean_return"],       r2["mean_return"])

    def test_different_seeds_different_results(self):
        """Different seeds should (almost certainly) produce different results."""
        r1 = self._run([(0.5, 0.15), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]], seed=1)
        r2 = self._run([(0.5, 0.15), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]], seed=2)
        self.assertNotEqual(r1["simulated_returns"], r2["simulated_returns"])

    # ------------------------------------------------------------------ #
    #  Statistical correctness
    # ------------------------------------------------------------------ #

    def test_single_asset_std_matches_sqrt_variance(self):
        """
        Single asset simulation: simulated std_dev should converge to
        sqrt(cov[0][0]) for large N.  Tested to within 5% tolerance.
        """
        vol  = 0.20
        cov  = [[vol**2]]
        result = PortfolioEngine.simulate_portfolio_monte_carlo(
            self._allocs([(1.0, 0.0)]), cov, num_simulations=50_000, seed=42
        )
        self.assertAlmostEqual(result["std_dev"], vol, delta=vol * 0.05,
            msg=f"Expected std ≈ {vol}, got {result['std_dev']:.4f}")

    def test_single_asset_mean_return_converges(self):
        """Mean return should converge to the specified expected return."""
        mu   = 0.15
        result = PortfolioEngine.simulate_portfolio_monte_carlo(
            self._allocs([(1.0, mu)]), [[0.04]], num_simulations=50_000, seed=99
        )
        self.assertAlmostEqual(result["mean_return"], mu, delta=0.01,
            msg=f"Expected mean ≈ {mu}, got {result['mean_return']:.4f}")

    def test_zero_covariance_std_matches_independent_formula(self):
        """
        Diagonal covariance (zero correlation) → simulated std should converge
        to sqrt(Σ wᵢ² σᵢ²) within 5%.
        """
        import math
        w1, w2  = 0.6, 0.4
        v1, v2  = 0.20, 0.15
        expected_vol = math.sqrt(w1**2 * v1**2 + w2**2 * v2**2)
        result = PortfolioEngine.simulate_portfolio_monte_carlo(
            self._allocs([(w1, 0.0), (w2, 0.0)]),
            [[v1**2, 0.0], [0.0, v2**2]],
            num_simulations=50_000, seed=42,
        )
        self.assertAlmostEqual(result["std_dev"], expected_vol,
            delta=expected_vol * 0.05,
            msg=f"Expected vol ≈ {expected_vol:.4f}, got {result['std_dev']:.4f}")

    def test_positive_correlation_increases_std(self):
        """
        Positive off-diagonal covariance should inflate simulated std
        relative to the zero-correlation (diagonal) case.
        """
        w1, w2 = 0.5, 0.5
        v1, v2 = 0.20, 0.20
        cov_diag = [[v1**2, 0.0],  [0.0,   v2**2]]
        cov_pos  = [[v1**2, 0.02], [0.02,  v2**2]]
        base = PortfolioEngine.simulate_portfolio_monte_carlo(
            self._allocs([(w1, 0.0), (w2, 0.0)]), cov_diag, 30_000, seed=1)
        corr = PortfolioEngine.simulate_portfolio_monte_carlo(
            self._allocs([(w1, 0.0), (w2, 0.0)]), cov_pos, 30_000, seed=1)
        self.assertGreater(corr["std_dev"], base["std_dev"])

    # ------------------------------------------------------------------ #
    #  Statistics invariants
    # ------------------------------------------------------------------ #

    def test_cvar_le_var(self):
        """CVaR (average of worst 5%) must be <= VaR (5th percentile cutoff)."""
        result = self._run([(0.5, 0.12), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]])
        self.assertLessEqual(result["cvar_95"], result["var_95"])

    def test_var_le_mean(self):
        """5th-percentile VaR must be below the mean for a dispersed distribution."""
        result = self._run([(0.5, 0.12), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]])
        self.assertLessEqual(result["var_95"], result["mean_return"])

    def test_probability_of_loss_bounded(self):
        """Probability of loss must be in [0, 1]."""
        result = self._run([(0.5, 0.12), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]])
        self.assertGreaterEqual(result["probability_of_loss"], 0.0)
        self.assertLessEqual(result["probability_of_loss"],    1.0)

    def test_high_return_low_probability_of_loss(self):
        """Very high expected return (30%) → near-zero probability of loss."""
        result = PortfolioEngine.simulate_portfolio_monte_carlo(
            self._allocs([(1.0, 0.30)]), [[0.001]], 10_000, seed=42
        )
        self.assertLess(result["probability_of_loss"], 0.01)

    # ------------------------------------------------------------------ #
    #  Explanation section
    # ------------------------------------------------------------------ #

    def test_monte_carlo_section_renders_string(self):
        """_monte_carlo_section() must return a non-empty string for valid input."""
        from chatbot.allocation_explanation_engine import AllocationExplanationEngine as AEE
        mc = self._run([(0.5, 0.15), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]])
        rendered = AEE._monte_carlo_section(mc)
        self.assertIsInstance(rendered, str)
        self.assertGreater(len(rendered), 0)

    def test_monte_carlo_section_contains_key_metrics(self):
        """Rendered section must include all five metric labels."""
        from chatbot.allocation_explanation_engine import AllocationExplanationEngine as AEE
        mc = self._run([(0.5, 0.15), (0.5, 0.10)], [[0.04, 0.01], [0.01, 0.09]])
        rendered = AEE._monte_carlo_section(mc)
        for label in ("Mean Return", "Volatility", "VaR", "CVaR", "Probability of Loss"):
            self.assertIn(label, rendered)

    def test_monte_carlo_section_empty_input(self):
        """_monte_carlo_section({}) must return empty string."""
        from chatbot.allocation_explanation_engine import AllocationExplanationEngine as AEE
        self.assertEqual(AEE._monte_carlo_section({}), "")
