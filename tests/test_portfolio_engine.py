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


if __name__ == "__main__":
    unittest.main()
