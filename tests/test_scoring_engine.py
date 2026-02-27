"""
tests/test_scoring_engine.py
-----------------------------
Unit tests for ScoringEngine — Sharpe Ratio integration.

Test coverage:
    sharpe weight impacts ranking    — adding sharpe weight changes the winner
    sharpe weight = 0 is backward compatible  — same result as no sharpe key
    sharpe normalisation in scoring  — sharpe is min-max normalised like CAGR
    full validated weights still sum to 1.0
"""

import unittest

from chatbot.metrics_engine import ScoringEngine


# ---------------------------------------------------------------------------
# Helper: two-ticker metrics dicts designed so only Sharpe differs
# ---------------------------------------------------------------------------

def _two_stock_metrics(
    *,
    sharpe_a: float,
    sharpe_b: float,
    cagr: float = 0.12,
    vol: float = 0.20,
    volume: float = 1_000_000,
):
    """
    Build a metrics_dict where A and B are identical on return/risk/volume
    but differ only on 'sharpe'.  This isolates the sharpe contribution.
    """
    return {
        "A": {"cagr": cagr, "volatility": vol, "avg_volume": volume, "latest_price": 500.0, "sharpe": sharpe_a},
        "B": {"cagr": cagr, "volatility": vol, "avg_volume": volume, "latest_price": 500.0, "sharpe": sharpe_b},
    }


# ===========================================================================
# 1. Sharpe weight impacts ranking
# ===========================================================================

class TestSharpeWeightImpactsRanking(unittest.TestCase):

    def test_sharpe_weight_makes_higher_sharpe_win(self):
        """
        A and B are equal on CAGR/vol/volume. A has Sharpe=2.0, B has Sharpe=0.5.
        With a non-zero sharpe weight, A must rank #1.
        """
        metrics = _two_stock_metrics(sharpe_a=2.0, sharpe_b=0.5)
        weights = {
            "return": 0.25,
            "risk":   0.25,
            "volume": 0.25,
            "sharpe": 0.25,
        }
        results = ScoringEngine.compute_weighted_scores(metrics, weights)
        self.assertEqual(results[0]["ticker"], "A")  # A has higher Sharpe

    def test_lower_sharpe_ranks_second(self):
        """Symmetric verification: B must be rank #2 when A dominates on Sharpe."""
        metrics = _two_stock_metrics(sharpe_a=2.0, sharpe_b=0.5)
        weights = {"return": 0.25, "risk": 0.25, "volume": 0.25, "sharpe": 0.25}
        results = ScoringEngine.compute_weighted_scores(metrics, weights)
        self.assertEqual(results[1]["ticker"], "B")

    def test_sharpe_only_weight(self):
        """When only sharpe weight is non-zero, ranking is purely by Sharpe."""
        metrics = _two_stock_metrics(sharpe_a=1.5, sharpe_b=0.3)
        weights = {"return": 0.0, "risk": 0.0, "volume": 0.0, "sharpe": 1.0}
        results = ScoringEngine.compute_weighted_scores(metrics, weights)
        self.assertEqual(results[0]["ticker"], "A")

    def test_sharpe_weighting_does_not_crash_on_equal_sharpe(self):
        """Equal Sharpe values must normalise to 1.0 each — no crash."""
        metrics = _two_stock_metrics(sharpe_a=1.0, sharpe_b=1.0)
        weights = {"return": 0.25, "risk": 0.25, "volume": 0.25, "sharpe": 0.25}
        results = ScoringEngine.compute_weighted_scores(metrics, weights)
        self.assertEqual(len(results), 2)
        # Equal sharpe → equal contribution → equal total scores
        self.assertAlmostEqual(
            results[0]["total_score"],
            results[1]["total_score"],
            places=5,
        )


# ===========================================================================
# 2. Sharpe weight = 0 is backward compatible
# ===========================================================================

class TestSharpeWeightZeroBackwardCompatible(unittest.TestCase):

    def test_zero_sharpe_weight_same_as_no_sharpe_key(self):
        """
        Providing sharpe weight = 0 must produce identical results to not
        providing a sharpe key at all (original 3-criterion behaviour).
        """
        metrics = _two_stock_metrics(sharpe_a=2.0, sharpe_b=0.1)

        weights_without = {"return": 0.34, "risk": 0.33, "volume": 0.33}
        weights_with_zero = {"return": 0.34, "risk": 0.33, "volume": 0.33, "sharpe": 0.0}

        result_without   = ScoringEngine.compute_weighted_scores(metrics, weights_without)
        result_with_zero = ScoringEngine.compute_weighted_scores(metrics, weights_with_zero)

        # Ranking order must be the same
        tickers_without   = [r["ticker"] for r in result_without]
        tickers_with_zero = [r["ticker"] for r in result_with_zero]
        self.assertEqual(tickers_without, tickers_with_zero)

        # Scores must be equal (within floating-point tolerance)
        for r1, r2 in zip(result_without, result_with_zero):
            self.assertAlmostEqual(r1["total_score"], r2["total_score"], places=5)

    def test_default_screener_weights_no_sharpe_still_work(self):
        """
        DEFAULT_SCREENER_WEIGHTS (return/risk/volume only) must still produce
        valid, non-crashing results now that ScoringEngine.CRITERIA includes sharpe.
        """
        from chatbot.constants import DEFAULT_SCREENER_WEIGHTS
        metrics = _two_stock_metrics(sharpe_a=1.0, sharpe_b=0.5)
        results = ScoringEngine.compute_weighted_scores(metrics, DEFAULT_SCREENER_WEIGHTS)
        self.assertEqual(len(results), 2)
        total_score_a = next(r["total_score"] for r in results if r["ticker"] == "A")
        total_score_b = next(r["total_score"] for r in results if r["ticker"] == "B")
        # With no sharpe weight, equal on return/risk/volume → equal scores
        self.assertAlmostEqual(total_score_a, total_score_b, places=5)


# ===========================================================================
# 3. Effective weights always sum to 1.0
# ===========================================================================

class TestSharpeWeightNormalisation(unittest.TestCase):

    def test_weights_including_sharpe_normalise_to_one(self):
        """validate_weights must normalise sharpe-inclusive weights to sum = 1.0."""
        raw = {"return": 0.4, "risk": 0.3, "volume": 0.2, "sharpe": 0.1}
        normalised = ScoringEngine.validate_weights(raw)
        self.assertAlmostEqual(sum(normalised.values()), 1.0, places=9)

    def test_unnormalised_sharpe_weight_auto_normalised(self):
        """Weights summing to != 1 are auto-normalised (including sharpe key)."""
        raw = {"return": 4, "risk": 3, "volume": 2, "sharpe": 1}  # sum = 10
        normalised = ScoringEngine.validate_weights(raw)
        self.assertAlmostEqual(sum(normalised.values()), 1.0, places=9)
        self.assertAlmostEqual(normalised["sharpe"], 0.1, places=9)

    def test_sharpe_criterion_in_results(self):
        """The 'sharpe' criterion appears in component_scores when weight > 0."""
        metrics = _two_stock_metrics(sharpe_a=1.5, sharpe_b=0.5)
        weights = {"return": 0.25, "risk": 0.25, "volume": 0.25, "sharpe": 0.25}
        results = ScoringEngine.compute_weighted_scores(metrics, weights)
        self.assertIn("sharpe", results[0]["component_scores"])
        self.assertIn("sharpe", results[0]["normalized"])
        self.assertIn("sharpe", results[0]["weights_used"])


if __name__ == "__main__":
    unittest.main()
