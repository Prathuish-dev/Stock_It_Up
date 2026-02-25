"""
Unit tests for ExplanationEngine.

Uses a synthetic scoring_results dict (no disk I/O, no DataLoader).
Verifies structure, determinism, threshold rules, and comparative logic.
"""

import unittest
from chatbot.explanation_engine import ExplanationEngine, _Thresholds
from chatbot.enums import RiskProfile, InvestmentHorizon


# ---------------------------------------------------------------------------
# Synthetic scoring_results (matches ScoringEngine.compute_weighted_scores output)
# ---------------------------------------------------------------------------

def _make_result(ticker, total_score, norm_ret, norm_risk, norm_vol,
                 cagr=0.12, vol=0.20, avgv=1_000_000, price=1500.0,
                 w_ret=0.50, w_risk=0.30, w_vol=0.20):
    return {
        "ticker": ticker,
        "total_score": total_score,
        "normalized":  {"return": norm_ret, "risk": norm_risk, "volume": norm_vol},
        "raw":         {"return": cagr,     "risk": vol,       "volume": avgv},
        "component_scores": {
            "return": round(norm_ret * w_ret, 5),
            "risk":   round(norm_risk * w_risk, 5),
            "volume": round(norm_vol * w_vol, 5),
        },
        "weights_used": {"return": w_ret, "risk": w_risk, "volume": w_vol},
        "metrics": {
            "cagr": cagr, "volatility": vol,
            "avg_volume": avgv, "latest_price": price,
        },
    }


# Two-stock ranked list: ALPHA #1, BETA #2
_RESULTS = [
    _make_result("ALPHA", 0.85000, norm_ret=1.0, norm_risk=0.9,  norm_vol=0.6,
                 cagr=0.18, vol=0.19, avgv=2_000_000, price=2500.0),
    _make_result("BETA",  0.45000, norm_ret=0.4, norm_risk=0.3,  norm_vol=1.0,
                 cagr=0.11, vol=0.26, avgv=5_000_000, price=900.0),
]


class TestExplanationEngineStructure(unittest.TestCase):
    """Verify that explain() always returns the five required keys."""

    def _explain(self, ticker="ALPHA", rp=None, horizon=None):
        return ExplanationEngine.explain(ticker, _RESULTS, rp, horizon)

    def test_returns_all_five_keys(self):
        exp = self._explain()
        self.assertEqual(
            set(exp.keys()),
            {"summary", "numeric_breakdown", "qualitative_analysis",
             "comparative_analysis", "final_statement"},
        )

    def test_all_sections_are_strings(self):
        exp = self._explain()
        for key, val in exp.items():
            self.assertIsInstance(val, str, f"Section '{key}' is not a string")

    def test_all_sections_non_empty(self):
        exp = self._explain()
        for key, val in exp.items():
            self.assertTrue(len(val.strip()) > 0, f"Section '{key}' is empty")

    def test_raises_for_unknown_ticker(self):
        with self.assertRaises(ValueError):
            ExplanationEngine.explain("ZZZNOPE", _RESULTS)

    def test_case_insensitive_ticker(self):
        exp_upper = ExplanationEngine.explain("ALPHA", _RESULTS)
        exp_lower = ExplanationEngine.explain("alpha", _RESULTS)
        self.assertEqual(exp_upper["summary"], exp_lower["summary"])

    def test_deterministic_same_output(self):
        exp1 = self._explain("ALPHA")
        exp2 = self._explain("ALPHA")
        self.assertEqual(exp1, exp2)


class TestExplanationEngineSummary(unittest.TestCase):

    def test_rank1_labelled_top_ranked(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("Top-ranked", exp["summary"])
        self.assertIn("ALPHA", exp["summary"])

    def test_rank2_labelled_upper_half(self):
        exp = ExplanationEngine.explain("BETA", _RESULTS)
        self.assertIn("BETA", exp["summary"])
        self.assertIn("#2", exp["summary"])

    def test_cagr_in_summary(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("CAGR", exp["summary"])


class TestExplanationEngineNumericBreakdown(unittest.TestCase):

    def test_contains_return_risk_volume(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        nb = exp["numeric_breakdown"]
        for label in ("return", "risk", "volume"):
            self.assertIn(label, nb)

    def test_contains_total_score(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("FINAL SCORE", exp["numeric_breakdown"])

    def test_numeric_score_matches(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("0.85000", exp["numeric_breakdown"])


class TestExplanationEngineQualitative(unittest.TestCase):
    """Verify threshold rules fire correctly."""

    def test_strong_return_commentary(self):
        # ALPHA: norm_ret=1.0 → above STRONG threshold (0.75)
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("Strong", exp["qualitative_analysis"])

    def test_below_average_return_commentary(self):
        # One-stock result with very low norm return
        low_result = [_make_result("LOW", 0.1, norm_ret=0.1, norm_risk=0.5, norm_vol=0.5)]
        exp = ExplanationEngine.explain("LOW", low_result)
        self.assertIn("Below-average", exp["qualitative_analysis"])

    def test_high_stability_commentary(self):
        # ALPHA: norm_risk=0.9 → highly stable
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("Highly stable", exp["qualitative_analysis"])

    def test_high_volatility_commentary(self):
        # BETA: norm_risk=0.3 → higher volatility
        exp = ExplanationEngine.explain("BETA", _RESULTS)
        self.assertIn("Higher volatility", exp["qualitative_analysis"])

    def test_high_liquidity_commentary(self):
        # BETA: norm_vol=1.0 → highly liquid
        exp = ExplanationEngine.explain("BETA", _RESULTS)
        self.assertIn("Highly liquid", exp["qualitative_analysis"])

    def test_low_risk_profile_note(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS, risk_profile=RiskProfile.LOW)
        self.assertIn("LOW risk", exp["qualitative_analysis"])

    def test_high_risk_profile_note(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS, risk_profile=RiskProfile.HIGH)
        self.assertIn("HIGH risk", exp["qualitative_analysis"])

    def test_horizon_note_present(self):
        exp = ExplanationEngine.explain(
            "ALPHA", _RESULTS, investment_horizon=InvestmentHorizon.LONG
        )
        self.assertIn("Long-term", exp["qualitative_analysis"])


class TestExplanationEngineComparative(unittest.TestCase):

    def test_rank1_compares_to_rank2(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("BETA", exp["comparative_analysis"])

    def test_rank2_compares_to_rank1(self):
        exp = ExplanationEngine.explain("BETA", _RESULTS)
        self.assertIn("ALPHA", exp["comparative_analysis"])

    def test_comparative_contains_score_gap(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("0.85", exp["comparative_analysis"])

    def test_single_stock_no_comparison(self):
        single = [_make_result("SOLO", 0.7, 0.8, 0.8, 0.8)]
        exp = ExplanationEngine.explain("SOLO", single)
        self.assertIn("no comparison", exp["comparative_analysis"].lower())


class TestExplanationEngineFinalStatement(unittest.TestCase):

    def test_rank1_recommends(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIn("RECOMMENDATION", exp["final_statement"])
        self.assertIn("ALPHA", exp["final_statement"])

    def test_rank2_notes(self):
        exp = ExplanationEngine.explain("BETA", _RESULTS)
        self.assertIn("NOTE", exp["final_statement"])

    def test_low_risk_recommendation_mentions_stability(self):
        exp = ExplanationEngine.explain(
            "ALPHA", _RESULTS, risk_profile=RiskProfile.LOW
        )
        self.assertIn("stability", exp["final_statement"].lower())

    def test_high_risk_recommendation_mentions_growth(self):
        exp = ExplanationEngine.explain(
            "ALPHA", _RESULTS, risk_profile=RiskProfile.HIGH
        )
        self.assertIn("growth", exp["final_statement"].lower())


class TestFormatForCLI(unittest.TestCase):

    def test_format_contains_all_sections(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        cli = ExplanationEngine.format_for_cli(exp)
        for key, val in exp.items():
            # Each section's first line should be findable in the CLI output
            first_line = val.splitlines()[0].strip()
            if first_line:
                self.assertIn(first_line, cli, f"Section '{key}' missing from CLI output")

    def test_format_returns_string(self):
        exp = ExplanationEngine.explain("ALPHA", _RESULTS)
        self.assertIsInstance(ExplanationEngine.format_for_cli(exp), str)


if __name__ == "__main__":
    unittest.main()
