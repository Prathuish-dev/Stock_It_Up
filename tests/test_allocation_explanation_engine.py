"""
tests/test_allocation_explanation_engine.py
--------------------------------------------
Unit tests for AllocationExplanationEngine.

Coverage:
    Empty input edge case
    _summary        – count, ticker, percentage
    _allocation_table – header + rows
    _strategy_rationale – all three methods + unknown
    _risk_distribution  – concentration bands + risk-profile notes
    _final_statement    – conservative / aggressive / neutral
    explain()           – integration: correct keys + values
    format_for_cli()    – all sections present in output string
"""

import unittest

from chatbot.allocation_explanation_engine import AllocationExplanationEngine as AEE


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make(allocations_scores: list) -> list:
    """Build a minimal allocations list from (ticker, alloc, score) tuples."""
    return [
        {"ticker": t, "allocation": a, "total_score": s}
        for t, a, s in allocations_scores
    ]


_THREE = _make([("TCS", 0.41, 0.82), ("INFY", 0.32, 0.65), ("HDFCBANK", 0.27, 0.52)])
_TWO   = _make([("TCS", 0.70, 0.80), ("INFY", 0.30, 0.60)])
_ONE   = _make([("TCS", 1.00, 0.90)])


# ===========================================================================
# 1. Empty Input
# ===========================================================================

class TestEmptyInput(unittest.TestCase):

    def test_empty_returns_no_allocation_summary(self):
        result = AEE.explain([], method="proportional")
        self.assertEqual(result["summary"], "No allocation available.")

    def test_empty_has_all_keys(self):
        result = AEE.explain([], method="proportional")
        expected_keys = {
            "summary", "allocation_table", "strategy_rationale",
            "risk_distribution", "risk_decomposition", "capital_distribution",
            "portfolio_risk", "monte_carlo", "final_statement",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_empty_non_summary_sections_are_empty_strings(self):
        result = AEE.explain([], method="proportional")
        for key in ("allocation_table", "strategy_rationale",
                    "risk_distribution", "risk_decomposition",
                    "capital_distribution", "final_statement"):
            self.assertEqual(result[key], "")


# ===========================================================================
# 2. _summary
# ===========================================================================

class TestSummary(unittest.TestCase):

    def test_contains_stock_count(self):
        result = AEE._summary(_THREE)
        self.assertIn("3", result)

    def test_contains_top_ticker(self):
        result = AEE._summary(_THREE)
        self.assertIn("TCS", result)

    def test_contains_percentage(self):
        result = AEE._summary(_THREE)
        self.assertIn("41.0%", result)

    def test_single_stock_grammar(self):
        """'1 stock' not '1 stocks'."""
        result = AEE._summary(_ONE)
        self.assertIn("1 stock.", result)
        self.assertNotIn("1 stocks", result)

    def test_plural_grammar(self):
        result = AEE._summary(_THREE)
        self.assertIn("stocks", result)

    def test_highest_allocator_is_selected(self):
        # INFY has higher allocation than TCS in this set
        stocks = _make([("TCS", 0.20, 0.80), ("INFY", 0.80, 0.60)])
        result = AEE._summary(stocks)
        self.assertIn("INFY", result)


# ===========================================================================
# 3. _allocation_table
# ===========================================================================

class TestAllocationTable(unittest.TestCase):

    def test_contains_header(self):
        table = AEE._allocation_table(_THREE)
        self.assertIn("Ticker", table)
        self.assertIn("Score", table)
        self.assertIn("Allocation", table)

    def test_contains_all_tickers(self):
        table = AEE._allocation_table(_THREE)
        for ticker in ("TCS", "INFY", "HDFCBANK"):
            self.assertIn(ticker, table)

    def test_contains_percentages(self):
        table = AEE._allocation_table(_THREE)
        self.assertIn("41.0%", table)
        self.assertIn("32.0%", table)
        self.assertIn("27.0%", table)

    def test_contains_scores(self):
        table = AEE._allocation_table(_THREE)
        self.assertIn("0.820", table)
        self.assertIn("0.650", table)

    def test_row_count_matches_stock_count(self):
        table = AEE._allocation_table(_THREE)
        # header line + 3 stock rows = 4 lines
        self.assertEqual(len(table.splitlines()), 4)

    def test_single_stock_table(self):
        table = AEE._allocation_table(_ONE)
        self.assertIn("TCS", table)
        self.assertIn("100.0%", table)


# ===========================================================================
# 4. _strategy_rationale
# ===========================================================================

class TestStrategyRationale(unittest.TestCase):

    def test_proportional_mentions_proportional(self):
        r = AEE._strategy_rationale("proportional")
        self.assertIn("proportional", r.lower())

    def test_softmax_mentions_softmax(self):
        r = AEE._strategy_rationale("softmax")
        self.assertIn("softmax", r.lower())

    def test_risk_adjusted_mentions_volatility(self):
        r = AEE._strategy_rationale("risk_adjusted")
        self.assertIn("volatility", r.lower())

    def test_unknown_method_returns_fallback(self):
        r = AEE._strategy_rationale("unknown_method")
        self.assertIn("strategy", r.lower())

    def test_all_known_methods_return_non_empty(self):
        for method in ("proportional", "softmax", "risk_adjusted"):
            self.assertTrue(len(AEE._strategy_rationale(method)) > 0)


# ===========================================================================
# 5. _risk_distribution
# ===========================================================================

class TestRiskDistribution(unittest.TestCase):

    def test_highly_concentrated(self):
        stocks = _make([("A", 0.90, 0.9), ("B", 0.10, 0.5)])
        r = AEE._risk_distribution(stocks, None)
        self.assertIn("Highly concentrated", r)

    def test_moderately_concentrated(self):
        stocks = _make([("A", 0.65, 0.9), ("B", 0.35, 0.5)])
        r = AEE._risk_distribution(stocks, None)
        self.assertIn("Moderately concentrated", r)

    def test_well_diversified(self):
        stocks = _make([("A", 0.55, 0.9), ("B", 0.45, 0.5)])
        r = AEE._risk_distribution(stocks, None)
        self.assertIn("Well diversified", r)

    def test_conservative_note_included(self):
        r = AEE._risk_distribution(_TWO, "LOW")
        self.assertIn("stability", r.lower())

    def test_aggressive_note_included(self):
        r = AEE._risk_distribution(_TWO, "HIGH")
        self.assertIn("aggressive", r.lower())

    def test_no_profile_no_extra_note(self):
        r = AEE._risk_distribution(_TWO, None)
        self.assertNotIn("stability", r.lower())
        self.assertNotIn("aggressive", r.lower())

    def test_conservative_alias(self):
        r1 = AEE._risk_distribution(_TWO, "LOW")
        r2 = AEE._risk_distribution(_TWO, "CONSERVATIVE")
        self.assertEqual(r1, r2)

    def test_aggressive_alias(self):
        r1 = AEE._risk_distribution(_TWO, "HIGH")
        r2 = AEE._risk_distribution(_TWO, "AGGRESSIVE")
        self.assertEqual(r1, r2)


# ===========================================================================
# 6. _final_statement
# ===========================================================================

class TestFinalStatement(unittest.TestCase):

    def test_conservative_mentions_stability(self):
        r = AEE._final_statement(_THREE, "LOW")
        self.assertIn("stabilit", r.lower())

    def test_aggressive_mentions_aggressive(self):
        r = AEE._final_statement(_THREE, "HIGH")
        self.assertIn("aggressive", r.lower())

    def test_neutral_mentions_balanced(self):
        r = AEE._final_statement(_THREE, None)
        self.assertIn("balanced", r.lower())

    def test_top_ticker_always_mentioned(self):
        for profile in ("LOW", "HIGH", None):
            r = AEE._final_statement(_THREE, profile)
            self.assertIn("TCS", r)

    def test_medium_profile_uses_neutral_path(self):
        r = AEE._final_statement(_THREE, "MEDIUM")
        self.assertIn("balanced", r.lower())


# ===========================================================================
# 7. explain() integration
# ===========================================================================

class TestExplain(unittest.TestCase):

    def _result(self, **kwargs):
        return AEE.explain(_THREE, method="proportional", **kwargs)

    def test_returns_dict_with_five_keys(self):
        r = self._result()
        self.assertEqual(
            set(r.keys()),
            {"summary", "allocation_table", "strategy_rationale",
             "risk_distribution", "risk_decomposition",
             "capital_distribution", "portfolio_risk", "final_statement"},
        )

    def test_all_sections_are_strings(self):
        r = self._result()
        for val in r.values():
            self.assertIsInstance(val, str)

    def test_all_sections_non_empty(self):
        r = self._result()
        for key, val in r.items():
            self.assertTrue(len(val) > 0, f"Section '{key}' is empty")

    def test_summary_consistent_with_standalone(self):
        r = self._result()
        self.assertEqual(r["summary"], AEE._summary(_THREE))

    def test_risk_profile_propagated(self):
        r = AEE.explain(_THREE, method="risk_adjusted", risk_profile="HIGH")
        self.assertIn("aggressive", r["risk_distribution"].lower())

    def test_deterministic(self):
        r1 = AEE.explain(_THREE, method="softmax", risk_profile="LOW")
        r2 = AEE.explain(_THREE, method="softmax", risk_profile="LOW")
        self.assertEqual(r1, r2)

    def test_does_not_mutate_input(self):
        import copy
        original = copy.deepcopy(_THREE)
        AEE.explain(_THREE, method="proportional")
        self.assertEqual(_THREE, original)


# ===========================================================================
# 8. format_for_cli()
# ===========================================================================

class TestFormatForCli(unittest.TestCase):

    def setUp(self):
        self.explanation = AEE.explain(
            _THREE, method="proportional", risk_profile=None
        )
        self.cli_out = AEE.format_for_cli(self.explanation)

    def test_returns_string(self):
        self.assertIsInstance(self.cli_out, str)

    def test_contains_header(self):
        self.assertIn("=== Portfolio Allocation Summary ===", self.cli_out)

    def test_contains_summary(self):
        self.assertIn(self.explanation["summary"], self.cli_out)

    def test_contains_table(self):
        self.assertIn(self.explanation["allocation_table"], self.cli_out)

    def test_contains_rationale(self):
        self.assertIn(self.explanation["strategy_rationale"], self.cli_out)

    def test_contains_risk_distribution(self):
        self.assertIn(self.explanation["risk_distribution"], self.cli_out)

    def test_contains_final_statement(self):
        self.assertIn(self.explanation["final_statement"], self.cli_out)

    def test_contains_risk_decomposition(self):
        self.assertIn("--- Risk Decomposition ---", self.cli_out)

    def test_contains_capital_distribution(self):
        self.assertIn("--- Capital Distribution ---", self.cli_out)

    def test_empty_explanation_renders(self):
        """format_for_cli must not crash on the empty-input response."""
        empty = AEE.explain([], method="proportional")
        out = AEE.format_for_cli(empty)
        self.assertIsInstance(out, str)
        self.assertIn("No allocation available.", out)




# ===========================================================================
# 9. _risk_decomposition
# ===========================================================================

class TestRiskDecompositionExplanation(unittest.TestCase):

    def test_returns_not_available_if_missing(self):
        r = AEE._risk_decomposition(_THREE)
        self.assertEqual(r, "Risk decomposition not available.")

    def test_renders_table(self):
        allocs = [{"ticker": "TCS", "risk_share": 0.6}, {"ticker": "INFY", "risk_share": 0.4}]
        r = AEE._risk_decomposition(allocs)
        self.assertIn("Ticker", r)
        self.assertIn("TCS", r)
        self.assertIn("60.0%", r)
        self.assertIn("INFY", r)
        self.assertIn("40.0%", r)
        self.assertIn("TCS contributes the largest share", r)


# ===========================================================================
# 10. _capital_distribution
# ===========================================================================

class TestCapitalDistributionExplanation(unittest.TestCase):

    def test_returns_not_available_if_missing(self):
        r = AEE._capital_distribution(_THREE)
        self.assertEqual(r, "Capital distribution not available.")

    def test_renders_table(self):
        allocs = [{"ticker": "TCS", "capital_amount": 41000.0}, {"ticker": "INFY", "capital_amount": 32000.5}]
        r = AEE._capital_distribution(allocs)
        self.assertIn("Ticker", r)
        self.assertIn("TCS", r)
        self.assertIn("41,000.00", r)
        self.assertIn("INFY", r)
        self.assertIn("32,000.50", r)


if __name__ == "__main__":
    unittest.main()
