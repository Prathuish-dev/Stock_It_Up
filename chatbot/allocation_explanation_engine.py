"""
chatbot/allocation_explanation_engine.py
-----------------------------------------
Deterministic, formatting-aware explanation engine for portfolio allocations.

Design contract:
  - Does NOT compute scores
  - Does NOT compute metrics
  - Does NOT mutate allocations
  - Only interprets and explains PortfolioEngine output
  - Fully stateless (all methods are @staticmethod)
  - Mirrors the 5-section structure of ExplanationEngine
"""

from typing import List, Dict, Optional


class AllocationExplanationEngine:
    """
    Produce structured, human-readable explanations for portfolio allocations.

    Entry point::

        explanation = AllocationExplanationEngine.explain(
            allocations, method="proportional", risk_profile="LOW"
        )

        Returns a dict with seven string sections:
        ``summary``              – one-line overview
        ``allocation_table``     – ASCII breakdown table
        ``strategy_rationale``   – why this method produces these weights
        ``risk_distribution``    – concentration analysis + risk-profile note
        ``risk_decomposition``   – risk share per stock
        ``capital_distribution`` – capital amount per stock
        ``final_statement``      – closing interpretation sentence
    """

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #

    @staticmethod
    def explain(
        allocations: List[Dict],
        method: str,
        risk_profile: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Build the full five-section explanation for *allocations*.

        Parameters
        ----------
        allocations:
            Output from ``PortfolioEngine.allocate()`` — each dict must
            contain ``"ticker"``, ``"allocation"``, and ``"total_score"``.
        method:
            Allocation strategy used: ``"proportional"``, ``"softmax"``,
            or ``"risk_adjusted"``.
        risk_profile:
            Investor risk profile string (``"LOW"`` / ``"MEDIUM"`` /
            ``"HIGH"``), or ``None``.

        Returns
        -------
        Dict[str, str]  with keys ``summary``, ``allocation_table``,
        ``strategy_rationale``, ``risk_distribution``, ``risk_decomposition``,
        ``capital_distribution``, ``final_statement``.
        """
        if not allocations:
            return AllocationExplanationEngine._empty_response()

        return {
            "summary":              AllocationExplanationEngine._summary(allocations),
            "allocation_table":     AllocationExplanationEngine._allocation_table(allocations),
            "strategy_rationale":   AllocationExplanationEngine._strategy_rationale(method),
            "risk_distribution":    AllocationExplanationEngine._risk_distribution(
                                        allocations, risk_profile
                                    ),
            "risk_decomposition":   AllocationExplanationEngine._risk_decomposition(allocations),
            "capital_distribution": AllocationExplanationEngine._capital_distribution(allocations),
            "final_statement":      AllocationExplanationEngine._final_statement(
                                        allocations, risk_profile
                                    ),
        }

    # ------------------------------------------------------------------ #
    #  Section builders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _summary(allocations: List[Dict]) -> str:
        """One-sentence overview: stock count and largest holding."""
        top = max(allocations, key=lambda x: x["allocation"])
        count = len(allocations)
        pct = round(top["allocation"] * 100, 2)
        return (
            f"Portfolio constructed with {count} stock{'s' if count != 1 else ''}. "
            f"Largest allocation: {top['ticker']} ({pct}%)."
        )

    @staticmethod
    def _allocation_table(allocations: List[Dict]) -> str:
        """Fixed-width ASCII table: Ticker | Score | Allocation."""
        lines = ["Ticker     Score     Allocation"]
        for stock in allocations:
            ticker = stock["ticker"]
            score  = stock["total_score"]
            pct    = round(stock["allocation"] * 100, 2)
            lines.append(
                f"{ticker:<10} {score:.3f}     {pct:>6}%"
            )
        return "\n".join(lines)

    @staticmethod
    def _strategy_rationale(method: str) -> str:
        """Human-readable description of why the chosen method works this way."""
        _RATIONALE: Dict[str, str] = {
            "proportional": (
                "Allocations are proportional to each stock's total score. "
                "Higher-scoring stocks receive higher capital weight."
            ),
            "softmax": (
                "Allocations are derived using a softmax transformation, "
                "which smooths differences while still favouring stronger stocks."
            ),
            "risk_adjusted": (
                "Scores were adjusted based on volatility before allocation. "
                "Lower-risk stocks were favoured in accordance with the risk profile."
            ),
        }
        return _RATIONALE.get(method, "Allocation strategy applied.")

    @staticmethod
    def _risk_distribution(
        allocations: List[Dict],
        risk_profile: Optional[str],
    ) -> str:
        """
        Describe portfolio concentration and add a risk-profile note.

        Concentration is classed by the spread between the largest and
        smallest allocation weights:
          > 0.40  → Highly concentrated
          > 0.20  → Moderately concentrated
          ≤ 0.20  → Well diversified
        """
        max_w = max(a["allocation"] for a in allocations)
        min_w = min(a["allocation"] for a in allocations)
        spread = max_w - min_w

        if spread > 0.40:
            concentration = "Highly concentrated"
        elif spread > 0.20:
            concentration = "Moderately concentrated"
        else:
            concentration = "Well diversified"

        profile_note = ""
        rp = (risk_profile or "").upper()
        if rp in ("LOW", "CONSERVATIVE"):
            profile_note = (
                " Allocation favours stability consistent with a conservative strategy."
            )
        elif rp in ("HIGH", "AGGRESSIVE"):
            profile_note = (
                " Higher allocation toward high-scoring growth stocks "
                "reflects an aggressive strategy."
            )

        return f"{concentration} allocation structure.{profile_note}"

    @staticmethod
    def _risk_decomposition(allocations: List[Dict]) -> str:
        """Present each stock's contribution to portfolio risk."""
        if "risk_share" not in allocations[0]:
            return "Risk decomposition not available."

        lines = ["Ticker     Risk Share"]
        for a in allocations:
            lines.append(
                f"{a['ticker']:<10} "
                f"{round(a['risk_share']*100,2):>6}%"
            )

        dominant = max(allocations, key=lambda x: x["risk_share"])
        lines.append("")
        lines.append(
            f"{dominant['ticker']} contributes the largest share of portfolio risk."
        )

        return "\n".join(lines)

    @staticmethod
    def _capital_distribution(allocations: List[Dict]) -> str:
        """Formulate the capital distribution table."""
        if "capital_amount" not in allocations[0]:
            return "Capital distribution not available."

        lines = ["Ticker     Capital Allocation (₹)"]
        for a in allocations:
            lines.append(
                f"{a['ticker']:<10} "
                f"{a['capital_amount']:>12,.2f}"
            )

        return "\n".join(lines)

    @staticmethod
    def _final_statement(
        allocations: List[Dict],
        risk_profile: Optional[str],
    ) -> str:
        """Closing interpretation tailored to risk profile."""
        top = max(allocations, key=lambda x: x["allocation"])
        rp  = (risk_profile or "").upper()

        if rp in ("LOW", "CONSERVATIVE"):
            return (
                f"The portfolio prioritises stability while maintaining exposure "
                f"to growth via {top['ticker']}."
            )
        if rp in ("HIGH", "AGGRESSIVE"):
            return (
                f"The portfolio aggressively emphasises high-performing stocks, "
                f"with {top['ticker']} leading the allocation."
            )
        return (
            f"Portfolio allocation reflects a balanced scoring methodology "
            f"with {top['ticker']} as the primary holding."
        )

    # ------------------------------------------------------------------ #
    #  Edge-case response
    # ------------------------------------------------------------------ #

    @staticmethod
    def _empty_response() -> Dict[str, str]:
        return {
            "summary":              "No allocation available.",
            "allocation_table":     "",
            "strategy_rationale":   "",
            "risk_distribution":    "",
            "risk_decomposition":   "",
            "capital_distribution": "",
            "final_statement":      "",
        }

    # ------------------------------------------------------------------ #
    #  CLI formatter
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_for_cli(explanation: Dict[str, str]) -> str:
        """
        Render all five sections as a single printable CLI string.

        Example::

            print(AllocationExplanationEngine.format_for_cli(explanation))
        """
        sections = [
            "=== Portfolio Allocation Summary ===",
            explanation["summary"],
            "",
            explanation["allocation_table"],
            "",
            explanation["strategy_rationale"],
            "",
            explanation["risk_distribution"],
            "",
            "--- Risk Decomposition ---",
            explanation["risk_decomposition"],
            "",
            "--- Capital Distribution ---",
            explanation["capital_distribution"],
            "",
            explanation["final_statement"],
        ]
        return "\n".join(sections)
