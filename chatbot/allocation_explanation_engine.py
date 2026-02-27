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

from chatbot.portfolio_engine import PortfolioEngine


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
            "portfolio_risk":       AllocationExplanationEngine._portfolio_risk_section(allocations),
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

        lines = ["Ticker     Capital Allocation (\u20b9)"]
        for a in allocations:
            lines.append(
                f"{a['ticker']:<10} "
                f"{a['capital_amount']:>12,.2f}"
            )

        return "\n".join(lines)

    @staticmethod
    def _portfolio_risk_section(allocations: List[Dict]) -> str:
        """
        Portfolio-level risk metrics using the independent-stock assumption.

        Requires each allocation dict to have ``"cagr"`` and ``"volatility"``
        keys (present when allocations are produced via the full pipeline).
        Returns a brief notice when those keys are unavailable.
        """
        if not allocations or "cagr" not in allocations[0]:
            return "Portfolio risk metrics not available (cagr/volatility not in allocation)."

        summary = PortfolioEngine.portfolio_summary(allocations)
        ret  = summary["portfolio_return"]     * 100
        vol  = summary["portfolio_volatility"] * 100
        sharpe = summary["portfolio_sharpe"]

        if sharpe >= 1.0:
            interpretation = (
                f"  The portfolio delivers {sharpe:.2f} units of excess return per unit "
                "of risk \u2014 strong risk-adjusted construction."
            )
        elif sharpe >= 0.5:
            interpretation = (
                f"  Balanced risk-return profile (Sharpe {sharpe:.2f}). "
                "Reasonable compensation for the volatility incurred."
            )
        elif sharpe >= 0.0:
            interpretation = (
                f"  Weak risk-return compensation (Sharpe {sharpe:.2f}). "
                "Returns are not fully justifying the portfolio volatility."
            )
        else:
            interpretation = (
                f"  Portfolio underperforms the risk-free rate on a risk-adjusted basis "
                f"(Sharpe {sharpe:.2f}). Consider rebalancing toward higher-Sharpe stocks."
            )

        sep = "-" * 46
        lines = [
            sep,
            "  Portfolio Risk-Adjusted Performance",
            sep,
            f"  Expected Return      : {ret:>7.2f}%",
            f"  Portfolio Volatility : {vol:>5.2f}%   (independent-stock assumption)",
            f"  Portfolio Sharpe     : {sharpe:>7.2f}",
        ]

        # Max Drawdown row — only when mdd is available in summary
        mdd = summary.get("portfolio_mdd")
        if mdd is not None and mdd > 0.0:
            lines.append(f"  Max Drawdown         : {mdd * 100:>5.2f}%")

        # Sortino row — only when available
        sortino = summary.get("portfolio_sortino")
        if sortino is not None:
            lines.append(f"  Portfolio Sortino    : {sortino:>7.2f}")

        lines += [sep, interpretation]

        # MDD interpretation
        if mdd is not None and mdd > 0.0:
            if mdd < 0.10:
                lines.append(
                    f"  Capital Stability    : Excellent (MDD {mdd*100:.2f}%). "
                    "Portfolio has minimal peak-to-trough decline history."
                )
            elif mdd < 0.25:
                lines.append(
                    f"  Capital Stability    : Moderate (MDD {mdd*100:.2f}%). "
                    "Expect meaningful but recoverable drawdowns."
                )
            else:
                lines.append(
                    f"  Capital Stability    : Significant (MDD {mdd*100:.2f}%). "
                    "High capital-pain risk; ensure position sizing reflects this."
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
            "portfolio_risk":       "",
            "monte_carlo":          "",
            "final_statement":      "",
        }

    @staticmethod
    def _monte_carlo_section(mc: dict) -> str:
        """
        Render a Monte Carlo simulation result block.

        Parameters
        ----------
        mc : dict
            Output from ``PortfolioEngine.simulate_portfolio_monte_carlo()``.
            Keys: mean_return, std_dev, var_95, cvar_95, probability_of_loss,
                  simulated_returns.

        Returns
        -------
        str
            Formatted multi-line string, or empty string when mc is empty.
        """
        if not mc:
            return ""

        n_sims   = len(mc.get("simulated_returns", []))
        mean_r   = mc["mean_return"]         * 100
        std_d    = mc["std_dev"]             * 100
        var_95   = mc["var_95"]              * 100
        cvar_95  = mc["cvar_95"]             * 100
        p_loss   = mc["probability_of_loss"] * 100

        if p_loss < 20:
            interpretation = "Low tail risk. Unlikely to produce negative annual returns under simulated conditions."
        elif p_loss < 40:
            interpretation = "Moderate downside tail risk observed. Diversification or hedging may reduce exposure."
        else:
            interpretation = "High tail risk. A significant proportion of simulated outcomes are negative."

        sep = "-" * 46
        lines = [
            sep,
            f"  Monte Carlo Stress Simulation ({n_sims:,} runs)",
            sep,
            f"  Mean Return        : {mean_r:>+7.2f}%",
            f"  Volatility (sim)   : {std_d:>7.2f}%",
            f"  95% VaR            : {var_95:>+7.2f}%",
            f"  95% CVaR           : {cvar_95:>+7.2f}%",
            f"  Probability of Loss: {p_loss:>6.1f}%",
            sep,
            f"  {interpretation}",
        ]
        return "\n".join(lines)


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
            explanation["portfolio_risk"],
            "",
            explanation.get("monte_carlo", ""),
            "",
            explanation["final_statement"],
        ]
        return "\n".join(sections)
