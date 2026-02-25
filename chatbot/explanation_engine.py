"""
ExplanationEngine — deterministic, rule-based natural language explainer.

Responsibilities
----------------
* Accept the structured output produced by :class:`ScoringEngine` and a
  ``SessionContext`` snapshot.
* Produce a human-readable, multi-section explanation for a single ticker.
* Never touch scoring logic, weights, or raw data computation.
* Return a structured dict so callers can format the sections however they
  need (CLI table, JSON API, future GUI, etc.).

Design principles
-----------------
* **Pure interpretation** — inputs are already-computed; this engine only
  reads and translates them into language.
* **Deterministic** — no AI calls.  Same inputs always produce same output.
* **Threshold-based qualitative commentary** — explicit rules make the
  commentary auditable and easy to extend.
* **Separation of concerns** — ResponseGenerator formats; ExplanationEngine
  explains; ScoringEngine scores.  No crossover.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Threshold constants — single source of truth for all qualitative rules
# ---------------------------------------------------------------------------

class _Thresholds:
    STRONG   = 0.75     # normalised score ≥ this → "strong" / "high"
    MODERATE_LO = 0.40  # normalised score between MODERATE_LO and STRONG
    WEAK     = 0.40     # normalised score < this → "weak" / "low"


class ExplanationEngine:
    """
    Generates structured, deterministic explanations for a scored stock.

    All public methods are static — the engine has no mutable state.

    Entry point
    -----------
    :meth:`explain` — accepts ticker + scoring result + ranked list +
    session context snapshot, returns a five-section explanation dict.

    Output schema
    -------------
    ::

        {
            "summary":              str,   # one-line verdict
            "numeric_breakdown":    str,   # table of raw → norm → weight → contrib
            "qualitative_analysis": str,   # threshold-based prose commentary
            "comparative_analysis": str,   # how this stock compares to #2
            "final_statement":      str,   # decision-support conclusion
        }
    """

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #

    @staticmethod
    def explain(
        ticker: str,
        scoring_results: list,
        risk_profile=None,
        investment_horizon=None,
    ) -> dict:
        """
        Generate a full structured explanation for *ticker*.

        Parameters
        ----------
        ticker : str
            Ticker to explain (case-insensitive).
        scoring_results : list
            Full ranked list produced by ``ScoringEngine.compute_weighted_scores``.
            Each element must contain:
            ``ticker``, ``total_score``, ``normalized``, ``raw``,
            ``component_scores``, ``weights_used``, ``metrics``.
        risk_profile : RiskProfile or None
            The user's risk preference (used for contextual commentary).
        investment_horizon : InvestmentHorizon or None
            The user's time horizon (used for contextual commentary).

        Returns
        -------
        dict
            Five-section explanation dict (see class docstring for schema).

        Raises
        ------
        ValueError
            If *ticker* is not found in *scoring_results*.
        """
        ticker = ticker.upper()

        # Locate result for this ticker
        target = next(
            (r for r in scoring_results if r["ticker"].upper() == ticker), None
        )
        if target is None:
            raise ValueError(
                f"Ticker '{ticker}' not found in scoring results. "
                f"Available: {[r['ticker'] for r in scoring_results]}"
            )

        rank = next(
            (i + 1 for i, r in enumerate(scoring_results)
             if r["ticker"].upper() == ticker),
            None,
        )
        total = len(scoring_results)

        # Build each section
        summary = ExplanationEngine._build_summary(target, rank, total)
        numeric = ExplanationEngine._numeric_breakdown(target)
        qualitative = ExplanationEngine._qualitative_analysis(
            target, risk_profile, investment_horizon
        )
        comparative = ExplanationEngine._comparative_analysis(
            target, scoring_results, rank
        )
        final_stmt = ExplanationEngine._final_statement(
            target, rank, total, risk_profile
        )

        return {
            "summary":              summary,
            "numeric_breakdown":    numeric,
            "qualitative_analysis": qualitative,
            "comparative_analysis": comparative,
            "final_statement":      final_stmt,
        }

    # ------------------------------------------------------------------ #
    #  Section A — Numeric breakdown
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numeric_breakdown(result: dict) -> str:
        """Formatted table: raw value | normalised | weight | contribution."""
        raw = result.get("raw", {})
        nrm = result.get("normalized", {})
        wts = result.get("weights_used", {})
        cmp = result.get("component_scores", {})

        sep   = "=" * 64
        dash  = "-" * 64
        lines = [
            sep,
            f"  Numeric Breakdown — {result['ticker']}",
            sep,
            f"{'Criterion':<12} {'Raw Value':>14} {'Norm':>6} {'Weight':>8} {'Contrib':>9}",
            dash,
        ]

        fmt = {
            "return": lambda v: f"{v * 100:+.2f}% CAGR",
            "risk":   lambda v: f"{v * 100:.2f}% vol",
            "volume": lambda v: f"{v:,.0f} shrs",
        }

        for label in ["return", "risk", "volume"]:
            raw_str = fmt[label](raw.get(label, 0.0))
            lines.append(
                f"{label:<12} {raw_str:>14} "
                f"{nrm.get(label, 0.0):>6.3f} "
                f"{wts.get(label, 0.0):>8.2%} "
                f"{cmp.get(label, 0.0):>9.5f}"
            )

        lines.append(dash)
        lines.append(f"{'FINAL SCORE':>41} {result['total_score']:>9.5f}")
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Section B — Qualitative commentary
    # ------------------------------------------------------------------ #

    @staticmethod
    def _qualitative_analysis(
        result: dict,
        risk_profile,
        investment_horizon,
    ) -> str:
        """
        Rule-based commentary on each normalised dimension, plus a
        risk-profile and horizon justification paragraph.
        """
        nrm  = result.get("normalized", {})
        raw  = result.get("raw", {})
        wts  = result.get("weights_used", {})
        m    = result.get("metrics", {})

        lines = ["Qualitative Analysis:"]

        # -- Return --
        nr = nrm.get("return", 0.0)
        cagr_pct = raw.get("return", 0.0) * 100
        if nr >= _Thresholds.STRONG:
            lines.append(
                f"  Growth : Strong performer (CAGR {cagr_pct:+.2f}%). "
                "Ranks near the top of this cohort on returns."
            )
        elif nr >= _Thresholds.MODERATE_LO:
            lines.append(
                f"  Growth : Moderate performer (CAGR {cagr_pct:+.2f}%). "
                "Reasonable returns relative to peers."
            )
        else:
            lines.append(
                f"  Growth : Below-average growth in this cohort "
                f"(CAGR {cagr_pct:+.2f}%). May lag peers over the horizon."
            )

        # -- Risk (volatility — inverted: higher norm = lower volatility) --
        nv  = nrm.get("risk", 0.0)
        vol = raw.get("risk", 0.0) * 100
        if nv >= _Thresholds.STRONG:
            lines.append(
                f"  Risk   : Highly stable relative to peers "
                f"(annualised volatility {vol:.2f}%). "
                "Lower price swings support capital preservation."
            )
        elif nv >= _Thresholds.MODERATE_LO:
            lines.append(
                f"  Risk   : Moderate risk profile (volatility {vol:.2f}%). "
                "Typical fluctuations for this market segment."
            )
        else:
            lines.append(
                f"  Risk   : Higher volatility compared to peers "
                f"(volatility {vol:.2f}%). "
                "Expect significant price swings."
            )

        # -- Volume (liquidity) --
        nl   = nrm.get("volume", 0.0)
        avgv = raw.get("volume", 0.0)
        if nl >= _Thresholds.STRONG:
            lines.append(
                f"  Liquid : Highly liquid ({avgv:,.0f} avg shares/day). "
                "Easy entry and exit at tight spreads."
            )
        elif nl >= _Thresholds.MODERATE_LO:
            lines.append(
                f"  Liquid : Average liquidity ({avgv:,.0f} avg shares/day). "
                "Suitable for most retail order sizes."
            )
        else:
            lines.append(
                f"  Liquid : Lower liquidity compared to others "
                f"({avgv:,.0f} avg shares/day). "
                "Large positions may face wider spreads."
            )

        # -- Risk profile contextual note --
        lines.append("")
        rp_label = risk_profile.value if risk_profile else "medium"
        ret_wt   = wts.get("return", 0.0)
        risk_wt  = wts.get("risk",   0.0)

        if rp_label == "low":
            lines.append(
                f"  Risk context : Your LOW risk profile means stability "
                f"carries {risk_wt:.0%} of the score. "
                "Stocks with lower volatility are rewarded here."
            )
        elif rp_label == "high":
            lines.append(
                f"  Risk context : Your HIGH risk tolerance means CAGR "
                f"carries {ret_wt:.0%} of the score. "
                "Growth potential is maximised over safety."
            )
        else:
            lines.append(
                f"  Risk context : Balanced weighting — return "
                f"({ret_wt:.0%}) and risk ({risk_wt:.0%}) "
                "are given roughly equal importance."
            )

        # -- Horizon note --
        if investment_horizon:
            horizon_map = {"short": 1, "medium": 3, "long": 7}
            years = horizon_map.get(investment_horizon.value, 3)
            lines.append(
                f"  Horizon      : {investment_horizon.value.capitalize()}-term "
                f"({years}-year window). "
                "CAGR and volatility are computed over this period only."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Section C — Comparative analysis vs second-best
    # ------------------------------------------------------------------ #

    @staticmethod
    def _comparative_analysis(
        target: dict,
        scoring_results: list,
        rank: int,
    ) -> str:
        """
        Compare the target stock to its nearest rival:
        * If ranked #1 → compare to #2
        * If ranked ≥ #2 → compare to #1
        """
        if len(scoring_results) < 2:
            return "Comparative Analysis: Only one stock in this analysis — no comparison available."

        rival_idx  = 1 if rank == 1 else 0   # #2 for top pick, else #1
        rival      = scoring_results[rival_idx]
        r_ticker   = rival["ticker"]

        t_score  = target["total_score"]
        r_score  = rival["total_score"]
        t_cagr   = target["raw"].get("return", 0.0) * 100
        r_cagr   = rival["raw"].get("return",  0.0) * 100
        t_vol    = target["raw"].get("risk",    0.0) * 100
        r_vol    = rival["raw"].get("risk",     0.0) * 100
        t_liq    = target["raw"].get("volume",  0.0)
        r_liq    = rival["raw"].get("volume",   0.0)

        direction = "leads" if t_score > r_score else "trails"
        score_gap = abs(t_score - r_score)

        lines = [
            f"Comparative Analysis: {target['ticker']} vs {r_ticker}",
            f"  Score     : {target['ticker']} {direction} {r_ticker} "
            f"by {score_gap:.5f} ({t_score:.4f} vs {r_score:.4f})",
        ]

        # CAGR comparison
        cagr_diff = t_cagr - r_cagr
        cagr_dir  = "higher" if cagr_diff > 0 else "lower"
        lines.append(
            f"  CAGR      : {target['ticker']} is {abs(cagr_diff):.2f}pp {cagr_dir} "
            f"({t_cagr:+.2f}% vs {r_cagr:+.2f}%)"
        )

        # Volatility comparison (lower is better)
        vol_diff = t_vol - r_vol
        vol_dir  = "more volatile" if vol_diff > 0 else "less volatile"
        lines.append(
            f"  Volatility: {target['ticker']} is {abs(vol_diff):.2f}pp {vol_dir} "
            f"({t_vol:.2f}% vs {r_vol:.2f}%)"
        )

        # Liquidity comparison
        liq_diff  = t_liq - r_liq
        liq_dir   = "higher" if liq_diff > 0 else "lower"
        lines.append(
            f"  Liquidity : {target['ticker']} has {abs(liq_diff):,.0f} "
            f"shares/day {liq_dir} avg volume "
            f"({t_liq:,.0f} vs {r_liq:,.0f})"
        )

        # Trade-off summary
        lines.append("")
        if rank == 1:
            if cagr_diff >= 0 and vol_diff <= 0:
                lines.append(
                    f"  Verdict: {target['ticker']} dominates {r_ticker} on "
                    "both return and stability — a clear top pick."
                )
            elif cagr_diff >= 0:
                lines.append(
                    f"  Verdict: {target['ticker']} wins on return but is more "
                    f"volatile than {r_ticker}. Justified for growth-oriented profiles."
                )
            else:
                lines.append(
                    f"  Verdict: {target['ticker']} ranks #1 via lower volatility "
                    f"despite lower CAGR than {r_ticker}. "
                    "Reflects the current weight on stability."
                )
        else:
            lines.append(
                f"  Verdict: {target['ticker']} ranks below {r_ticker}. "
                "Consider whether the trade-off fits your profile."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Helper builders
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_summary(result: dict, rank: int, total: int) -> str:
        score    = result["total_score"]
        ticker   = result["ticker"]
        cagr     = result.get("raw", {}).get("return", 0.0) * 100
        vol      = result.get("raw", {}).get("risk",   0.0) * 100

        if rank == 1:
            verdict = "Top-ranked pick"
        elif rank <= max(2, total // 2):
            verdict = "Upper-half performer"
        else:
            verdict = "Lower-ranked option"

        return (
            f"{ticker} — {verdict}  "
            f"[Rank #{rank}/{total}  |  Score {score:.4f}  |  "
            f"CAGR {cagr:+.1f}%  |  Volatility {vol:.1f}%]"
        )

    @staticmethod
    def _final_statement(
        result: dict,
        rank: int,
        total: int,
        risk_profile,
    ) -> str:
        ticker    = result["ticker"]
        score     = result["total_score"]
        rp_label  = risk_profile.value if risk_profile else "medium"

        if rank == 1:
            if rp_label == "low":
                return (
                    f"RECOMMENDATION: {ticker} is the top-ranked stock "
                    "and aligns well with your conservative profile — "
                    "it scored highest on stability-weighted criteria. "
                    "Consider allocating your full budget here unless "
                    "diversification is a priority."
                )
            elif rp_label == "high":
                return (
                    f"RECOMMENDATION: {ticker} leads the cohort on weighted "
                    "return criteria (score {score:.4f}). "
                    "Suitable for your high-risk, growth-oriented stance. "
                    "Monitor closely — higher-return stocks often carry "
                    "elevated volatility."
                )
            else:
                return (
                    f"RECOMMENDATION: {ticker} is the top pick under balanced "
                    f"scoring (score {score:.4f}). A reasonable first allocation "
                    "candidate given your medium risk profile."
                )
        elif rank == 2:
            return (
                f"NOTE: {ticker} ranks #2 of {total}. It is a viable "
                "alternative if you want to diversify away from the top pick, "
                "or if your risk profile changes."
            )
        else:
            return (
                f"NOTE: {ticker} ranks #{rank} of {total} under your current "
                "criteria and weights. Consider whether adjusting the weights "
                "or time horizon would better surface this stock's strengths."
            )

    # ------------------------------------------------------------------ #
    #  Convenience: format the full dict as printable CLI text
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_for_cli(explanation: dict) -> str:
        """
        Render all five sections as a single, printable CLI string.

        This is a formatting-only helper — it never recomputes anything.
        """
        sep = "=" * 64
        parts = [
            sep,
            f"  {explanation['summary']}",
            sep,
            "",
            explanation["numeric_breakdown"],
            "",
            explanation["qualitative_analysis"],
            "",
            explanation["comparative_analysis"],
            "",
            explanation["final_statement"],
            sep,
        ]
        return "\n".join(parts)
