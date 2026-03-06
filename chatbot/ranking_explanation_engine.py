"""
chatbot/ranking_explanation_engine.py
--------------------------------------
Deterministic, formatting-aware explanation engine for ranking results.

Design contract:
  - Does NOT compute scores or metrics
  - Does NOT mutate results
  - Only interprets and explains RankingService / ScreenerEngine output
  - Fully stateless (all methods are @staticmethod)
"""

from __future__ import annotations

from typing import List, Dict, Optional


class RankingExplanationEngine:
    """
    Produce structured, human-readable explanations for a ranked stock list.

    Entry point::

        explanation = RankingExplanationEngine.explain(
            results,
            metric="cagr",
            order="best",
            exchange="NSE",
            horizon_years=3,
            weights=None,
        )

    Returns a dict with string sections:
        ``summary``          – one-line overview of the screen
        ``methodology``      – how stocks were scored / ranked
        ``top_stocks``       – breakdown table of the top results
        ``metric_insight``   – what the chosen metric means
        ``weights_used``     – (score/custom mode only) active weight breakdown
        ``final_statement``  – closing remark
    """

    # -- Human-readable metric descriptions ---------------------------------

    METRIC_DESCRIPTIONS: Dict[str, str] = {
        "cagr": (
            "CAGR (Compound Annual Growth Rate) measures the mean annual growth of "
            "an investment over a specified period. Higher CAGR indicates stronger "
            "long-term price appreciation."
        ),
        "volatility": (
            "Volatility captures the standard deviation of daily returns, annualised. "
            "Lower volatility implies a smoother ride; 'best' here means the least "
            "volatile (most stable) stocks."
        ),
        "avg_volume": (
            "Average daily traded volume is a liquidity proxy. Higher volume means "
            "easier entry and exit without large price impact."
        ),
        "latest_price": (
            "Latest closing price in ₹. Stocks are ranked by raw price — this is "
            "informational only and does not imply value."
        ),
        "sharpe": (
            "Sharpe Ratio measures risk-adjusted return: (CAGR − risk-free rate) / "
            "volatility. Higher Sharpe means more return earned per unit of risk taken."
        ),
        "max_drawdown": (
            "Max Drawdown is the largest peak-to-trough decline in the window. "
            "Lower max drawdown means better capital preservation during downturns."
        ),
        "sortino": (
            "Sortino Ratio is like Sharpe but penalises only downside volatility. "
            "Higher Sortino indicates superior risk-adjusted performance with less "
            "downside exposure."
        ),
        "score": (
            "Custom Score is a composite, weighted metric combining Return (CAGR), "
            "Risk (Volatility), Volume, and optionally Sharpe, Max Drawdown, and "
            "Sortino. Weights are auto-normalised to sum to 100%."
        ),
    }

    # -----------------------------------------------------------------------
    #  Public entry point
    # -----------------------------------------------------------------------

    @staticmethod
    def explain(
        results: List[Dict],
        *,
        metric: str,
        order: str,
        exchange: str,
        horizon_years: int,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, str]:
        """
        Build the full explanation for a ranked results list.

        Parameters
        ----------
        results : list of RankingItem dicts (or similar with ticker/metric_value/cagr/etc.)
        metric  : active metric key, e.g. "cagr", "score"
        order   : "best" or "worst"
        exchange: "NSE" or "BSE"
        horizon_years: look-back window used
        weights : effective normalised weights (for score/custom mode)

        Returns
        -------
        dict with string keys: summary, methodology, top_stocks,
        metric_insight, weights_used, final_statement
        """
        if not results:
            return RankingExplanationEngine._empty_response()

        return {
            "summary": RankingExplanationEngine._summary(
                results, metric=metric, order=order,
                exchange=exchange, horizon_years=horizon_years,
            ),
            "methodology": RankingExplanationEngine._methodology(
                metric=metric, order=order
            ),
            "top_stocks": RankingExplanationEngine._top_stocks_table(results),
            "metric_insight": RankingExplanationEngine._metric_insight(metric),
            "weights_used": RankingExplanationEngine._weights_section(weights),
            "final_statement": RankingExplanationEngine._final_statement(
                results, metric=metric, order=order
            ),
        }

    # -----------------------------------------------------------------------
    #  Section builders
    # -----------------------------------------------------------------------

    @staticmethod
    def _summary(
        results: List[Dict],
        *,
        metric: str,
        order: str,
        exchange: str,
        horizon_years: int,
    ) -> str:
        count = len(results)
        top = results[0]
        ticker = getattr(top, "ticker", None) or top.get("ticker", "N/A")
        label = "highest" if order == "best" else "lowest"
        metric_display = metric.upper().replace("_", " ")
        return (
            f"Screened {count} {exchange} stock{'s' if count != 1 else ''} "
            f"over a {horizon_years}-year horizon ranked by {metric_display}. "
            f"{ticker} leads with the {label} {metric_display} in this selection."
        )

    @staticmethod
    def _methodology(*, metric: str, order: str) -> str:
        direction = "descending" if order == "best" else "ascending"
        if metric == "score":
            return (
                "Stocks were scored using a composite weighted metric across multiple "
                f"financial criteria, then ranked in {direction} order. "
                "Weights are auto-normalised so they always sum to 100%."
            )
        metric_display = metric.upper().replace("_", " ")
        return (
            f"Stocks were ranked by {metric_display} in {direction} order. "
            "All metrics are computed from adjusted closing prices over the "
            "selected horizon window."
        )

    @staticmethod
    def _top_stocks_table(results: List[Dict]) -> str:
        top_n = results[:10]
        lines = ["Rank   Ticker       Value"]
        lines.append("-" * 36)
        for row in top_n:
            rank_label = getattr(row, "rank_label", None) or row.get("rank_label", "")
            ticker = getattr(row, "ticker", None) or row.get("ticker", "")
            display = getattr(row, "display_value", None) or row.get("display_value", "")
            lines.append(f"{rank_label:<6} {ticker:<12} {display}")
        return "\n".join(lines)

    @staticmethod
    def _metric_insight(metric: str) -> str:
        return RankingExplanationEngine.METRIC_DESCRIPTIONS.get(
            metric,
            f"Ranking by {metric.upper().replace('_', ' ')}.",
        )

    @staticmethod
    def _weights_section(weights: Optional[Dict[str, float]]) -> str:
        if not weights:
            return ""
        lines = ["Active weight configuration (normalised):"]
        lines.append("-" * 40)
        for criterion, w in sorted(weights.items(), key=lambda x: -x[1]):
            pct = round(w * 100, 1)
            bar = "█" * int(pct // 5)
            lines.append(f"  {criterion:<14} {pct:>5.1f}%  {bar}")
        dominant = max(weights, key=weights.get)
        lines.append("")
        lines.append(
            f"'{dominant}' has the highest weight and drives the ranking most strongly."
        )
        return "\n".join(lines)

    @staticmethod
    def _final_statement(
        results: List[Dict],
        *,
        metric: str,
        order: str,
    ) -> str:
        top = results[0]
        ticker = getattr(top, "ticker", None) or top.get("ticker", "N/A")
        display = getattr(top, "display_value", None) or top.get("display_value", "")
        label = "excels" if order == "best" else "ranks lowest"
        metric_display = metric.upper().replace("_", " ")
        return (
            f"{ticker} {label} on {metric_display} ({display}) within this screen. "
            "Use this ranking alongside portfolio diversification "
            "and risk analysis before making investment decisions."
        )

    # -----------------------------------------------------------------------
    #  Edge-case response
    # -----------------------------------------------------------------------

    @staticmethod
    def _empty_response() -> Dict[str, str]:
        return {
            "summary":         "No ranking results available.",
            "methodology":     "",
            "top_stocks":      "",
            "metric_insight":  "",
            "weights_used":    "",
            "final_statement": "",
        }
