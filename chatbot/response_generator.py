import math
from typing import List, Dict
from chatbot.enums import RiskProfile, InvestmentHorizon, ConversationState
from chatbot.session_context import SessionContext


# ---------------------------------------------------------------------------
# Simulated stock data (offline — no API calls needed)
# In a real deployment this would be fetched from an API or local CSV.
# ---------------------------------------------------------------------------
_STOCK_DATABASE: Dict[str, Dict[str, float]] = {
    # ticker: {annual_return_pct, beta, dividend_yield_pct, pe_ratio}
    "RELIANCE":  {"annual_return": 14.2, "beta": 1.1,  "dividend_yield": 0.4,  "pe_ratio": 22.5},
    "TCS":       {"annual_return": 18.5, "beta": 0.85, "dividend_yield": 1.2,  "pe_ratio": 28.0},
    "INFY":      {"annual_return": 16.0, "beta": 0.90, "dividend_yield": 1.8,  "pe_ratio": 24.0},
    "HDFCBANK":  {"annual_return": 12.5, "beta": 0.75, "dividend_yield": 1.1,  "pe_ratio": 18.5},
    "AAPL":      {"annual_return": 22.0, "beta": 1.20, "dividend_yield": 0.6,  "pe_ratio": 30.0},
    "MSFT":      {"annual_return": 25.0, "beta": 0.95, "dividend_yield": 0.8,  "pe_ratio": 35.0},
    "TSLA":      {"annual_return": 35.0, "beta": 2.0,  "dividend_yield": 0.0,  "pe_ratio": 60.0},
    "JNJ":       {"annual_return":  9.0, "beta": 0.60, "dividend_yield": 2.8,  "pe_ratio": 16.0},
    "KO":        {"annual_return":  8.0, "beta": 0.55, "dividend_yield": 3.1,  "pe_ratio": 22.0},
    "GOOGL":     {"annual_return": 20.0, "beta": 1.05, "dividend_yield": 0.0,  "pe_ratio": 26.0},
}


class ResponseGenerator:
    """
    Builds human-readable responses and performs weighted multi-criteria
    analysis to rank stocks.
    """

    # ------------------------------------------------------------------ #
    #  Conversational prompts
    # ------------------------------------------------------------------ #

    def greeting(self) -> str:
        return (
            "👋 Welcome to **Stock It Up** — your offline stock decision advisor!\n\n"
            "I'll guide you through a short questionnaire and then rank stocks using "
            "weighted multi-criteria analysis.\n\n"
            "Let's start: What is your **investment budget**? "
            "(e.g. '50000', '50k', '$10000')"
        )

    def ask_risk(self) -> str:
        return (
            "What is your **risk tolerance**?\n"
            "  • low  — prefer safe, stable investments\n"
            "  • medium — balanced approach\n"
            "  • high  — willing to take significant risk for higher returns"
        )

    def ask_horizon(self) -> str:
        return (
            "What is your **investment time horizon**?\n"
            "  • short  — less than 1 year\n"
            "  • medium — 1 to 5 years\n"
            "  • long   — more than 5 years"
        )

    def ask_weights(self, current: Dict[str, float]) -> str:
        defaults = ", ".join(f"{k}={v:.0%}" for k, v in current.items())
        return (
            f"Shall I use the **default criteria weights**?\n  [{defaults}]\n\n"
            "Type 'yes' to accept, or provide custom weights as four decimals "
            "that sum to 1 (e.g. '0.35 0.25 0.15 0.25').\n"
            "Order: return, risk, dividend_yield, valuation"
        )

    def ask_stocks(self) -> str:
        available = ", ".join(sorted(_STOCK_DATABASE.keys()))
        return (
            f"Which stocks should I evaluate?\n"
            f"Available tickers: {available}\n\n"
            "Enter them separated by commas or spaces (e.g. 'TCS INFY RELIANCE')."
        )

    def confirm_weights_accepted(self) -> str:
        return "✅ Default weights accepted. Now let's pick the stocks to analyse."

    def confirm_budget(self, amount: float) -> str:
        return f"✅ Budget set to **₹{amount:,.2f}**. What is your risk tolerance?"

    def confirm_risk(self, profile: RiskProfile) -> str:
        return f"✅ Risk profile: **{profile.value}**. What is your investment horizon?"

    def confirm_horizon(self, horizon: InvestmentHorizon) -> str:
        return f"✅ Horizon: **{horizon.value}-term**. Let me check your criteria weights."

    def unknown(self) -> str:
        return (
            "🤔 I didn't quite understand that. "
            "Please try again, or type 'help' for guidance."
        )

    def quit_message(self) -> str:
        return "👋 Thanks for using Stock It Up. Goodbye!"

    def restart_message(self) -> str:
        return "🔄 Session restarted. Let's begin again.\n\n" + self.greeting()

    def follow_up(self) -> str:
        return (
            "\nWould you like to:\n"
            "  • 'explain' — get details on how a specific stock was scored\n"
            "  • 'restart'  — analyse a new set of stocks\n"
            "  • 'exit'     — quit"
        )

    def explanation(self, results: List[Dict], ticker: str) -> str:
        match = next((r for r in results if r["ticker"] == ticker.upper()), None)
        if not match:
            return f"No result found for **{ticker.upper()}**."
        lines = [f"📊 Explanation for **{match['ticker']}**:"]
        for criterion, score in match["component_scores"].items():
            lines.append(f"  • {criterion:20s}: {score:.3f}")
        lines.append(f"  {'Weighted Total':20s}: {match['total_score']:.3f}")
        return "\n".join(lines)

    def not_in_database(self, ticker: str) -> str:
        available = ", ".join(sorted(_STOCK_DATABASE.keys()))
        return (
            f"⚠️  '{ticker}' is not in the local database.\n"
            f"Available tickers: {available}"
        )

    # ------------------------------------------------------------------ #
    #  Core analysis engine
    # ------------------------------------------------------------------ #

    def analyse(self, context: SessionContext) -> str:
        """
        Run weighted multi-criteria analysis over the chosen stocks and
        return a formatted ranking.
        """
        missing = [s for s in context.stocks if s not in _STOCK_DATABASE]
        if missing:
            return self.not_in_database(missing[0])

        stock_data = {s: _STOCK_DATABASE[s] for s in context.stocks}

        # Apply horizon/risk adjustments to weights
        weights = self._adjust_weights(context.weights, context.risk_profile,
                                        context.investment_horizon)

        # Normalize each criterion to [0, 1] across the candidate set
        scores = self._score_stocks(stock_data, weights)

        # Store results in context
        context.results = scores
        return self._format_results(scores, context.budget)

    def _adjust_weights(self, base: Dict[str, float],
                         risk: RiskProfile,
                         horizon: InvestmentHorizon) -> Dict[str, float]:
        """Tweak default weights based on risk/horizon preferences."""
        w = dict(base)
        if risk == RiskProfile.LOW:
            w["risk"] = min(w["risk"] + 0.05, 0.50)
            w["dividend_yield"] = min(w["dividend_yield"] + 0.05, 0.50)
            w["return"] = max(w["return"] - 0.05, 0.05)
            w["valuation"] = max(w["valuation"] - 0.05, 0.05)
        elif risk == RiskProfile.HIGH:
            w["return"] = min(w["return"] + 0.10, 0.60)
            w["risk"] = max(w["risk"] - 0.10, 0.05)
        if horizon == InvestmentHorizon.LONG:
            w["dividend_yield"] = min(w["dividend_yield"] + 0.05, 0.50)
            w["valuation"] = max(w["valuation"] - 0.05, 0.05)
        # Re-normalise so they sum to 1
        total = sum(w.values())
        return {k: v / total for k, v in w.items()}

    def _score_stocks(self, stock_data: Dict, weights: Dict) -> List[Dict]:
        """Min-max normalise and apply weights."""
        criteria = ["annual_return", "beta", "dividend_yield", "pe_ratio"]
        # For 'beta' and 'pe_ratio' lower is better; for the others, higher is better
        higher_is_better = {"annual_return", "dividend_yield"}

        # Compute min/max for normalisation
        ranges: Dict[str, tuple] = {}
        for c in criteria:
            values = [d[c] for d in stock_data.values()]
            ranges[c] = (min(values), max(values))

        results = []
        weight_map = {
            "annual_return": weights.get("return", 0.25),
            "beta":          weights.get("risk", 0.25),
            "dividend_yield": weights.get("dividend_yield", 0.25),
            "pe_ratio":      weights.get("valuation", 0.25),
        }

        for ticker, data in stock_data.items():
            component_scores: Dict[str, float] = {}
            total = 0.0
            for c in criteria:
                lo, hi = ranges[c]
                if hi == lo:
                    norm = 1.0
                else:
                    norm = (data[c] - lo) / (hi - lo)
                if c not in higher_is_better:
                    norm = 1.0 - norm   # invert so lower beta/PE → better score
                w = weight_map[c]
                component_scores[c] = round(norm * w, 4)
                total += component_scores[c]

            results.append({
                "ticker": ticker,
                "total_score": round(total, 4),
                "component_scores": component_scores,
                "data": data,
            })

        results.sort(key=lambda x: x["total_score"], reverse=True)
        return results

    def _format_results(self, results: List[Dict], budget: float) -> str:
        lines = [
            "=" * 52,
            f"{'RANK':<5} {'TICKER':<12} {'SCORE':>8}",
            "-" * 52,
        ]
        for i, r in enumerate(results, 1):
            ticker = r["ticker"]
            score = r["total_score"]
            annual_ret = r["data"]["annual_return"]
            div = r["data"]["dividend_yield"]
            lines.append(
                f"{i:<5} {ticker:<12} {score:>8.4f}"
                f"   (ret={annual_ret:.1f}%  div={div:.1f}%)"
            )
        lines.append("=" * 52)
        top = results[0]["ticker"]
        lines.append(f"\n🏆 Top pick: **{top}**")
        lines.append(
            "Type 'explain <TICKER>' to see score breakdown, "
            "'restart' for a new analysis, or 'exit' to quit."
        )
        return "\n".join(lines)