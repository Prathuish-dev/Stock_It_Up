from __future__ import annotations

import math
from typing import List, Dict, Optional

from chatbot.enums import RiskProfile, InvestmentHorizon, Exchange
from chatbot.session_context import SessionContext
from chatbot.data_loader import DataLoader
from chatbot.metrics_engine import MetricsEngine, ScoringEngine
from chatbot.explanation_engine import ExplanationEngine
from chatbot.constants import HORIZON_YEARS as _HORIZON_YEARS, METRIC_REGISTRY


class ResponseGenerator:
    """
    Builds human-readable bot messages.

    **Formatting-only** — all computation is delegated to
    :class:`MetricsEngine` (analytics) and :class:`DataLoader` (I/O).
    This class must not perform heavy calculations itself.
    """

    def __init__(self):
        self._loader = DataLoader()
        self._engine = MetricsEngine()

    # ------------------------------------------------------------------ #
    #  Conversational prompts
    # ------------------------------------------------------------------ #

    def greeting(self) -> str:
        return (
            "👋 Welcome to **Stock It Up** — your offline stock decision advisor!\n\n"
            "I'll guide you through a short questionnaire and then rank stocks using "
            "weighted multi-criteria analysis on real historical data.\n\n"
            "First, which **exchange** are you interested in?\n"
            "  • NSE  — National Stock Exchange (Nifty)\n"
            "  • BSE  — Bombay Stock Exchange (Sensex)"
        )

    def ask_exchange(self) -> str:
        return (
            "Which exchange? Type **NSE** or **BSE**."
        )

    def ask_budget(self) -> str:
        return (
            "What is your **investment budget**? "
            "(e.g. '50000', '50k', '₹1 lakh')"
        )

    def ask_risk(self) -> str:
        return (
            "What is your **risk tolerance**?\n"
            "  • low    — prefer safe, stable investments\n"
            "  • medium — balanced approach\n"
            "  • high   — willing to take significant risk for higher returns"
        )

    def ask_horizon(self) -> str:
        return (
            "What is your **investment time horizon**?\n"
            "  • short  — less than 1 year  (1-year price window)\n"
            "  • medium — 1 to 5 years      (3-year price window)\n"
            "  • long   — more than 5 years (7-year price window)"
        )

    def ask_weights(self, current: Dict[str, float]) -> str:
        defaults = ", ".join(f"{k}={v:.0%}" for k, v in current.items())
        return (
            f"Shall I use the **default criteria weights**?\n  [{defaults}]\n\n"
            "Type 'yes' to accept, or provide custom weights as four decimals "
            "that sum to 1 (e.g. '0.35 0.25 0.15 0.25').\n"
            "Order: return (CAGR), risk (volatility), volume, horizon_score"
        )

    def ask_stocks(self, exchange: Exchange) -> str:
        tickers = self._loader.list_available(exchange.value)
        sample = ", ".join(tickers[:20])
        total = len(tickers)
        return (
            f"Which **{exchange.value}** stocks should I evaluate?\n"
            f"  ({total} stocks available — sample: {sample} …)\n\n"
            "Enter tickers separated by commas or spaces (e.g. 'TCS INFY RELIANCE')."
        )

    def confirm_exchange(self, exchange: Exchange) -> str:
        return f"✅ Exchange: **{exchange.value}**. What is your investment budget?"

    def confirm_budget(self, amount: float) -> str:
        return f"✅ Budget set to **₹{amount:,.0f}**. What is your risk tolerance?"

    def confirm_risk(self, profile: RiskProfile) -> str:
        return f"✅ Risk profile: **{profile.value}**. What is your investment horizon?"

    def confirm_horizon(self, horizon: InvestmentHorizon) -> str:
        years = _HORIZON_YEARS[horizon]
        return (
            f"✅ Horizon: **{horizon.value}-term** ({years}-year data window). "
            "Let me check your criteria weights."
        )

    def confirm_weights_accepted(self) -> str:
        return "Default weights accepted."

    def unknown(self) -> str:
        return (
            "I didn't quite understand that. "
            "Type 'help' to see all supported commands."
        )

    def quit_message(self) -> str:
        return "Thanks for using Stock It Up. Goodbye!"

    def restart_message(self) -> str:
        return "Session restarted. Let's begin again.\n\n" + self.greeting()

    def follow_up(self) -> str:
        return (
            "\nWhat next?\n"
            "  * 'explain <TICKER>' -- full score breakdown\n"
            "  * 'search <query>'   -- find tickers by name\n"
            "  * 'list'             -- browse all companies\n"
            "  * 'restart'          -- new analysis\n"
            "  * 'help'             -- all commands\n"
            "  * 'exit'             -- quit"
        )

    # ------------------------------------------------------------------ #
    #  Discoverability responses
    # ------------------------------------------------------------------ #

    PAGE_SIZE = 50

    def list_companies(
        self,
        companies: list,
        exchange: str = "",
        page: int = 1,
    ) -> str:
        """
        Show a paginated slice of the company list.

        Parameters
        ----------
        companies : list[str]
            Full sorted list of tickers.
        exchange : str
            Label for the header.
        page : int
            1-indexed page number (50 tickers per page).
        """
        total = len(companies)
        if total == 0:
            return f"No companies found on {exchange}."

        pages = max(1, -(-total // self.PAGE_SIZE))   # ceiling division
        page  = max(1, min(page, pages))               # clamp to valid range
        start = (page - 1) * self.PAGE_SIZE
        end   = start + self.PAGE_SIZE
        slice_  = companies[start:end]

        lines = [
            f"--- {exchange} Companies  |  page {page}/{pages}  "
            f"({total} total) ---",
            ", ".join(slice_),
        ]
        if page < pages:
            lines.append(
                f"\nType 'list {exchange} page {page + 1}' for next page."
            )
        if page > 1:
            lines.append(
                f"Type 'list {exchange} page {page - 1}' for previous page."
            )
        return "\n".join(lines)

    def search_results(
        self,
        query: str,
        matches: list,
        exchange: str = "",
    ) -> str:
        """Render ticker search results with autocomplete suggestions."""
        if not matches:
            return (
                f"No tickers matching '{query}' found on {exchange}.\n"
                "Try a shorter prefix, e.g. 'search BAJ' for Bajaj stocks."
            )

        # Show at most 30 results to keep output readable
        shown  = matches[:30]
        lines  = [f"Search results for '{query}' on {exchange} "
                  f"({len(matches)} match{'es' if len(matches) != 1 else ''}):"]

        # Group: exact, prefix, other
        exact  = [t for t in shown if t == query.upper()]
        prefix = [t for t in shown if t.startswith(query.upper()) and t != query.upper()]
        others = [t for t in shown if not t.startswith(query.upper())]

        if exact:
            lines.append(f"  Exact  : {', '.join(exact)}")
        if prefix:
            lines.append(f"  Prefix : {', '.join(prefix)}")
        if others:
            lines.append(f"  Related: {', '.join(others)}")

        if len(matches) > 30:
            lines.append(f"  ... and {len(matches) - 30} more. Refine your query.")

        lines.append(
            "\nTo analyse a stock, add it to your ticker list when prompted."
        )
        return "\n".join(lines)

    def show_keywords(self) -> str:
        """Return the full command reference card."""
        return (
            "--- Supported Commands ---\n"
            "\nScreener (works any time):\n"
            "  top 10 NSE by cagr         Best CAGR on NSE\n"
            "  top 5 BSE by score         Weighted score ranking (BSE)\n"
            "  lowest 10 NSE by volatility Safest stocks (lowest volatility)\n"
            "  top 20 BSE by volume       Most liquid BSE stocks\n"
            "  Note: 'top N by volatility' = most volatile; use 'lowest' for safest\n"
            "\nDiscoverability:\n"
            "  list                  Show all companies (current exchange)\n"
            "  list NSE              Show all NSE companies\n"
            "  list BSE              Show all BSE companies\n"
            "  list NSE page 2       Page through the list\n"
            "  search <query>        Find tickers by prefix/name\n"
            "  exchanges             Show available exchanges\n"
            "\nAnalysis session:\n"
            "  NSE / BSE             Select exchange\n"
            "  <amount>              Set budget (e.g. 50000 or 50k)\n"
            "  low / medium / high   Set risk profile\n"
            "  short / medium / long Set investment horizon\n"
            "  yes                   Accept default weights\n"
            "  TCS INFY RELIANCE     Set stocks to analyse\n"
            "\nPost-analysis:\n"
            "  explain <TICKER>      Full score breakdown with formula\n"
            "\nSystem:\n"
            "  help / keywords       Show this reference card\n"
            "  restart               Start a new session\n"
            "  exit / quit           Exit the application"
        )

    def show_exchanges(self) -> str:
        """Return exchange overview."""
        nse_count = len(self._loader.list_available("NSE"))
        bse_count = len(self._loader.list_available("BSE"))
        return (
            "--- Available Exchanges ---\n\n"
            f"  NSE  National Stock Exchange of India   {nse_count:,} stocks\n"
            f"  BSE  Bombay Stock Exchange              {bse_count:,} stocks\n\n"
            "Type 'list NSE' or 'list BSE' to browse, "
            "or 'top 10 NSE by cagr' to screen the market."
        )

    # ------------------------------------------------------------------ #
    #  Screener results formatter
    # ------------------------------------------------------------------ #

    def format_screener_results(
        self,
        results: list,
        metric: str,
        exchange: str,
        limit: int,
        horizon_years: int,
        direction: str = "desc",
    ) -> str:
        """
        Render screener results as a compact ranked table.

        Parameters
        ----------
        results : list[dict]
            Output from ``ScreenerEngine.run()``.
        metric : str
            Internal metric identifier (e.g. ``"cagr"``).
        exchange : str
            Exchange label for the header.
        limit : int
            Requested number of results (used in header).
        horizon_years : int
            Data window used (shown in header).
        direction : str
            ``"desc"`` (top) or ``"asc"`` (lowest).
        """
        if not results:
            return (
                f"No results found for '{metric.upper()}' on {exchange}.\n"
                "This can happen if insufficient historical data is available.\n"
                "Try a shorter horizon or a different metric."
            )

        info         = METRIC_REGISTRY.get(metric, {})
        display_name = info.get("display", metric.upper())
        rank_word    = "Lowest" if direction == "asc" else "Top"
        plural_yr    = "year" if horizon_years == 1 else "years"

        sep   = "=" * 52
        lines = [
            sep,
            f"{rank_word} {len(results)} {exchange}  by {display_name}  "
            f"({horizon_years}-{plural_yr} horizon)",
            "-" * 52,
            f"{'#':<4} {'TICKER':<14} {display_name:>16}",
            "-" * 52,
        ]

        for i, r in enumerate(results, 1):
            lines.append(
                f"{i:<4} {r['ticker']:<14} {r['display_value']:>16}"
            )

        lines.append(sep)

        scanned = len(results)
        footer_parts = [
            f"{scanned} result{'s' if scanned != 1 else ''} shown.",
        ]
        if metric != "score":
            footer_parts.append("Type 'top N by score' for weighted ranking.")
        else:
            footer_parts.append("Type 'top N by cagr' for growth-only ranking.")
        if direction == "desc":
            footer_parts.append("Use 'lowest' instead of 'top' to reverse order.")

        lines.append("\n" + "  ".join(footer_parts))
        return "\n".join(lines)

    def explanation(
        self,
        results: List[Dict],
        ticker: str,
        risk_profile=None,
        investment_horizon=None,
    ) -> str:
        """Delegate to ExplanationEngine and render as CLI text."""
        try:
            exp = ExplanationEngine.explain(
                ticker=ticker,
                scoring_results=results,
                risk_profile=risk_profile,
                investment_horizon=investment_horizon,
            )
            return ExplanationEngine.format_for_cli(exp)
        except ValueError:
            return f"No result found for **{ticker.upper()}** in the last analysis."

    def ticker_not_found(self, exchange: str, ticker: str) -> str:
        return (
            f"⚠️  '{ticker}' was not found on **{exchange}**. "
            "Please check the ticker and try again."
        )

    def insufficient_data(self, ticker: str, detail: str) -> str:
        return f"⚠️  Skipping **{ticker}**: {detail}"

    # ------------------------------------------------------------------ #
    #  Core analysis — Load -> Filter -> Compute -> Score -> Format
    # ------------------------------------------------------------------ #

    def analyse(self, context: SessionContext) -> str:
        """
        Full pipeline: load -> compute metrics -> ScoringEngine -> format.
        ResponseGenerator is formatting-only; all math is delegated.
        """
        horizon_years = _HORIZON_YEARS.get(context.investment_horizon, 3)

        raw_metrics: Dict[str, dict] = {}
        errors: List[str] = []

        for ticker in context.stocks:
            try:
                df_full = self._loader.load_stock(context.exchange.value, ticker)
                df = MetricsEngine.filter_by_horizon(df_full, horizon_years)
                raw_metrics[ticker] = MetricsEngine.compute_all(df)
            except FileNotFoundError:
                errors.append(self.ticker_not_found(context.exchange.value, ticker))
            except ValueError as e:
                errors.append(self.insufficient_data(ticker, str(e)))

        if not raw_metrics:
            return (
                "\n".join(errors) + "\n\n"
                "No valid tickers to rank. Please try again with different symbols."
            )

        # ScoringEngine handles: weight validation, risk adjustment,
        # min-max normalisation, inverse metrics, and explainability storage
        scores = ScoringEngine.compute_weighted_scores(
            raw_metrics,
            context.weights,
            risk_profile=context.risk_profile,
        )
        context.results = scores

        output = []
        if errors:
            output.append("Skipped tickers:\n" + "\n".join(errors))
        output.append(self._format_table(scores, context.budget))
        return "\n".join(output)

    def _format_table(self, results: List[Dict], budget: Optional[float]) -> str:
        sep = "="*72
        lines = [
            sep,
            f"{'RK':<4} {'TICKER':<14} {'SCORE':>7}  "
            f"{'CAGR':>8}  {'VOLAT':>7}  {'PRICE (Rs.)':>12}  {'SHARES':>8}",
            "-"*72,
        ]

        for i, r in enumerate(results, 1):
            m = r["metrics"]
            cagr_str  = f"{m['cagr']*100:+.1f}%"
            volat_str = f"{m['volatility']*100:.1f}%"
            price_str = f"{m['latest_price']:,.1f}"

            if budget and m["latest_price"] > 0:
                shares = int(budget // m["latest_price"])
                shares_str = f"{shares:,}"
            else:
                shares_str = "-"

            lines.append(
                f"{i:<4} {r['ticker']:<14} {r['total_score']:>7.4f}  "
                f"{cagr_str:>8}  {volat_str:>7}  {price_str:>12}  {shares_str:>8}"
            )

        lines.append(sep)
        top = results[0]["ticker"]
        lines.append(f"\nTop pick: {top}")
        if budget:
            top_price  = results[0]["metrics"]["latest_price"]
            top_shares = int(budget // top_price) if top_price > 0 else 0
            cost       = top_shares * top_price
            remaining  = budget - cost
            lines.append(
                f"  With Rs.{budget:,.0f} you can buy {top_shares:,} shares "
                f"of {top} @ Rs.{top_price:,.1f}  "
                f"(cost: Rs.{cost:,.0f}, remaining: Rs.{remaining:,.0f})"
            )
        lines.append(
            "\nType 'explain <TICKER>' for full score formula, "
            "'restart' to re-run, or 'exit' to quit."
        )
        return "\n".join(lines)