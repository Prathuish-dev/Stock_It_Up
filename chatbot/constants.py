"""
chatbot/constants.py
--------------------
Business-logic constants shared across modules.

Placing these here keeps formatting layers (ResponseGenerator) and
processing layers (ConversationManager, ScreenerEngine) aligned on
a single source of truth without creating circular imports.
"""

from __future__ import annotations

from chatbot.enums import InvestmentHorizon


# ---------------------------------------------------------------------------
# Horizon → calendar years
# ---------------------------------------------------------------------------

HORIZON_YEARS: dict = {
    InvestmentHorizon.SHORT:  1,
    InvestmentHorizon.MEDIUM: 3,
    InvestmentHorizon.LONG:   7,
}

# Default horizon in years when no session context is available.
DEFAULT_HORIZON_YEARS: int = 3


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------
# Each entry drives:
#   - ResponseGenerator formatting (display name, unit, value sign)
#   - ScreenerEngine sort direction (higher_is_better)
#
# Keys must exactly match the string identifiers used in MetricsEngine
# and ScreenerEngine.
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, dict] = {
    "cagr": {
        "display":         "CAGR",
        "unit":            "%",
        "higher_is_better": True,
        "scale":           100.0,   # multiply raw float to display as percent
    },
    "volatility": {
        "display":         "Volatility",
        "unit":            "%",
        "higher_is_better": False,
        "scale":           100.0,
    },
    "avg_volume": {
        "display":         "Avg Volume",
        "unit":            "",
        "higher_is_better": True,
        "scale":           1.0,
    },
    "latest_price": {
        "display":         "Price",
        "unit":            "\u20b9",
        "higher_is_better": None,   # neutral — not used for ranking
        "scale":           1.0,
    },
    "score": {
        "display":         "Score",
        "unit":            "",
        "higher_is_better": True,
        "scale":           1.0,
    },
    "sharpe": {
        "display":         "Sharpe",
        "unit":            "",
        "higher_is_better": True,
        "scale":           1.0,
    },
    "max_drawdown": {
        "display":         "Max Drawdown",
        "unit":            "%",
        "higher_is_better": False,
        "scale":           100.0,   # show as percentage
    },
    "sortino": {
        "display":         "Sortino",
        "unit":            "",
        "higher_is_better": True,
        "scale":           1.0,
    },
}


# ---------------------------------------------------------------------------
# Default weights when risk_profile is None in score-mode screening
# ---------------------------------------------------------------------------

DEFAULT_SCREENER_WEIGHTS: dict[str, float] = {
    "return": 0.34,
    "risk":   0.33,
    "volume": 0.33,
}
