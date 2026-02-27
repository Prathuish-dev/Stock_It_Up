"""
chatbot/config.py
-----------------
Shared financial configuration constants.

Keeping these separate from chatbot/constants.py (which holds UI/business-logic
constants) ensures a clean boundary: this file owns tunable financial parameters
that are independent of display or session logic.
"""

# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------
# Annual risk-free rate used as the baseline for Sharpe Ratio computation.
#
# Default: 6.0% — approximate India 10-year Government Securities yield.
# Update this when the macro environment changes, or make it user-overridable
# in a future release.

RISK_FREE_RATE: float = 0.06   # 6% p.a.

# ---------------------------------------------------------------------------
# Sharpe Ratio outlier clipping bounds
# ---------------------------------------------------------------------------
# Sharpe Ratio is theoretically unbounded. A single outlier stock with an
# extremely high Sharpe can compress all other normalised scores toward zero
# via min-max scaling, which distorts weighted rankings.
#
# Clipping to [-1.0, 3.0] before normalisation:
#   * Preserves meaningful differentiation across the typical range
#   * Prevents one extreme value from dominating the score vector
#   * Still treats Sharpe < 0 (below risk-free) as clearly inferior
#   * Allows genuinely excellent stocks (Sharpe ~2-3) to rank near the top

SHARPE_CLIP_MIN: float = -1.0
SHARPE_CLIP_MAX: float =  3.0
