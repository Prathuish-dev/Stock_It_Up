from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime

from chatbot.config import RISK_FREE_RATE, SHARPE_CLIP_MIN, SHARPE_CLIP_MAX


class MetricsEngine:
    """
    Computes financial metrics from a cleaned historical price DataFrame.

    All computation uses ``Adj Close`` to correctly account for dividends
    and splits.  Only static methods are exposed — no shared state.
    """

    # Minimum trading days required to produce reliable metrics.
    _MIN_ROWS = 252   # ~1 full trading year

    # ------------------------------------------------------------------
    # Horizon filter
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_horizon(df: pd.DataFrame, years: int) -> pd.DataFrame:
        """
        Return only the rows within the last *years* calendar years.

        Parameters
        ----------
        df : pd.DataFrame
            Full-history DataFrame with a ``Date`` column.
        years : int
            How many years back from the latest row to keep.

        Returns
        -------
        pd.DataFrame
            Filtered (and already sorted) subset.
        """
        if df.empty:
            return df

        latest = df["Date"].max()
        cutoff = latest.replace(year=latest.year - years)
        filtered = df[df["Date"] >= cutoff].copy()
        return filtered

    # ------------------------------------------------------------------
    # Metric calculators
    # ------------------------------------------------------------------

    @staticmethod
    def compute_cagr(df: pd.DataFrame) -> float:
        """
        Compound Annual Growth Rate computed on ``Adj Close``.

        Raises
        ------
        ValueError
            If fewer than ``_MIN_ROWS`` rows are present — prevents
            misleadingly high/low CAGR from sparse data.
        """
        if len(df) < MetricsEngine._MIN_ROWS:
            raise ValueError(
                f"Insufficient data: {len(df)} rows (minimum {MetricsEngine._MIN_ROWS} required). "
                "Try a longer horizon or choose a different ticker."
            )

        start_price = df["Adj Close"].iloc[0]
        end_price = df["Adj Close"].iloc[-1]

        if start_price <= 0:
            raise ValueError("Start price is zero or negative — data may be corrupt.")

        years = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25

        if years < 0.1:
            raise ValueError("Date range too short to compute CAGR meaningfully.")

        return (end_price / start_price) ** (1 / years) - 1

    @staticmethod
    def compute_volatility(df: pd.DataFrame) -> float:
        """
        Annualised volatility = std(daily returns) × √252.

        Daily returns use ``Adj Close``.

        Raises
        ------
        ValueError
            If fewer than ``_MIN_ROWS`` rows are present.
        """
        if len(df) < MetricsEngine._MIN_ROWS:
            raise ValueError(
                f"Insufficient data: {len(df)} rows (minimum {MetricsEngine._MIN_ROWS} required)."
            )

        daily_returns = df["Adj Close"].pct_change().dropna()
        return float(daily_returns.std() * np.sqrt(252))

    @staticmethod
    def compute_avg_volume(df: pd.DataFrame) -> float:
        """Average daily traded volume — used as a liquidity proxy."""
        return float(df["Volume"].mean()) if "Volume" in df.columns else 0.0

    @staticmethod
    def latest_price(df: pd.DataFrame) -> float:
        """Most recent available closing price (``Adj Close``)."""
        return float(df["Adj Close"].iloc[-1])

    # ------------------------------------------------------------------
    # Convenience: compute all metrics at once
    # ------------------------------------------------------------------

    @staticmethod
    def compute_all(df: pd.DataFrame) -> dict:
        """
        Compute and return all metrics as a single dict.

        Returns
        -------
        dict with keys:
            ``cagr``, ``volatility``, ``avg_volume``, ``latest_price``,
            ``sharpe``, ``max_drawdown``, ``sortino``
        """
        return {
            "cagr":         MetricsEngine.compute_cagr(df),
            "volatility":   MetricsEngine.compute_volatility(df),
            "avg_volume":   MetricsEngine.compute_avg_volume(df),
            "latest_price": MetricsEngine.latest_price(df),
            "sharpe":       MetricsEngine.compute_sharpe(df),
            "max_drawdown": MetricsEngine.compute_max_drawdown(df),
            "sortino":      MetricsEngine.compute_sortino(df),
        }

    @staticmethod
    def compute_metric(df: pd.DataFrame, metric_name: str) -> float:
        """
        Compute only the single requested metric — avoids full compute_all()
        overhead during screener scans.

        Valid metric names: ``cagr``, ``volatility``, ``avg_volume``,
        ``latest_price``, ``sharpe``, ``max_drawdown``, ``sortino``.
        """
        dispatch: dict = {
            "cagr":         MetricsEngine.compute_cagr,
            "volatility":   MetricsEngine.compute_volatility,
            "avg_volume":   MetricsEngine.compute_avg_volume,
            "latest_price": MetricsEngine.latest_price,
            "sharpe":       MetricsEngine.compute_sharpe,
            "max_drawdown": MetricsEngine.compute_max_drawdown,   # NEW
            "sortino":      MetricsEngine.compute_sortino,         # NEW
        }
        fn = dispatch.get(metric_name)
        if fn is None:
            raise ValueError(
                f"Unknown metric: '{metric_name}'. "
                f"Valid options: {list(dispatch.keys())}"
            )
        return fn(df)

    @staticmethod
    def compute_max_drawdown(df: pd.DataFrame) -> float:
        """
        Compute the maximum peak-to-trough drawdown over the horizon window.

        Formula::

            MDD = max_t( (peak_t - trough_t) / peak_t )

        Parameters
        ----------
        df : pd.DataFrame
            Pre-filtered horizon DataFrame (``Adj Close`` column required).

        Returns
        -------
        float
            Positive decimal in [0, 1].  E.g. ``0.25`` means a 25%
            peak-to-trough decline.  **Lower is better.**
            Returns ``0.0`` on empty input.
        """
        if df.empty or len(df) < 2:
            return 0.0

        prices       = df["Adj Close"]
        running_peak = prices.cummax()
        drawdowns    = (running_peak - prices) / running_peak
        return float(drawdowns.max())

    @staticmethod
    def compute_sortino(df: pd.DataFrame) -> float:
        """
        Compute the annualised Sortino Ratio.

        Unlike Sharpe, Sortino penalises **only downside** deviations below
        the daily risk-free hurdle, leaving upside volatility unpunished.

        Formula::

            daily_rf    = RISK_FREE_RATE / 252
            downside    = returns[returns < daily_rf]
            sigma_down  = sqrt( mean(downside**2) ) * sqrt(252)
            Sortino     = (CAGR - RISK_FREE_RATE) / sigma_down

        Parameters
        ----------
        df : pd.DataFrame
            **Must** be pre-filtered to the desired horizon before calling.

        Returns
        -------
        float
            Clipped to ``[SHARPE_CLIP_MIN, SHARPE_CLIP_MAX]`` (``[-1, 3]``)
            for consistency with Sharpe normalisation.
            Returns ``0.0`` when there are no below-threshold days or when
            downside deviation is zero.
        """
        if df.empty or len(df) < MetricsEngine._MIN_ROWS:
            return 0.0

        daily_rf = RISK_FREE_RATE / 252
        returns  = df["Adj Close"].pct_change().dropna()

        # Only returns that fall below the daily risk-free hurdle contribute
        downside = returns[returns < daily_rf]
        if len(downside) == 0:
            return 0.0

        # Annualise downside deviation (same unit as CAGR)
        downside_std = float(np.sqrt((downside ** 2).mean()) * np.sqrt(252))
        if downside_std == 0.0:
            return 0.0

        cagr        = MetricsEngine.compute_cagr(df)
        raw_sortino = (cagr - RISK_FREE_RATE) / downside_std

        return float(max(SHARPE_CLIP_MIN, min(SHARPE_CLIP_MAX, raw_sortino)))

    @staticmethod
    def compute_sharpe(df: pd.DataFrame) -> float:
        """
        Compute the annualised Sharpe Ratio for the given (pre-filtered) DataFrame.

        Formula::

            Sharpe = (CAGR - RISK_FREE_RATE) / annualised_volatility

        Parameters
        ----------
        df : pd.DataFrame
            **Must** be pre-filtered to the desired horizon via
            ``MetricsEngine.filter_by_horizon()`` before calling this method.

        Returns
        -------
        float
            Sharpe Ratio clipped to [SHARPE_CLIP_MIN, SHARPE_CLIP_MAX]
            (default [-1.0, 3.0]).  Returns ``0.0`` when volatility is zero.
        """
        cagr = MetricsEngine.compute_cagr(df)
        vol  = MetricsEngine.compute_volatility(df)

        if vol == 0.0:
            return 0.0

        raw_sharpe = (cagr - RISK_FREE_RATE) / vol

        # Clip to prevent outlier domination in min-max normalisation
        return float(max(SHARPE_CLIP_MIN, min(SHARPE_CLIP_MAX, raw_sharpe)))




# ===========================================================================
# ScoringEngine — pure ranking logic, fully decoupled from data loading
# ===========================================================================

class ScoringEngine:
    """
    Converts raw per-ticker metrics into a normalised, weighted ranking.

    Design goals
    ------------
    * **Fair** — min-max normalisation removes scale dominance (e.g. volume
      in millions vs CAGR in decimals).
    * **Inverse-aware** — metrics where *lower is better* (volatility) are
      flipped so the scoring direction is consistent.
    * **Explainable** — every intermediate value is stored so the user can
      inspect exactly how each stock was scored.
    * **Adaptive** — weights are auto-adjusted for the user's risk profile
      before scoring begins.
    * **Robust** — weights are auto-normalised to sum to 1.0; edge cases
      (all identical values, single stock) are handled gracefully.

    Pipeline::

        raw metrics
            → validate_weights (normalise to sum=1)
            → adjust_for_risk_profile
            → normalize_metric (per criterion, with direction)
            → weighted_sum
            → sorted results with full breakdown
    """

    # Criteria definition: label → (metrics_key, higher_is_better)
    CRITERIA: dict = {
        "return":       ("cagr",         True),
        "risk":         ("volatility",   False),   # lower vol = better
        "volume":       ("avg_volume",   True),
        "sharpe":       ("sharpe",       True),
        "max_drawdown": ("max_drawdown", False),   # lower MDD = better
        "sortino":      ("sortino",      True),
    }

    # ------------------------------------------------------------------ #
    #  Step 5 — Weight validation + auto-normalisation
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate_weights(weights: dict) -> dict:
        """
        Ensure weights sum to 1.0.

        If the user provides proportional values (e.g. ``{0.4, 0.4, 0.4}``)
        they are auto-normalised.  Missing keys default to 0.

        Parameters
        ----------
        weights : dict
            Mapping of criterion label → raw weight value.

        Returns
        -------
        dict
            Normalised weights guaranteed to sum to 1.0.

        Raises
        ------
        ValueError
            If all weights are zero.
        """
        keys = list(ScoringEngine.CRITERIA.keys())
        # Fill missing keys with 0
        w = {k: float(weights.get(k, 0.0)) for k in keys}
        total = sum(w.values())
        if total == 0:
            raise ValueError(
                "All weights are zero — cannot produce a meaningful ranking."
            )
        return {k: v / total for k, v in w.items()}

    # ------------------------------------------------------------------ #
    #  Step 6 — Risk-profile weight adjustment
    # ------------------------------------------------------------------ #

    @staticmethod
    def adjust_for_risk_profile(weights: dict, risk_profile) -> dict:
        """
        Shift weights based on user's risk tolerance before scoring.

        * LOW  → penalise return weight, reward risk weight
        * HIGH → reward return weight, penalise risk weight
        * MEDIUM → unchanged

        Always re-validates (normalises) after adjustment.
        """
        from chatbot.enums import RiskProfile, InvestmentHorizon

        w = dict(weights)
        if risk_profile is not None:
            delta = 0.10
            if risk_profile.value == "low":
                w["risk"]   = w.get("risk",   0.0) + delta
                w["return"] = max(w.get("return", 0.0) - delta, 0.0)
            elif risk_profile.value == "high":
                w["return"] = w.get("return", 0.0) + delta
                w["risk"]   = max(w.get("risk",   0.0) - delta, 0.0)
        # Always re-normalise after adjustment
        return ScoringEngine.validate_weights(w)

    # ------------------------------------------------------------------ #
    #  Step 2 & 3 — Min-max normalisation with edge-case handling
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_metric(values: list, higher_is_better: bool = True) -> list:
        """
        Min-max normalise a list of raw values to the [0, 1] range.

        Parameters
        ----------
        values : list of float
        higher_is_better : bool
            If ``True``, higher raw values produce scores closer to 1.
            If ``False`` (e.g. volatility), the scale is inverted.

        Returns
        -------
        list of float
            Normalised scores in [0, 1], one per input value.

        Edge cases
        ----------
        * All identical → return ``[1.0, 1.0, …]`` (neutral, no penalisation)
        * Single value  → same as all-identical
        """
        min_v = min(values)
        max_v = max(values)

        # Step 3: edge case — all stocks equal on this metric
        if max_v == min_v:
            return [1.0 for _ in values]

        if higher_is_better:
            return [(v - min_v) / (max_v - min_v) for v in values]
        else:
            return [(max_v - v) / (max_v - min_v) for v in values]

    # ------------------------------------------------------------------ #
    #  Step 4 & 7 — Weighted scoring with full explainability
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_weighted_scores(
        metrics_dict: dict,
        weights: dict,
        risk_profile=None,
    ) -> list:
        """
        Main entry point.  Produces a sorted ranking with full breakdown.

        Parameters
        ----------
        metrics_dict : dict
            ``{ticker: {"cagr": float, "volatility": float,
                         "avg_volume": float, "latest_price": float}}``
        weights : dict
            Raw (unnormalised) criterion weights, e.g.
            ``{"return": 0.5, "risk": 0.3, "volume": 0.2}``
        risk_profile : RiskProfile or None
            When provided, weights are automatically adjusted before scoring.

        Returns
        -------
        list of dict, sorted descending by ``total_score``
            Each entry::

                {
                    "ticker": str,
                    "total_score": float,
                    "component_scores": {label: weighted_contribution},
                    "normalized":       {label: normalised_value_0_to_1},
                    "raw":              {label: raw_metric_value},
                    "weights_used":     {label: effective_weight},
                    "metrics":          full_metrics_dict,
                }
        """
        tickers = list(metrics_dict.keys())

        if not tickers:
            return []

        # --- Step 5: validate + normalise weights -----------------------
        validated = ScoringEngine.validate_weights(weights)

        # --- Step 6: adjust for risk profile ----------------------------
        effective_weights = ScoringEngine.adjust_for_risk_profile(
            validated, risk_profile
        )

        # --- Steps 2 & 3: normalise each criterion ----------------------
        norm_vectors: dict = {}
        raw_vectors:  dict = {}

        for label, (metric_key, higher_better) in ScoringEngine.CRITERIA.items():
            raw_vals = [metrics_dict[t].get(metric_key, 0.0) for t in tickers]
            raw_vectors[label]  = raw_vals
            norm_vectors[label] = ScoringEngine.normalize_metric(
                raw_vals, higher_better
            )

        # --- Step 4 & 7: weighted sum + explainability store -----------
        results = []
        for i, ticker in enumerate(tickers):
            component_scores: dict = {}
            normalized:       dict = {}
            raw:              dict = {}
            total = 0.0

            for label in ScoringEngine.CRITERIA:
                norm_val   = norm_vectors[label][i]
                raw_val    = raw_vectors[label][i]
                weight     = effective_weights[label]
                contribution = round(norm_val * weight, 5)

                component_scores[label] = contribution
                normalized[label]       = round(norm_val, 5)
                raw[label]              = raw_val
                total                  += contribution

            results.append({
                "ticker":           ticker,
                "total_score":      round(total, 5),
                "component_scores": component_scores,
                "normalized":       normalized,
                "raw":              raw,
                "weights_used":     {k: round(v, 4) for k, v in effective_weights.items()},
                "metrics":          metrics_dict[ticker],
            })

        # Sort descending by total score
        results.sort(key=lambda x: x["total_score"], reverse=True)
        return results

