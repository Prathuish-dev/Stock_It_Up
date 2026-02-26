"""
chatbot/screener_engine.py
--------------------------
Scans all tickers on an exchange, computes a single metric lazily,
and returns the top-N results ranked by that metric.

Design decisions
----------------
* **Lazy/selective computation** — for raw metrics (CAGR, volatility, etc.)
  only the requested metric is computed per ticker.  ``compute_all()`` is
  only used in score mode where cross-stock normalisation requires all dims.
* **heapq.nlargest / nsmallest** — maintains only top-N results in memory
  during the full scan.  Avoids O(n log n) sort of the full ~2 000-stock list.
* **Chunk-based progress** — tickers processed in configurable chunks; easy
  to extend with a progress callback in future.
* **Silent error skipping** — FileNotFoundError / ValueError (insufficient
  data) are silently skipped; the screener should always return a result
  rather than crashing on a single bad ticker.
* **Score mode without risk_profile** — falls back to equal weights so the
  ranking is always deterministic regardless of session state.
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from chatbot.constants import DEFAULT_SCREENER_WEIGHTS, METRIC_REGISTRY
from chatbot.metrics_engine import MetricsEngine, ScoringEngine

if TYPE_CHECKING:
    from chatbot.data_loader import DataLoader


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _format_value(metric: str, raw: float) -> str:
    """Format a raw metric value using the METRIC_REGISTRY display rules."""
    info = METRIC_REGISTRY.get(metric, {})
    scale = info.get("scale", 1.0)
    unit  = info.get("unit", "")
    val   = raw * scale
    if unit == "%":
        return f"{val:+.2f}%"
    if unit == "₹":
        return f"₹{val:,.1f}"
    if val > 1_000_000:
        return f"{val/1_000_000:.2f}M"
    if val > 1_000:
        return f"{val/1_000:.1f}K"
    return f"{val:.4f}"


# ---------------------------------------------------------------------------
# ScreenerEngine
# ---------------------------------------------------------------------------

class ScreenerEngine:
    """
    Market screener — scans all tickers on an exchange and ranks by a metric.

    All heavy I/O and computation is delegated to ``DataLoader`` and
    ``MetricsEngine``; this class is pure orchestration.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def run(
        exchange: str,
        metric: str,
        limit: int,
        horizon_years: int,
        direction: str,
        data_loader: "DataLoader",
        risk_profile=None,
        weights: dict | None = None,
        chunk_size: int = 100,
    ) -> list[dict]:
        """
        Scan *exchange*, compute *metric*, return top-*limit* tickers.

        Parameters
        ----------
        exchange : str
            ``"NSE"`` or ``"BSE"``
        metric : str
            One of ``"cagr"``, ``"volatility"``, ``"avg_volume"``,
            ``"latest_price"``, or ``"score"``.
        limit : int
            Number of results to return.
        horizon_years : int
            Data window sliced from the most recent date.
        direction : str
            ``"desc"`` (highest first) or ``"asc"`` (lowest first).
        data_loader : DataLoader
            Injected I/O layer — makes the engine fully testable.
        risk_profile : RiskProfile | None
            Used only in score mode to adjust criterion weights.
            When ``None`` in score mode, equal weights are used.
        weights : dict | None
            Custom criterion weights for score mode.  If ``None``,
            ``DEFAULT_SCREENER_WEIGHTS`` or session weights are used.
        chunk_size : int
            Number of tickers to load per batch (memory control).

        Returns
        -------
        list[dict]
            Sorted list of result dicts, each containing at minimum:
            ``ticker``, ``value``, ``metric``, ``display_value``.
            Score-mode entries additionally contain ``total_score`` and
            ``component_scores``.
        """
        if metric == "score":
            return ScreenerEngine._run_score_mode(
                exchange, limit, horizon_years, direction,
                data_loader, risk_profile, weights,
            )
        return ScreenerEngine._run_metric_mode(
            exchange, metric, limit, horizon_years, direction,
            data_loader, chunk_size,
        )

    # ------------------------------------------------------------------
    # Metric mode (raw single-metric scan)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_metric_mode(
        exchange: str,
        metric: str,
        limit: int,
        horizon_years: int,
        direction: str,
        data_loader: "DataLoader",
        chunk_size: int,
    ) -> list[dict]:
        """Scan using heapq to maintain top-N without a full sort."""
        tickers = data_loader.list_available(exchange)
        if not tickers:
            return []

        # heapq.nlargest/nsmallest wants (value, ticker) tuples.
        # We invert the value for nsmallest so the heap logic stays uniform.
        heap_fn  = heapq.nsmallest if direction == "asc" else heapq.nlargest
        heap_key = lambda item: item[0]  # noqa: E731

        candidates: list[tuple[float, str]] = []

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            for ticker in chunk:
                try:
                    df_full = data_loader.load_stock(exchange, ticker)
                    df      = MetricsEngine.filter_by_horizon(df_full, horizon_years)
                    value   = MetricsEngine.compute_metric(df, metric)
                except (FileNotFoundError, ValueError):
                    continue  # silent skip — insufficient data / missing file

                candidates.append((value, ticker))

            # Trim to top-N after every chunk to keep memory low
            if len(candidates) > limit * 2:
                candidates = heap_fn(limit, candidates, key=heap_key)

        top = heap_fn(limit, candidates, key=heap_key)

        info = METRIC_REGISTRY.get(metric, {})
        return [
            {
                "ticker":        ticker,
                "value":         value,
                "metric":        metric,
                "display_value": _format_value(metric, value),
                "display_name":  info.get("display", metric.upper()),
            }
            for value, ticker in top
        ]

    # ------------------------------------------------------------------
    # Score mode (weighted multi-metric scan)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_score_mode(
        exchange: str,
        limit: int,
        horizon_years: int,
        direction: str,
        data_loader: "DataLoader",
        risk_profile=None,
        weights: dict | None = None,
    ) -> list[dict]:
        """
        Full metric scan → ScoringEngine normalisation → ranked results.

        Score mode must collect *all* tickers before normalising because
        min-max normalisation is cross-stock; you cannot score stocks
        independently.
        """
        tickers  = data_loader.list_available(exchange)
        all_metrics: dict[str, dict] = {}

        for ticker in tickers:
            try:
                df_full = data_loader.load_stock(exchange, ticker)
                df      = MetricsEngine.filter_by_horizon(df_full, horizon_years)
                all_metrics[ticker] = MetricsEngine.compute_all(df)
            except (FileNotFoundError, ValueError):
                continue

        if not all_metrics:
            return []

        # Fallback weights when no risk profile or custom weights are set
        effective_weights = weights or DEFAULT_SCREENER_WEIGHTS

        scored = ScoringEngine.compute_weighted_scores(
            all_metrics,
            effective_weights,
            risk_profile=risk_profile,   # None → equal weights (no adjustment)
        )

        # ScoringEngine already returns sorted descending by total_score.
        if direction == "asc":
            scored = list(reversed(scored))

        results = []
        for entry in scored[:limit]:
            results.append({
                "ticker":           entry["ticker"],
                "value":            entry["total_score"],
                "metric":           "score",
                "display_value":    f"{entry['total_score']:.4f}",
                "display_name":     "Score",
                "total_score":      entry["total_score"],
                "component_scores": entry["component_scores"],
                "weights_used":     entry["weights_used"],
                "metrics":          entry["metrics"],
            })
        return results
