"""
chatbot/screener_engine.py
--------------------------
Scans all tickers on an exchange, computes a single metric lazily,
and returns the top-N results ranked by that metric.

Design decisions
----------------
* **Metric cache (fast path)** — when a ``MetricCache`` instance is passed to
  ``run()``, results are served from the pre-built in-memory dict in O(n) time
  (no CSV reads at all).  This reduces a 4-minute NSE scan to < 1 second.
* **Live-scan fallback** — when no cache is available the original chunked
  CSV-reading paths are used transparently.
* **heapq.nlargest / nsmallest** — maintains only top-N results in memory.
* **Chunk-based progress** — dot per chunk; easy to extend with a callback.
* **Silent error skipping** — bad tickers skipped; screener always returns a
  result rather than crashing.
* **Score mode without risk_profile** — falls back to equal weights.
"""

from __future__ import annotations

import heapq
import sys
from typing import TYPE_CHECKING, Optional

from chatbot.constants import DEFAULT_SCREENER_WEIGHTS, METRIC_REGISTRY
from chatbot.metrics_engine import MetricsEngine, ScoringEngine

if TYPE_CHECKING:
    from chatbot.data_loader import DataLoader
    from chatbot.metric_cache import MetricCache


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


def _ordinal(n: int) -> str:
    """Return the ordinal string for a positive integer (1→'1st', 2→'2nd', …)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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
        cache: Optional["MetricCache"] = None,
    ) -> list[dict]:
        """
        Scan *exchange*, compute *metric*, return top-*limit* tickers.

        Parameters
        ----------
        cache : MetricCache, optional
            Pre-built metric cache.  When provided, results are served
            from the in-memory cache (< 1 second) instead of re-scanning
            all ticker CSV files (~4 minutes).

        All other parameters are unchanged from the original signature.
        """
        # ------------------------------------------------------------------
        # Fast path — serve directly from pre-built cache
        # ------------------------------------------------------------------
        if cache is not None:
            cached_metrics = cache.get_or_build(exchange, horizon_years)
            if metric == "score":
                return ScreenerEngine._run_score_mode_cached(
                    exchange, limit, direction, cached_metrics,
                    risk_profile, weights,
                )
            return ScreenerEngine._run_metric_mode_cached(
                exchange, metric, limit, direction, cached_metrics,
            )

        # ------------------------------------------------------------------
        # Slow fallback — live CSV scan
        # ------------------------------------------------------------------
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
        processed = 0

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
                processed += 1

            # Live progress dot per chunk
            sys.stdout.write(".")
            sys.stdout.flush()

            # Trim to top-N after every chunk to keep memory low
            if len(candidates) > limit * 2:
                candidates = heap_fn(limit, candidates, key=heap_key)

        sys.stdout.write(f" done ({processed} tickers scanned)\n")
        sys.stdout.flush()

        # Deterministic tie-breaking: secondary sort by ticker name ascending
        top = sorted(
            heap_fn(limit, candidates, key=heap_key),
            key=lambda x: (-x[0] if direction == "desc" else x[0], x[1]),
        )

        info = METRIC_REGISTRY.get(metric, {})
        return [
            {
                "rank":          i,
                "rank_label":    _ordinal(i),
                "ticker":        ticker,
                "value":         value,
                "metric":        metric,
                "display_value": _format_value(metric, value),
                "display_name":  info.get("display", metric.upper()),
            }
            for i, (value, ticker) in enumerate(top, 1)
        ]

    # ------------------------------------------------------------------
    # Cached fast paths (O(n) dict look-up — no CSV reads)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_metric_mode_cached(
        exchange: str,
        metric: str,
        limit: int,
        direction: str,
        cached_metrics: dict,
    ) -> list[dict]:
        """Serve a metric-mode screener query from the in-memory cache."""
        heap_fn  = heapq.nsmallest if direction == "asc" else heapq.nlargest
        heap_key = lambda item: item[0]  # noqa: E731

        candidates: list[tuple[float, str]] = []
        for ticker, row in cached_metrics.items():
            value = row.get(metric)
            if value is None:
                continue
            candidates.append((value, ticker))

        # Deterministic tie-breaking: secondary sort by ticker name ascending
        top = sorted(
            heap_fn(limit, candidates, key=heap_key),
            key=lambda x: (-x[0] if direction == "desc" else x[0], x[1]),
        )
        info = METRIC_REGISTRY.get(metric, {})
        return [
            {
                "rank":          i,
                "rank_label":    _ordinal(i),
                "ticker":        ticker,
                "value":         value,
                "metric":        metric,
                "display_value": _format_value(metric, value),
                "display_name":  info.get("display", metric.upper()),
            }
            for i, (value, ticker) in enumerate(top, 1)
        ]

    @staticmethod
    def _run_score_mode_cached(
        exchange: str,
        limit: int,
        direction: str,
        cached_metrics: dict,
        risk_profile=None,
        weights: dict | None = None,
    ) -> list[dict]:
        """Serve a score-mode screener query from the in-memory cache."""
        if not cached_metrics:
            return []

        # Filter out tickers where any cached metric is None
        all_metrics = {
            t: {k: v for k, v in row.items() if v is not None}
            for t, row in cached_metrics.items()
            if any(v is not None for v in row.values())
        }

        if not all_metrics:
            return []

        effective_weights = weights or DEFAULT_SCREENER_WEIGHTS
        scored = ScoringEngine.compute_weighted_scores(
            all_metrics, effective_weights, risk_profile=risk_profile,
        )

        if direction == "asc":
            scored = list(reversed(scored))
        # Deterministic tie-breaking: secondary sort by ticker name
        scored = sorted(
            scored,
            key=lambda e: (
                e["total_score"] if direction == "asc" else -e["total_score"],
                e["ticker"],
            ),
        )

        results = []
        for i, entry in enumerate(scored[:limit], 1):
            results.append({
                "rank":             i,
                "rank_label":       _ordinal(i),
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
        processed = 0

        for i, ticker in enumerate(tickers):
            try:
                df_full = data_loader.load_stock(exchange, ticker)
                df      = MetricsEngine.filter_by_horizon(df_full, horizon_years)
                all_metrics[ticker] = MetricsEngine.compute_all(df)
                processed += 1
            except (FileNotFoundError, ValueError):
                continue

            # Progress dot every 100 tickers
            if (i + 1) % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        sys.stdout.write(f" done ({processed} tickers scanned)\n")
        sys.stdout.flush()

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
        # Deterministic tie-breaking: secondary sort by ticker name
        scored = sorted(
            scored,
            key=lambda e: (
                e["total_score"] if direction == "asc" else -e["total_score"],
                e["ticker"],
            ),
        )

        results = []
        for i, entry in enumerate(scored[:limit], 1):
            results.append({
                "rank":             i,
                "rank_label":       _ordinal(i),
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

    # ------------------------------------------------------------------
    # Positional query — fetch a single stock at a specific rank
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_position(
        exchange: str,
        metric: str,
        position: int,
        from_end: bool,
        horizon_years: int,
        data_loader: "DataLoader",
        risk_profile=None,
        weights: dict | None = None,
        cache: Optional["MetricCache"] = None,
    ) -> "dict | None":
        """
        Return the single stock dict at the requested ordinal position.

        Position is 1-based.  ``from_end=True`` counts from the worst end.

        Metric-aware direction
        ----------------------
        Uses ``METRIC_REGISTRY[metric]["higher_is_better"]`` to decide which
        scan order produces the correct 'best' / 'worst' ordering:

          * ``worst by cagr``       → asc  (low CAGR first)
          * ``worst by volatility`` → desc (high vol first — higher = worse)
          * ``best by volatility``  → asc  (low vol first — lower = safer)
        """
        info = METRIC_REGISTRY.get(metric, {})
        higher_is_better = info.get("higher_is_better", True)
        if higher_is_better is None:
            higher_is_better = True  # neutral metrics default to desc for 'best'

        if from_end:
            # worst end: for higher-is-better metrics scan ascending (lowest first)
            direction = "asc" if higher_is_better else "desc"
        else:
            # best end: for higher-is-better metrics scan descending (highest first)
            direction = "desc" if higher_is_better else "asc"

        fetch_limit = max(position + 10, 50)
        results = ScreenerEngine.run(
            exchange=exchange,
            metric=metric,
            limit=fetch_limit,
            horizon_years=horizon_years,
            direction=direction,
            data_loader=data_loader,
            risk_profile=risk_profile,
            weights=weights,
            cache=cache,
        )

        if not results or position > len(results):
            return None

        row = results[position - 1].copy()
        row["rank"]            = position
        row["rank_label"]      = _ordinal(position)
        row["from_end"]        = from_end
        row["direction_label"] = "from the bottom" if from_end else "from the top"
        return row
