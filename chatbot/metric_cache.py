"""
chatbot/metric_cache.py
-----------------------
Persistent metric cache for the screener.

Problem
-------
Scanning all 1,970 NSE (or ~5,000 BSE) tickers to compute CAGR / volatility /
Sharpe etc. takes 3-4 minutes because every ticker requires reading multiple
CSV files from disk.  The data changes at most daily (and users typically run
the tool many times per day), so caching these computed metrics is safe.

Design
------
* One JSON file per (exchange, horizon) pair stored in
  ``comp_stock_data/.cache/<exchange>_<horizon>y.json``.
* Cache key: SHA-256 digest of the sorted list of CSV filenames + their
  modification timestamps → any new or updated file invalidates the entry.
* ``MetricCache.get_or_build()`` returns a ``{ticker: {metric: float}}`` dict
  either instantly from disk or after a full build with live progress output.
* ``MetricCache.build()`` is also callable explicitly (e.g. via a
  ``rebuild cache`` chatbot command).

Thread / process safety
-----------------------
Builds are written atomically via a temp file + rename so a mid-build crash
never leaves a corrupt cache on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from chatbot.constants import DEFAULT_HORIZON_YEARS
from chatbot.metrics_engine import MetricsEngine

if TYPE_CHECKING:
    from chatbot.data_loader import DataLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXCHANGE_DIR = {
    "NSE": "stock_data_NSE",
    "BSE": "stock_data_BSE",
}

# Metrics stored per ticker in the cache
_CACHED_METRICS = ("cagr", "volatility", "avg_volume", "latest_price",
                   "sharpe", "max_drawdown", "sortino")


def _cache_dir(base: Path) -> Path:
    d = base / ".cache"
    d.mkdir(exist_ok=True)
    return d


def _cache_path(base: Path, exchange: str, horizon: int) -> Path:
    return _cache_dir(base) / f"{exchange.upper()}_{horizon}y.json"


def _fingerprint(base: Path, exchange: str) -> str:
    """
    SHA-256 of sorted (path, mtime) pairs for all CSVs in the exchange folder.
    A fingerprint mismatch means new/changed data → cache is stale.
    """
    exchange_dir = base / _EXCHANGE_DIR.get(exchange.upper(),
                                             f"stock_data_{exchange.upper()}")
    if not exchange_dir.exists():
        return ""

    entries = []
    for csv_path in sorted(exchange_dir.rglob("*.csv")):
        entries.append(f"{csv_path}:{csv_path.stat().st_mtime:.1f}")

    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# MetricCache
# ---------------------------------------------------------------------------

class MetricCache:
    """
    Persistent per-exchange metric cache stored as JSON on disk.

    Usage (normal path — transparent to callers)
    ----------------------------------------------
    ::

        cache = MetricCache(data_loader)
        metrics = cache.get_or_build("NSE", horizon_years=3)
        # metrics → {ticker: {metric: float, ...}, ...}

    Explicit rebuild (e.g. from a chatbot ``rebuild cache`` command)
    ----------------------------------------------------------------
    ::

        cache = MetricCache(data_loader)
        cache.build("NSE", horizon_years=3)
    """

    def __init__(self, data_loader: "DataLoader"):
        self._loader = data_loader
        self._base   = data_loader._base        # Path to comp_stock_data/

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_or_build(
        self,
        exchange: str,
        horizon_years: int = DEFAULT_HORIZON_YEARS,
    ) -> dict[str, dict]:
        """
        Return ``{ticker: {metric: float}}`` from cache if valid,
        otherwise build cache first (with progress output) then return.

        Parameters
        ----------
        exchange : str  — ``"NSE"`` or ``"BSE"``
        horizon_years : int — data window in years (e.g. 3)

        Returns
        -------
        dict[str, dict]
            Ticker → metric values.  Missing metrics default to ``None``.
        """
        cached = self._load(exchange, horizon_years)
        if cached is not None:
            return cached
        return self.build(exchange, horizon_years)

    def build(
        self,
        exchange: str,
        horizon_years: int = DEFAULT_HORIZON_YEARS,
    ) -> dict[str, dict]:
        """
        Scan all tickers for *exchange*, compute all cached metrics, and
        persist to disk.  Progress dots are printed every 100 tickers.

        Returns the freshly-built metrics dict.
        """
        exchange = exchange.upper()
        tickers  = self._loader.list_available(exchange)

        print(
            f"\n📦 Building metric cache for {exchange} "
            f"({len(tickers):,} tickers, {horizon_years}y horizon)…"
            f"\n   [Each dot = 100 tickers scanned — please wait]\n",
            flush=True,
        )

        t_start  = time.time()
        metrics: dict[str, dict] = {}
        ok = 0

        for i, ticker in enumerate(tickers, 1):
            try:
                df_full = self._loader.load_stock(exchange, ticker)
                df      = MetricsEngine.filter_by_horizon(df_full, horizon_years)
                row: dict = {}
                for m in _CACHED_METRICS:
                    try:
                        row[m] = MetricsEngine.compute_metric(df, m)
                    except Exception:
                        row[m] = None
                metrics[ticker] = row
                ok += 1
            except (FileNotFoundError, ValueError):
                continue

            if i % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        elapsed = time.time() - t_start
        sys.stdout.write(f"\n✅ Cached {ok:,} / {len(tickers):,} tickers in {elapsed:.0f}s\n\n")
        sys.stdout.flush()

        self._save(exchange, horizon_years, metrics)
        return metrics

    def is_valid(self, exchange: str, horizon_years: int = DEFAULT_HORIZON_YEARS) -> bool:
        """Return True if a valid, up-to-date cache file exists."""
        return self._load(exchange, horizon_years) is not None

    def invalidate(self, exchange: str, horizon_years: int = DEFAULT_HORIZON_YEARS) -> None:
        """Delete the cache file for this exchange / horizon."""
        p = _cache_path(self._base, exchange, horizon_years)
        if p.exists():
            p.unlink()

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _load(self, exchange: str, horizon_years: int) -> dict | None:
        """
        Load cache from disk if it exists and the fingerprint matches.
        Returns ``None`` if cache is missing or stale.
        """
        p = _cache_path(self._base, exchange.upper(), horizon_years)
        if not p.exists():
            return None

        try:
            with open(p, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None

        stored_fp  = payload.get("fingerprint", "")
        current_fp = _fingerprint(self._base, exchange)

        if stored_fp != current_fp:
            return None   # data on disk changed → stale

        return payload.get("metrics", {})

    def _save(self, exchange: str, horizon_years: int, metrics: dict) -> None:
        """Atomically write the cache file (temp → rename)."""
        p   = _cache_path(self._base, exchange.upper(), horizon_years)
        tmp = p.with_suffix(".tmp")

        payload = {
            "exchange":    exchange.upper(),
            "horizon":     horizon_years,
            "fingerprint": _fingerprint(self._base, exchange),
            "metrics":     metrics,
        }

        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, allow_nan=False)
            os.replace(tmp, p)   # atomic on POSIX and Windows
        except OSError as exc:
            # Non-fatal — screener will just fall back to live scan
            print(f"  ⚠  Cache write failed ({exc}); results not saved.", flush=True)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
