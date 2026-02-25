from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import pandas as pd


# Default dataset root — resolved relative to this file so it works regardless
# of which directory the user launches from.
_DEFAULT_BASE = Path(__file__).parent.parent / "comp_stock_data"

# Exchange folder name map
_EXCHANGE_DIR = {
    "NSE": "stock_data_NSE",
    "BSE": "stock_data_BSE",
}


class DataLoader:
    """
    Loads and caches per-ticker historical price data from the
    ``comp_stock_data`` directory tree.

    File layout::

        comp_stock_data/
            stock_data_NSE/<TICKER>/<TICKER>_<YEAR>.csv
            stock_data_BSE/<TICKER>/<TICKER>_<YEAR>.csv

    Each CSV must contain at minimum the columns:
    ``Date``, ``Adj Close``, ``Close``, ``Volume``.
    """

    def __init__(self, base_path: str | Path = _DEFAULT_BASE):
        self._base = Path(base_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_stock(self, exchange: str, ticker: str) -> pd.DataFrame:
        """
        Return a full-history DataFrame for *ticker* on *exchange*.

        Results are cached via ``@lru_cache`` on the private helper so that
        subsequent calls within the same Python process are free.

        Parameters
        ----------
        exchange : str
            ``"NSE"`` or ``"BSE"``
        ticker : str
            Uppercase ticker symbol, e.g. ``"TCS"``

        Raises
        ------
        FileNotFoundError
            If no CSV files are found for the given exchange/ticker.
        ValueError
            If the resulting DataFrame contains no rows after loading.
        """
        return _load_cached(str(self._base), exchange.upper(), ticker.upper())

    def list_available(self, exchange: str) -> list[str]:
        """
        Return a sorted list of ticker symbols available for *exchange*.
        """
        exchange = exchange.upper()
        folder = self._base / _EXCHANGE_DIR.get(exchange, f"stock_data_{exchange}")
        if not folder.exists():
            return []
        return sorted(p.name for p in folder.iterdir() if p.is_dir())

    def ticker_exists(self, exchange: str, ticker: str) -> bool:
        """Return True if at least one CSV file exists for this ticker."""
        exchange = exchange.upper()
        ticker = ticker.upper()
        folder = (
            self._base
            / _EXCHANGE_DIR.get(exchange, f"stock_data_{exchange}")
            / ticker
        )
        return bool(list(folder.glob(f"{ticker}_*.csv"))) if folder.exists() else False


# ------------------------------------------------------------------
# Module-level cached loader (instance-agnostic — only base+exchange+ticker
# matter for cache key, which are all plain strings).
# ------------------------------------------------------------------

@lru_cache(maxsize=128)
def _load_cached(base: str, exchange: str, ticker: str) -> pd.DataFrame:
    """
    Internal cached implementation.  ``base``, ``exchange``, and ``ticker``
    are all strings so they are hashable and work with ``lru_cache``.
    """
    exchange_dir = _EXCHANGE_DIR.get(exchange, f"stock_data_{exchange}")
    ticker_dir = Path(base) / exchange_dir / ticker

    if not ticker_dir.exists():
        raise FileNotFoundError(
            f"Ticker '{ticker}' not found on {exchange}. "
            f"Expected directory: {ticker_dir}"
        )

    pattern = str(ticker_dir / f"{ticker}_*.csv")
    csv_files = sorted(glob.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found for {exchange}:{ticker} in {ticker_dir}"
        )

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            # Skip completely empty (header-only) files
            if len(df) > 0:
                frames.append(df)
        except Exception:
            continue  # Silently skip corrupt files

    if not frames:
        raise ValueError(
            f"All CSV files for {exchange}:{ticker} are empty or unreadable."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("Date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Ensure required columns exist
    required = {"Date", "Adj Close", "Volume"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(
            f"CSV for {exchange}:{ticker} is missing columns: {missing}"
        )

    return combined
