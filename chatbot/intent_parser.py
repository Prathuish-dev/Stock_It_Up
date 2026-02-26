import re
from typing import Optional, List
from chatbot.enums import Intent, RiskProfile, InvestmentHorizon, Exchange


# ---------------------------------------------------------------------------
# COMMAND_MAP — single source of truth for all keyword → intent mappings.
# Order matters: more specific phrases checked before single words.
# ---------------------------------------------------------------------------
COMMAND_MAP: dict[Intent, list[str]] = {
    Intent.QUIT:           ["exit", "quit", "bye", "goodbye", "stop", "q"],
    Intent.RESTART:        ["restart", "reset", "start over", "again", "new session"],
    Intent.SHOW_KEYWORDS:  ["help", "keywords", "commands", "what can you do", "supported"],
    Intent.SHOW_EXCHANGES: ["exchanges", "markets", "which exchange", "what exchanges"],
    Intent.LIST_COMPANIES: ["list companies", "show companies", "available stocks",
                            "list stocks", "show stocks", "list nse", "list bse",
                            "display companies", "list all", "companies", "list"],
    Intent.SEARCH_COMPANY: ["search", "find", "lookup", "look up"],
    Intent.REQUEST_EXPLANATION: ["explain", "why", "how", "detail", "breakdown"],
    Intent.REQUEST_ANALYSIS:    ["analyse", "analyze", "result", "rank", "recommend"],
}

_GREET_PHRASES = {"hi", "hello", "hey", "start"}

# Regex that identifies a screener command (must appear BEFORE budget extraction
# so "top 1 ..." is not mistaken for budget ₹1).
_SCREENER_RE = re.compile(
    r"\b(top|lowest|best|worst|safest)\b",
    re.IGNORECASE,
)


class IntentParser:
    """
    Parses raw user input into structured intents and extracted values.

    Detection priority (checked in order):
    1. COMMAND_MAP (global commands — quit, help, list, search …)
    2. Exchange / budget / risk / horizon / weight / stock value extractors
    3. UNKNOWN fallback
    """

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def parse_intent(self, text: str) -> Intent:
        """Return the most likely Intent for *text*."""
        normalized = text.strip().lower()

        # 1. Check COMMAND_MAP (most-specific phrases evaluated first)
        for intent, phrases in COMMAND_MAP.items():
            if any(p in normalized for p in phrases):
                return intent

        # 2. Single-word greeting
        if normalized in _GREET_PHRASES:
            return Intent.GREET

        # 3. Screener command — checked BEFORE budget/exchange extractors so that
        #    a number like "top 1" is not swallowed by extract_budget().
        if _SCREENER_RE.search(normalized):
            return Intent.SCREEN_TOP

        # 4. Value-providing intents (order matters — exchange before budget)
        if self.extract_exchange(text) is not None:
            return Intent.PROVIDE_EXCHANGE
        if self.extract_budget(text) is not None:
            return Intent.PROVIDE_BUDGET
        if self.extract_risk(text) is not None:
            return Intent.PROVIDE_RISK
        if self.extract_horizon(text) is not None:
            return Intent.PROVIDE_HORIZON
        if self._looks_like_weights(text):
            return Intent.PROVIDE_WEIGHTS
        if self._looks_like_stocks(text):
            return Intent.PROVIDE_STOCKS

        return Intent.UNKNOWN

    # ------------------------------------------------------------------ #
    #  Discoverability extractors
    # ------------------------------------------------------------------ #

    def extract_screener_params(self, text: str) -> dict:
        """
        Parse parameters from a SCREEN_TOP command.

        Returns
        -------
        dict with keys:
            ``limit``     : int           (default 10)
            ``exchange``  : Exchange | None
            ``metric``    : str           (default ``"cagr"``)
            ``direction`` : str           ``"desc"`` | ``"asc"``
        """
        lower = text.strip().lower()

        # --- limit: first standalone integer found, default 10 ---
        num_match = re.search(r"\b(\d+)\b", lower)
        limit = int(num_match.group(1)) if num_match else 10

        # --- exchange: reuse existing extractor ---
        exchange = self.extract_exchange(text)

        # --- metric: keyword matching with aliases ---
        _METRIC_ALIASES: dict[str, str] = {
            "growth":   "cagr",
            "return":   "cagr",
            "cagr":     "cagr",
            "safe":     "volatility",
            "safest":   "volatility",
            "volatile": "volatility",
            "volatility": "volatility",
            "score":    "score",
            "rating":   "score",
            "ranked":   "score",
        }
        metric = "cagr"  # default
        for keyword, mapped in _METRIC_ALIASES.items():
            if keyword in lower:
                metric = mapped
                break

        # --- direction: asc for "lowest / safest / worst", desc otherwise ---
        _ASC_TRIGGERS = re.compile(r"\b(lowest|safest|worst)\b", re.IGNORECASE)
        direction = "asc" if _ASC_TRIGGERS.search(text) else "desc"

        return {
            "limit":     limit,
            "exchange":  exchange,
            "metric":    metric,
            "direction": direction,
        }

    def extract_list_params(self, text: str) -> dict:
        """
        Parse parameters from a LIST_COMPANIES command.

        Returns
        -------
        dict with keys:
            ``exchange`` : Exchange | None
            ``page``     : int (1-indexed, default 1)
            ``query``    : str | None  (for 'list tcs' style search within list)
        """
        lower = text.strip().lower()

        # Exchange
        exchange = self.extract_exchange(text)

        # Page number: "list nse page 2"  or  "list nse 2"
        page = 1
        page_match = re.search(r"\bpage\s+(\d+)\b|\blistings?\s+(\d+)\b", lower)
        if not page_match:
            # Trailing bare integer: "list nse 3"
            bare_int = re.search(r"\b([2-9]\d*)\b", lower)
            if bare_int:
                page = int(bare_int.group(1))
        else:
            page = int(page_match.group(1) or page_match.group(2))

        return {"exchange": exchange, "page": max(1, page)}

    def extract_search_query(self, text: str) -> str:
        """
        Extract the search term from a SEARCH_COMPANY command.

        e.g. "search tcs" → "TCS",  "find bajaj" → "BAJAJ"
        """
        lower = text.strip().lower()
        # Remove trigger words
        for trigger in ["search", "find", "lookup", "look up"]:
            lower = lower.replace(trigger, "").strip()
        return lower.upper() if lower else ""

    # ------------------------------------------------------------------ #
    #  Standard value extractors
    # ------------------------------------------------------------------ #

    def extract_exchange(self, text: str) -> Optional[Exchange]:
        """Detect NSE or BSE from free text."""
        upper = text.strip().upper()
        if re.search(r"\bNSE\b", upper):
            return Exchange.NSE
        if re.search(r"\bBSE\b", upper):
            return Exchange.BSE
        lower = text.strip().lower()
        if any(w in lower for w in ("national stock exchange", "nifty")):
            return Exchange.NSE
        if any(w in lower for w in ("bombay stock exchange", "sensex")):
            return Exchange.BSE
        return None

    def extract_budget(self, text: str) -> Optional[float]:
        """Extract a numeric budget value (supports k/m/lakh suffixes)."""
        pattern = r"[\$\u20b9]?\s*(\d[\d,]*\.?\d*)\s*(k|m|lakh|lac)?"
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        raw = float(match.group(1).replace(",", ""))
        suffix = (match.group(2) or "").lower()
        if suffix == "k":
            raw *= 1_000
        elif suffix == "m":
            raw *= 1_000_000
        elif suffix in ("lakh", "lac"):
            raw *= 100_000
        return raw

    def extract_risk(self, text: str) -> Optional[RiskProfile]:
        """Extract a risk profile from the user's input."""
        lower = text.lower()
        if any(w in lower for w in ("low", "safe", "conservative", "minimal")):
            return RiskProfile.LOW
        if any(w in lower for w in ("high", "aggressive", "risky", "bold")):
            return RiskProfile.HIGH
        if any(w in lower for w in ("medium", "moderate", "balanced", "neutral")):
            return RiskProfile.MEDIUM
        return None

    def extract_horizon(self, text: str) -> Optional[InvestmentHorizon]:
        """Extract an investment time horizon."""
        lower = text.lower()
        if any(w in lower for w in ("short", "< 1", "less than 1", "month")):
            return InvestmentHorizon.SHORT
        if any(w in lower for w in ("long", "long term", "5 year", "10 year", "decade")):
            return InvestmentHorizon.LONG
        if any(w in lower for w in ("medium", "mid", "1-5", "few year")):
            return InvestmentHorizon.MEDIUM
        return None

    def extract_stocks(self, text: str) -> List[str]:
        """Parse a comma/space-separated list of ticker symbols."""
        tokens = re.split(r"[,\s]+", text.strip())
        return [
            t.upper() for t in tokens
            if re.match(r"^[A-Z0-9]{1,15}([-&][A-Z]{1,10})?$", t.upper())
            and len(t) >= 2
        ]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _looks_like_weights(self, text: str) -> bool:
        numbers = re.findall(r"0?\.\d+", text)
        if len(numbers) >= 2:
            total = sum(float(n) for n in numbers)
            return 0.98 <= total <= 1.02
        return False

    def _looks_like_stocks(self, text: str) -> bool:
        tokens = re.split(r"[,\s]+", text.strip())
        if not tokens or len(tokens) > 20:
            return False
        return all(
            re.match(r"^[A-Z0-9]{2,15}([-&][A-Z]{1,10})?$", t.upper())
            for t in tokens if t
        )