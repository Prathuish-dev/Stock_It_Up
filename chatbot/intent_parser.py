import re
from typing import Optional, List
from chatbot.enums import Intent, RiskProfile, InvestmentHorizon, Exchange


class IntentParser:
    """
    Parses raw user input into structured intents and extracted values.
    Uses rule-based pattern matching suitable for an offline CLI tool.
    """

    QUIT_PHRASES = {"exit", "quit", "bye", "goodbye", "stop", "q"}
    RESTART_PHRASES = {"restart", "reset", "start over", "again", "new session"}
    GREET_PHRASES = {"hi", "hello", "hey", "start", "help"}

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def parse_intent(self, text: str) -> Intent:
        """Return the most likely intent for *text*."""
        normalized = text.strip().lower()

        if normalized in self.QUIT_PHRASES:
            return Intent.QUIT
        if any(p in normalized for p in self.RESTART_PHRASES):
            return Intent.RESTART
        if normalized in self.GREET_PHRASES:
            return Intent.GREET
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
        if any(kw in normalized for kw in ("explain", "why", "how", "detail")):
            return Intent.REQUEST_EXPLANATION
        if any(kw in normalized for kw in ("analyse", "analyze", "result", "rank", "recommend")):
            return Intent.REQUEST_ANALYSIS

        return Intent.UNKNOWN

    # ------------------------------------------------------------------ #
    #  Value extractors
    # ------------------------------------------------------------------ #

    def extract_exchange(self, text: str) -> Optional[Exchange]:
        """Detect NSE or BSE from free text."""
        upper = text.strip().upper()
        # Exact or embedded match
        if re.search(r'\bNSE\b', upper):
            return Exchange.NSE
        if re.search(r'\bBSE\b', upper):
            return Exchange.BSE
        # Common aliases
        lower = text.strip().lower()
        if any(w in lower for w in ("national stock exchange", "nifty")):
            return Exchange.NSE
        if any(w in lower for w in ("bombay stock exchange", "sensex")):
            return Exchange.BSE
        return None

    def extract_budget(self, text: str) -> Optional[float]:
        """Extract a numeric budget value (supports k/m suffixes)."""
        pattern = r"[\$₹]?\s*(\d[\d,]*\.?\d*)\s*(k|m|lakh|lac)?"
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
        return [t.upper() for t in tokens if re.match(r"^[A-Z0-9]{1,15}([-&][A-Z]{1,10})?$", t.upper()) and len(t) >= 2]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _looks_like_weights(self, text: str) -> bool:
        """Heuristic: does the input contain multiple decimals that sum ~1?"""
        numbers = re.findall(r"0?\.\d+", text)
        if len(numbers) >= 2:
            total = sum(float(n) for n in numbers)
            return 0.98 <= total <= 1.02
        return False

    def _looks_like_stocks(self, text: str) -> bool:
        """Heuristic: all words look like plausible ticker symbols (≥2 chars, all caps/digits)."""
        tokens = re.split(r"[,\s]+", text.strip())
        if not tokens or len(tokens) > 20:
            return False
        return all(
            re.match(r"^[A-Z0-9]{2,15}([-&][A-Z]{1,10})?$", t.upper())
            for t in tokens if t
        )