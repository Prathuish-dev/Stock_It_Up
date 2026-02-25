from enum import Enum, auto


class ConversationState(Enum):
    """Tracks the current stage of the conversation flow."""
    GREETING = auto()
    COLLECT_EXCHANGE = auto()   # NEW — first step; required to validate tickers
    COLLECT_BUDGET = auto()
    COLLECT_RISK = auto()
    COLLECT_HORIZON = auto()
    COLLECT_CRITERIA_WEIGHTS = auto()
    COLLECT_STOCKS = auto()
    ANALYZING = auto()
    SHOW_RESULTS = auto()
    FOLLOW_UP = auto()
    DONE = auto()


class Intent(Enum):
    """User intent categories detected from input."""
    GREET = auto()
    PROVIDE_EXCHANGE = auto()   # NEW
    PROVIDE_BUDGET = auto()
    PROVIDE_RISK = auto()
    PROVIDE_HORIZON = auto()
    PROVIDE_WEIGHTS = auto()
    PROVIDE_STOCKS = auto()
    REQUEST_ANALYSIS = auto()
    REQUEST_EXPLANATION = auto()
    RESTART = auto()
    QUIT = auto()
    UNKNOWN = auto()


class RiskProfile(Enum):
    """Investor risk tolerance levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InvestmentHorizon(Enum):
    """Investment time horizon categories."""
    SHORT = "short"    # < 1 year
    MEDIUM = "medium"  # 1–5 years
    LONG = "long"      # > 5 years


class Exchange(Enum):
    """Stock exchange identifier."""
    NSE = "NSE"   # National Stock Exchange of India
    BSE = "BSE"   # Bombay Stock Exchange