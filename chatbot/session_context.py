from dataclasses import dataclass, field
from typing import Optional, List, Dict
from chatbot.enums import ConversationState, RiskProfile, InvestmentHorizon


@dataclass
class SessionContext:
    """
    Holds all state gathered during a single conversation session.
    Tracks the conversation stage and any data collected from the user.
    """
    state: ConversationState = ConversationState.GREETING

    # User profile inputs
    budget: Optional[float] = None
    risk_profile: Optional[RiskProfile] = None
    investment_horizon: Optional[InvestmentHorizon] = None

    # Criteria weights (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "return": 0.30,
        "risk": 0.25,
        "dividend_yield": 0.20,
        "valuation": 0.25,
    })

    # Stocks to evaluate
    stocks: List[str] = field(default_factory=list)

    # Last analysis results
    results: Optional[List[Dict]] = None

    def is_complete(self) -> bool:
        """Return True when the conversation has fully finished."""
        return self.state == ConversationState.DONE

    def reset(self):
        """Reset the session to its initial state."""
        self.__init__()