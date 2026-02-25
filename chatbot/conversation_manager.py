import re
from chatbot.enums import ConversationState, Intent
from chatbot.session_context import SessionContext
from chatbot.intent_parser import IntentParser
from chatbot.response_generator import ResponseGenerator


class ConversationManager:
    """
    Orchestrates the multi-turn conversation.

    Maintains a :class:`SessionContext`, delegates intent recognition to
    :class:`IntentParser`, and delegates response building / analysis to
    :class:`ResponseGenerator`.  The external entry point is
    :meth:`handle_message`.
    """

    def __init__(self):
        self.context = SessionContext()
        self.parser = IntentParser()
        self.generator = ResponseGenerator()

    # ------------------------------------------------------------------ #
    #  Public entry point (called by main.py)
    # ------------------------------------------------------------------ #

    def handle_message(self, user_input: str) -> str:
        """
        Process one turn of user text and return the bot's reply.

        Global intents (quit / restart) are checked before routing to
        the state-specific handler so they always work regardless of stage.
        """
        text = user_input.strip()
        if not text:
            return "Please type something — I'm listening! 👂"

        intent = self.parser.parse_intent(text)

        # --- global overrides ---
        if intent == Intent.QUIT:
            self.context.state = ConversationState.DONE
            return self.generator.quit_message()

        if intent == Intent.RESTART:
            self.context.reset()
            self.context.state = ConversationState.COLLECT_BUDGET
            return self.generator.restart_message()

        # --- route to current state handler ---
        return self._dispatch(text, intent)

    # ------------------------------------------------------------------ #
    #  State dispatcher
    # ------------------------------------------------------------------ #

    def _dispatch(self, text: str, intent: Intent) -> str:
        state = self.context.state

        if state == ConversationState.GREETING:
            return self._handle_greeting(text, intent)
        if state == ConversationState.COLLECT_BUDGET:
            return self._handle_budget(text)
        if state == ConversationState.COLLECT_RISK:
            return self._handle_risk(text)
        if state == ConversationState.COLLECT_HORIZON:
            return self._handle_horizon(text)
        if state == ConversationState.COLLECT_CRITERIA_WEIGHTS:
            return self._handle_weights(text)
        if state == ConversationState.COLLECT_STOCKS:
            return self._handle_stocks(text)
        if state == ConversationState.SHOW_RESULTS:
            return self._handle_follow_up(text, intent)
        if state == ConversationState.DONE:
            return self.generator.quit_message()

        return self.generator.unknown()

    # ------------------------------------------------------------------ #
    #  Per-state handlers
    # ------------------------------------------------------------------ #

    def _handle_greeting(self, text: str, intent: Intent) -> str:
        self.context.state = ConversationState.COLLECT_BUDGET
        return self.generator.greeting()

    def _handle_budget(self, text: str) -> str:
        budget = self.parser.extract_budget(text)
        if budget is None or budget <= 0:
            return "Please enter a valid budget amount (e.g. '50000' or '10k')."
        self.context.budget = budget
        self.context.state = ConversationState.COLLECT_RISK
        return self.generator.confirm_budget(budget)

    def _handle_risk(self, text: str) -> str:
        risk = self.parser.extract_risk(text)
        if risk is None:
            return "Please specify your risk tolerance: low, medium, or high."
        self.context.risk_profile = risk
        self.context.state = ConversationState.COLLECT_HORIZON
        return self.generator.confirm_risk(risk)

    def _handle_horizon(self, text: str) -> str:
        horizon = self.parser.extract_horizon(text)
        if horizon is None:
            return "Please specify your investment horizon: short, medium, or long."
        self.context.investment_horizon = horizon
        self.context.state = ConversationState.COLLECT_CRITERIA_WEIGHTS
        return self.generator.ask_weights(self.context.weights)

    def _handle_weights(self, text: str) -> str:
        lower = text.lower().strip()

        # User accepts defaults
        if lower in ("yes", "y", "ok", "default", "accept", "sure"):
            self.context.state = ConversationState.COLLECT_STOCKS
            return (
                self.generator.confirm_weights_accepted()
                + "\n\n"
                + self.generator.ask_stocks()
            )

        # Try to parse four decimals
        numbers = re.findall(r"0?\.\d+", text)
        if len(numbers) == 4:
            w = [float(n) for n in numbers]
            if abs(sum(w) - 1.0) < 0.02:
                keys = ["return", "risk", "dividend_yield", "valuation"]
                self.context.weights = dict(zip(keys, w))
                self.context.state = ConversationState.COLLECT_STOCKS
                return (
                    "✅ Custom weights saved.\n\n"
                    + self.generator.ask_stocks()
                )
            return "The four weights must sum to 1.0. Please try again."

        return (
            "Type 'yes' to use defaults, or enter exactly four decimals "
            "summing to 1 (e.g. '0.30 0.25 0.20 0.25')."
        )

    def _handle_stocks(self, text: str) -> str:
        stocks = self.parser.extract_stocks(text)
        if not stocks:
            return (
                "I couldn't recognise any valid ticker symbols. "
                "Please enter uppercase tickers separated by commas or spaces."
            )
        self.context.stocks = stocks
        self.context.state = ConversationState.SHOW_RESULTS
        result_text = self.generator.analyse(self.context)
        return result_text + "\n" + self.generator.follow_up()

    def _handle_follow_up(self, text: str, intent: Intent) -> str:
        lower = text.lower().strip()

        # 'explain <TICKER>'
        explain_match = re.match(r"explain\s+([a-z0-9\.]+)", lower)
        if explain_match or intent == Intent.REQUEST_EXPLANATION:
            if explain_match:
                ticker = explain_match.group(1).upper()
            else:
                # fall back: grab last word
                ticker = lower.split()[-1].upper()
            if self.context.results:
                return self.generator.explanation(self.context.results, ticker)
            return "No results available yet. Please run an analysis first."

        return (
            "Type 'explain <TICKER>' for a score breakdown, "
            "'restart' for a new analysis, or 'exit' to quit."
        )

    # ------------------------------------------------------------------ #
    #  Initialisation helper (called by main.py before the loop)
    # ------------------------------------------------------------------ #

    def start(self) -> str:
        """Return the opening greeting without requiring any user input."""
        self.context.state = ConversationState.COLLECT_BUDGET
        return self.generator.greeting()