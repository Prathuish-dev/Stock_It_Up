import re
from chatbot.enums import ConversationState, Intent, Exchange
from chatbot.session_context import SessionContext
from chatbot.intent_parser import IntentParser
from chatbot.response_generator import ResponseGenerator


class ConversationManager:
    """
    Orchestrates the multi-turn conversation via a finite-state machine.

    Updated flow (exchange moved to first step so ticker validation works)::

        GREETING
          → COLLECT_EXCHANGE
          → COLLECT_BUDGET
          → COLLECT_RISK
          → COLLECT_HORIZON
          → COLLECT_CRITERIA_WEIGHTS
          → COLLECT_STOCKS
          → SHOW_RESULTS  (loop: explain / restart / exit)
          → DONE
    """

    def __init__(self):
        self.context = SessionContext()
        self.parser = IntentParser()
        self.generator = ResponseGenerator()

    # ------------------------------------------------------------------ #
    #  Public entry point (called by main.py)
    # ------------------------------------------------------------------ #

    def handle_message(self, user_input: str) -> str:
        """Process one turn of user text and return the bot's reply."""
        text = user_input.strip()
        if not text:
            return "Please type something — I'm listening! 👂"

        intent = self.parser.parse_intent(text)

        # --- global overrides (always work regardless of state) ---
        if intent == Intent.QUIT:
            self.context.state = ConversationState.DONE
            return self.generator.quit_message()

        if intent == Intent.RESTART:
            self.context.reset()
            self.context.state = ConversationState.COLLECT_EXCHANGE
            return self.generator.restart_message()

        return self._dispatch(text, intent)

    def start(self) -> str:
        """Return the opening greeting without requiring user input."""
        self.context.state = ConversationState.COLLECT_EXCHANGE
        return self.generator.greeting()

    # ------------------------------------------------------------------ #
    #  State dispatcher
    # ------------------------------------------------------------------ #

    def _dispatch(self, text: str, intent: Intent) -> str:
        state = self.context.state

        handlers = {
            ConversationState.GREETING:               self._handle_greeting,
            ConversationState.COLLECT_EXCHANGE:       self._handle_exchange,
            ConversationState.COLLECT_BUDGET:         self._handle_budget,
            ConversationState.COLLECT_RISK:           self._handle_risk,
            ConversationState.COLLECT_HORIZON:        self._handle_horizon,
            ConversationState.COLLECT_CRITERIA_WEIGHTS: self._handle_weights,
            ConversationState.COLLECT_STOCKS:         self._handle_stocks,
            ConversationState.SHOW_RESULTS:           self._handle_follow_up,
            ConversationState.DONE:                   lambda t, i: self.generator.quit_message(),
        }

        handler = handlers.get(state)
        return handler(text, intent) if handler else self.generator.unknown()

    # ------------------------------------------------------------------ #
    #  Per-state handlers
    # ------------------------------------------------------------------ #

    def _handle_greeting(self, text: str, intent: Intent) -> str:
        self.context.state = ConversationState.COLLECT_EXCHANGE
        return self.generator.greeting()

    def _handle_exchange(self, text: str, intent: Intent) -> str:
        exchange = self.parser.extract_exchange(text)
        if exchange is None:
            return self.generator.ask_exchange()
        self.context.exchange = exchange
        self.context.state = ConversationState.COLLECT_BUDGET
        return self.generator.confirm_exchange(exchange)

    def _handle_budget(self, text: str, intent: Intent) -> str:
        budget = self.parser.extract_budget(text)
        if budget is None or budget <= 0:
            return "Please enter a valid budget amount (e.g. '50000' or '10k')."
        self.context.budget = budget
        self.context.state = ConversationState.COLLECT_RISK
        return self.generator.confirm_budget(budget)

    def _handle_risk(self, text: str, intent: Intent) -> str:
        risk = self.parser.extract_risk(text)
        if risk is None:
            return "Please specify your risk tolerance: **low**, **medium**, or **high**."
        self.context.risk_profile = risk
        self.context.state = ConversationState.COLLECT_HORIZON
        return self.generator.confirm_risk(risk)

    def _handle_horizon(self, text: str, intent: Intent) -> str:
        horizon = self.parser.extract_horizon(text)
        if horizon is None:
            return "Please specify your investment horizon: **short**, **medium**, or **long**."
        self.context.investment_horizon = horizon
        self.context.state = ConversationState.COLLECT_CRITERIA_WEIGHTS
        return self.generator.ask_weights(self.context.weights)

    def _handle_weights(self, text: str, intent: Intent) -> str:
        lower = text.lower().strip()

        if lower in ("yes", "y", "ok", "default", "accept", "sure"):
            self.context.state = ConversationState.COLLECT_STOCKS
            return (
                self.generator.confirm_weights_accepted()
                + "\n\n"
                + self.generator.ask_stocks(self.context.exchange)
            )

        numbers = re.findall(r"0?\.\d+", text)
        if len(numbers) == 4:
            w = [float(n) for n in numbers]
            if abs(sum(w) - 1.0) < 0.02:
                keys = ["return", "risk", "volume", "horizon_score"]
                self.context.weights = dict(zip(keys, w))
                self.context.state = ConversationState.COLLECT_STOCKS
                return (
                    "✅ Custom weights saved.\n\n"
                    + self.generator.ask_stocks(self.context.exchange)
                )
            return "The four weights must sum to 1.0. Please try again."

        return (
            "Type 'yes' to use defaults, or enter exactly four decimals "
            "summing to 1 (e.g. '0.30 0.25 0.25 0.20')."
        )

    def _handle_stocks(self, text: str, intent: Intent) -> str:
        stocks = self.parser.extract_stocks(text)
        if not stocks:
            return (
                "I couldn't recognise any valid ticker symbols. "
                "Please enter uppercase tickers separated by commas or spaces "
                f"(e.g. 'TCS INFY RELIANCE' for NSE)."
            )
        self.context.stocks = stocks
        self.context.state = ConversationState.SHOW_RESULTS
        print("\n⏳ Loading data and computing metrics… (this may take a few seconds)\n")
        result_text = self.generator.analyse(self.context)
        return result_text + "\n" + self.generator.follow_up()

    def _handle_follow_up(self, text: str, intent: Intent) -> str:
        lower = text.lower().strip()

        # 'explain <TICKER>'
        explain_match = re.match(r"explain\s+([a-z0-9\-&\.]+)", lower)
        if explain_match or intent == Intent.REQUEST_EXPLANATION:
            if explain_match:
                ticker = explain_match.group(1).upper()
            else:
                ticker = lower.split()[-1].upper()
            if self.context.results:
                return self.generator.explanation(
                    self.context.results,
                    ticker,
                    risk_profile=self.context.risk_profile,
                    investment_horizon=self.context.investment_horizon,
                )
            return "No results available yet. Please run an analysis first."

        return (
            "Type 'explain <TICKER>' for a score breakdown, "
            "'restart' for a new analysis, or 'exit' to quit."
        )