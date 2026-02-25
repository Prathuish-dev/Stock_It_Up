import unittest
from chatbot.intent_parser import IntentParser
from chatbot.enums import Intent, RiskProfile, InvestmentHorizon


class TestIntentParser(unittest.TestCase):
    def setUp(self):
        self.parser = IntentParser()

    def test_quit_intent(self):
        self.assertEqual(self.parser.parse_intent("exit"), Intent.QUIT)
        self.assertEqual(self.parser.parse_intent("quit"), Intent.QUIT)

    def test_budget_extraction(self):
        self.assertAlmostEqual(self.parser.extract_budget("50k"), 50000)
        self.assertAlmostEqual(self.parser.extract_budget("1.5m"), 1500000)
        self.assertAlmostEqual(self.parser.extract_budget("₹1,00,000"), 100000)

    def test_risk_extraction(self):
        self.assertEqual(self.parser.extract_risk("I prefer low risk"), RiskProfile.LOW)
        self.assertEqual(self.parser.extract_risk("high"), RiskProfile.HIGH)
        self.assertEqual(self.parser.extract_risk("moderate"), RiskProfile.MEDIUM)

    def test_horizon_extraction(self):
        self.assertEqual(self.parser.extract_horizon("long term"), InvestmentHorizon.LONG)
        self.assertEqual(self.parser.extract_horizon("short"), InvestmentHorizon.SHORT)

    def test_stock_extraction(self):
        stocks = self.parser.extract_stocks("TCS, INFY, RELIANCE")
        self.assertEqual(stocks, ["TCS", "INFY", "RELIANCE"])


if __name__ == "__main__":
    unittest.main()