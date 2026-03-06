import unittest
from unittest.mock import patch, MagicMock
from django.conf import settings
from django.http import HttpRequest

if not settings.configured:
    settings.configure(DEFAULT_CHARSET='utf-8')

from django_api.views import api_ranking

class TestDjangoApiRanking(unittest.TestCase):
    @patch("django_api.views.ranking_service")
    def test_api_ranking_parses_weights(self, mock_ranking_service):
        mock_payload = MagicMock()
        mock_payload.model_dump.return_value = {"ok": True}
        mock_ranking_service.build_payload.return_value = mock_payload

        request = HttpRequest()
        request.method = "GET"
        # Simulate query parameters
        request.GET = {
            "exchange": "NSE",
            "metric": "score",
            "order": "best",
            "weight_return": "0.5",
            "weight_risk": "0.3",
            "weight_volume": "0.2",
            "weight_invalid": "abc"
        }

        response = api_ranking(request)
        
        self.assertEqual(response.status_code, 200)
        
        # Verify build_payload was called with the correct weights dictionary
        mock_ranking_service.build_payload.assert_called_once()
        _, kwargs = mock_ranking_service.build_payload.call_args
        
        expected_weights = {
            "return": 0.5,
            "risk": 0.3,
            "volume": 0.2
            # 'invalid' should be ignored because it's not a float
        }
        
        self.assertEqual(kwargs.get("weights"), expected_weights)

if __name__ == "__main__":
    unittest.main()
