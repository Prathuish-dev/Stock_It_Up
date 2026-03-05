from __future__ import annotations

from django.shortcuts import redirect, render

from app.api.services.ranking_service import ranking_service
from chatbot.constants import DEFAULT_HORIZON_YEARS, METRIC_REGISTRY


def _base_context():
    return {
        "market_options": ranking_service.MARKET_OPTIONS,
        "metric_options": ranking_service.METRIC_OPTIONS,
        "order_options": ranking_service.ORDER_OPTIONS,
        "metric_registry": METRIC_REGISTRY,
    }


def home(request):
    return redirect("/dashboard")


def dashboard(request):
    context = _base_context()
    context["nse_count"] = len(ranking_service.loader.list_available("NSE"))
    context["bse_count"] = len(ranking_service.loader.list_available("BSE"))
    return render(request, "dashboard.html", context)


def ranking(request):
    exchange = request.GET.get("exchange", "NSE")
    metric = request.GET.get("metric", "cagr")
    order = request.GET.get("order", "best")
    limit = request.GET.get("limit", 10)
    horizon_years = request.GET.get("horizon_years", DEFAULT_HORIZON_YEARS)

    context = _base_context()
    context["error"] = None
    context["warning"] = None
    context["results"] = []
    context["selected"] = {
        "exchange": exchange,
        "metric": metric,
        "order": order,
        "limit": limit,
        "horizon_years": horizon_years,
    }
    context["pagination"] = {"page": 1, "total_pages": 1, "total_results": 0}
    context["chart_data_json"] = "{}"

    return render(request, "ranking.html", context)


def portfolio(request):
    context = _base_context()
    context["portfolio_defaults"] = {
        "exchange": "NSE",
        "tickers": "TCS INFY RELIANCE",
        "budget": 100000,
        "method": "proportional",
        "risk_profile": "MEDIUM",
        "horizon_years": 3,
    }
    context["portfolio_methods"] = ["proportional", "softmax", "risk_adjusted"]
    context["portfolio_risks"] = ["LOW", "MEDIUM", "HIGH"]
    return render(request, "portfolio.html", context)


def risk(request):
    context = _base_context()
    context["risk_defaults"] = {
        "exchange": "NSE",
        "tickers": "TCS INFY RELIANCE",
        "method": "proportional",
        "risk_profile": "MEDIUM",
        "horizon_years": 3,
        "num_simulations": 3000,
    }
    context["risk_methods"] = ["proportional", "softmax", "risk_adjusted"]
    context["risk_profiles"] = ["LOW", "MEDIUM", "HIGH"]
    return render(request, "risk.html", context)
