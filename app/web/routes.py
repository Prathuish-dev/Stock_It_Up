from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import RedirectResponse

from app.api.services.ranking_service import ranking_service
from chatbot.constants import DEFAULT_HORIZON_YEARS, METRIC_REGISTRY

router = APIRouter(include_in_schema=False)


def _base_context() -> dict:
    return {
        "market_options": ranking_service.MARKET_OPTIONS,
        "metric_options": ranking_service.METRIC_OPTIONS,
        "order_options": ranking_service.ORDER_OPTIONS,
        "metric_registry": METRIC_REGISTRY,
    }


@router.get("/")
def home() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@router.get("/dashboard")
def dashboard(request: Request):
    templates = request.app.state.templates
    context = _base_context()
    context["request"] = request
    context["nse_count"] = len(ranking_service.loader.list_available("NSE"))
    context["bse_count"] = len(ranking_service.loader.list_available("BSE"))
    return templates.TemplateResponse("dashboard.html", context)


@router.get("/ranking")
def ranking(
    request: Request,
    exchange: str = Query(default="NSE"),
    metric: str = Query(default="cagr"),
    order: str = Query(default="best"),
    limit: int = Query(default=10),
    page: int = Query(default=1),
    horizon_years: int = Query(default=DEFAULT_HORIZON_YEARS),
):
    templates = request.app.state.templates
    context = _base_context()
    context["request"] = request
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

    try:
        payload = ranking_service.build_payload(
            exchange=exchange,
            metric=metric,
            order=order,
            limit=limit,
            page=page,
            horizon_years=horizon_years,
        )
        context["error"] = payload.error
        context["warning"] = payload.warning
        context["selected"] = payload.selected.model_dump()
        context["results"] = payload.results
        context["pagination"] = payload.pagination.model_dump()
        context["chart_data_json"] = payload.chart_data.model_dump_json()
    except Exception as exc:
        detail = getattr(exc, "detail", str(exc))
        context["error"] = str(detail)

    return templates.TemplateResponse("ranking.html", context)


@router.post("/ranking")
def ranking_submit(
    exchange: str = Form(default="NSE"),
    metric: str = Form(default="cagr"),
    order: str = Form(default="best"),
    limit: int = Form(default=10),
    horizon_years: int = Form(default=DEFAULT_HORIZON_YEARS),
):
    query = urlencode(
        {
            "exchange": exchange,
            "metric": metric,
            "order": order,
            "limit": ranking_service.safe_int(
                limit, default=10, low=1, high=ranking_service.MAX_LIMIT
            ),
            "horizon_years": ranking_service.safe_int(
                horizon_years, default=DEFAULT_HORIZON_YEARS, low=1, high=10
            ),
            "page": 1,
        }
    )
    return RedirectResponse(url=f"/ranking?{query}", status_code=303)


@router.get("/portfolio")
def portfolio(request: Request):
    templates = request.app.state.templates
    context = _base_context()
    context["request"] = request
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
    return templates.TemplateResponse("portfolio.html", context)


@router.get("/risk")
def risk(request: Request):
    templates = request.app.state.templates
    context = _base_context()
    context["request"] = request
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
    return templates.TemplateResponse("risk.html", context)
