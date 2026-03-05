from __future__ import annotations

import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from app.api.services.chat_service import chat_service
from app.api.services.portfolio_service import portfolio_service
from app.api.services.ranking_service import ranking_service
from app.api.services.risk_service import risk_service
from chatbot.constants import DEFAULT_HORIZON_YEARS
from chatbot.data_loader import DataLoader

_loader = DataLoader()


def _error_response(exc: Exception, fallback_status: int = 500) -> JsonResponse:
    status = getattr(exc, "status_code", fallback_status)
    detail = str(getattr(exc, "detail", str(exc)))
    return JsonResponse({"ok": False, "error": detail, "status": status}, status=status)


def _json_body(request):
    try:
        return json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        return {}


@require_GET
def api_ranking(request):
    try:
        payload = ranking_service.build_payload(
            exchange=request.GET.get("exchange", "NSE"),
            metric=request.GET.get("metric", "cagr"),
            order=request.GET.get("order", "best"),
            limit=request.GET.get("limit", 10),
            page=request.GET.get("page", 1),
            horizon_years=request.GET.get("horizon_years", DEFAULT_HORIZON_YEARS),
        )
        return JsonResponse(payload.model_dump(), status=200)
    except Exception as exc:
        return _error_response(exc)


@csrf_exempt
@require_http_methods(["POST"])
def api_portfolio(request):
    body = _json_body(request)
    try:
        payload = portfolio_service.build_payload(
            exchange=body.get("exchange", "NSE"),
            tickers=body.get("tickers", []),
            budget=body.get("budget", 100000),
            method=body.get("method", "proportional"),
            risk_profile=body.get("risk_profile", "MEDIUM"),
            horizon_years=body.get("horizon_years", 3),
            include_explanation=body.get("include_explanation", True),
        )
        return JsonResponse(payload.model_dump(), status=200)
    except Exception as exc:
        return _error_response(exc)


@csrf_exempt
@require_http_methods(["POST"])
def api_risk(request):
    body = _json_body(request)
    try:
        payload = risk_service.build_payload(
            exchange=body.get("exchange", "NSE"),
            tickers=body.get("tickers", []),
            method=body.get("method", "proportional"),
            risk_profile=body.get("risk_profile", "MEDIUM"),
            horizon_years=body.get("horizon_years", 3),
            num_simulations=body.get("num_simulations", 3000),
            include_explanation=body.get("include_explanation", True),
        )
        return JsonResponse(payload.model_dump(), status=200)
    except Exception as exc:
        return _error_response(exc)


@csrf_exempt
@require_http_methods(["POST"])
def api_chat_start(request):
    body = _json_body(request)
    session_id = str(body.get("session_id", "")).strip()
    if not session_id:
        return JsonResponse({"ok": False, "error": "session_id is required", "status": 400}, status=400)
    return JsonResponse(chat_service.start(session_id), status=200)


@csrf_exempt
@require_http_methods(["POST"])
def api_chat_message(request):
    body = _json_body(request)
    session_id = str(body.get("session_id", "")).strip()
    message = str(body.get("message", "")).strip()
    if not session_id or not message:
        return JsonResponse({"ok": False, "error": "session_id and message are required", "status": 400}, status=400)
    return JsonResponse(chat_service.message(session_id, message), status=200)


@csrf_exempt
@require_http_methods(["POST"])
def api_chat_reset(request):
    body = _json_body(request)
    session_id = str(body.get("session_id", "")).strip()
    if not session_id:
        return JsonResponse({"ok": False, "error": "session_id is required", "status": 400}, status=400)
    return JsonResponse(chat_service.reset(session_id), status=200)


@require_GET
def api_ticker_search(request):
    exchange = str(request.GET.get("exchange", "NSE")).upper().strip()
    if exchange not in {"NSE", "BSE"}:
        return JsonResponse({"ok": False, "error": f"Invalid market '{exchange}'", "status": 400}, status=400)

    query = str(request.GET.get("q", "")).strip().upper()
    limit = max(1, min(int(request.GET.get("limit", 12)), 30))
    if query:
        items = _loader.search_tickers(exchange, query)[:limit]
    else:
        items = _loader.list_available(exchange)[:limit]
    return JsonResponse({"exchange": exchange, "query": query, "items": items}, status=200)

