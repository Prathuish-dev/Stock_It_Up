from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.schemas.ticker import TickerSearchResponse
from chatbot.data_loader import DataLoader

router = APIRouter(tags=["ticker"])
_loader = DataLoader()
_allowed_exchanges = {"NSE", "BSE"}


@router.get("/api/tickers/search", response_model=TickerSearchResponse)
def search_tickers(
    exchange: str = Query(default="NSE"),
    q: str = Query(default=""),
    limit: int = Query(default=12),
) -> TickerSearchResponse:
    ex = exchange.upper().strip()
    if ex not in _allowed_exchanges:
        raise HTTPException(status_code=400, detail=f"Invalid market '{exchange}'.")

    safe_limit = max(1, min(int(limit), 30))
    query = q.strip().upper()

    if query:
        items = _loader.search_tickers(ex, query)[:safe_limit]
    else:
        items = _loader.list_available(ex)[:safe_limit]

    return TickerSearchResponse(exchange=ex, query=query, items=items)

