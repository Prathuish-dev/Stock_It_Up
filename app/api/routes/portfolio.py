from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.portfolio import PortfolioRequest, PortfolioResponse
from app.api.services.portfolio_service import portfolio_service

router = APIRouter(tags=["portfolio"])


@router.post("/api/portfolio", response_model=PortfolioResponse)
def analyze_portfolio(payload: PortfolioRequest) -> PortfolioResponse:
    return portfolio_service.build_payload(
        exchange=payload.exchange,
        tickers=payload.tickers,
        budget=payload.budget,
        method=payload.method,
        risk_profile=payload.risk_profile,
        horizon_years=payload.horizon_years,
    )

