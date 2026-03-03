from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.risk import RiskRequest, RiskResponse
from app.api.services.risk_service import risk_service

router = APIRouter(tags=["risk"])


@router.post("/api/risk", response_model=RiskResponse)
def analyze_risk(payload: RiskRequest) -> RiskResponse:
    return risk_service.build_payload(
        exchange=payload.exchange,
        tickers=payload.tickers,
        method=payload.method,
        risk_profile=payload.risk_profile,
        horizon_years=payload.horizon_years,
        num_simulations=payload.num_simulations,
    )

