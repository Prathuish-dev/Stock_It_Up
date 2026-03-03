from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.schemas.ranking import RankingResponse
from app.api.services.ranking_service import ranking_service
from chatbot.constants import DEFAULT_HORIZON_YEARS

router = APIRouter(tags=["ranking"])


@router.get("/api/ranking", response_model=RankingResponse)
def get_ranking(
    exchange: str = Query(default="NSE"),
    metric: str = Query(default="cagr"),
    order: str = Query(default="best"),
    limit: int = Query(default=10),
    page: int = Query(default=1),
    horizon_years: int = Query(default=DEFAULT_HORIZON_YEARS),
) -> RankingResponse:
    return ranking_service.build_payload(
        exchange=exchange,
        metric=metric,
        order=order,
        limit=limit,
        page=page,
        horizon_years=horizon_years,
    )

