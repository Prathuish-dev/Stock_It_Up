from __future__ import annotations

from pydantic import BaseModel, Field


class RankingItem(BaseModel):
    rank: int
    rank_label: str
    ticker: str
    metric_value: float
    display_value: str
    cagr: float
    volatility: float
    sharpe: float


class RankingSelection(BaseModel):
    exchange: str
    metric: str
    order: str
    limit: int
    horizon_years: int


class RankingPagination(BaseModel):
    page: int
    total_pages: int
    total_results: int


class ScatterPoint(BaseModel):
    x: float
    y: float
    ticker: str


class BarChartData(BaseModel):
    labels: list[str]
    values: list[float]


class PieChartData(BaseModel):
    labels: list[str]
    values: list[float]


class RankingChartData(BaseModel):
    scatter: list[ScatterPoint] = Field(default_factory=list)
    cagrBar: BarChartData
    pie: PieChartData


class RankingResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    warning: str | None = None
    selected: RankingSelection
    metric_display: str
    results: list[RankingItem] = Field(default_factory=list)
    pagination: RankingPagination
    chart_data: RankingChartData
    execution_ms: float
    cache_hit: bool


class ErrorResponse(BaseModel):
    error: str
    status: int

