from __future__ import annotations

from pydantic import BaseModel, Field


class RiskRequest(BaseModel):
    exchange: str = "NSE"
    tickers: list[str] = Field(default_factory=list)
    method: str = "proportional"
    risk_profile: str = "MEDIUM"
    horizon_years: int = 3
    num_simulations: int = 3000


class RiskAllocationItem(BaseModel):
    ticker: str
    allocation: float
    cagr: float
    volatility: float
    sharpe: float
    risk_share: float


class RiskSummary(BaseModel):
    mean_return: float
    std_dev: float
    var_95: float
    cvar_95: float
    probability_of_loss: float
    portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    portfolio_mdd: float
    portfolio_sortino: float


class RiskChartData(BaseModel):
    histogram_labels: list[str] = Field(default_factory=list)
    histogram_values: list[float] = Field(default_factory=list)
    risk_labels: list[str] = Field(default_factory=list)
    risk_values: list[float] = Field(default_factory=list)
    scatter_assets: list[dict] = Field(default_factory=list)
    scatter_portfolio: dict = Field(default_factory=dict)


class RiskResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    warning: str | None = None
    exchange: str
    method: str
    risk_profile: str
    horizon_years: int
    num_simulations: int
    allocations: list[RiskAllocationItem] = Field(default_factory=list)
    summary: RiskSummary
    chart_data: RiskChartData
    execution_ms: float

