from __future__ import annotations

from pydantic import BaseModel, Field


class PortfolioRequest(BaseModel):
    exchange: str = "NSE"
    tickers: list[str] = Field(default_factory=list)
    budget: float = 100000.0
    method: str = "proportional"
    risk_profile: str = "MEDIUM"
    horizon_years: int = 3
    include_explanation: bool = True
    weights: dict[str, float] | None = None


class ExplanationSchema(BaseModel):
    summary: str
    allocation_table: str
    strategy_rationale: str
    risk_distribution: str
    risk_decomposition: str
    capital_distribution: str
    portfolio_risk: str
    monte_carlo: str | None = None
    final_statement: str


class PortfolioAllocationItem(BaseModel):
    ticker: str
    allocation: float
    total_score: float
    capital_amount: float
    cagr: float
    volatility: float
    sharpe: float
    risk_share: float = 0.0


class PortfolioSummary(BaseModel):
    portfolio_return: float
    portfolio_volatility: float
    portfolio_sharpe: float
    portfolio_mdd: float
    portfolio_sortino: float
    var_95: float
    cvar_95: float
    probability_of_loss: float


class PortfolioChartData(BaseModel):
    allocation_labels: list[str] = Field(default_factory=list)
    allocation_values: list[float] = Field(default_factory=list)
    capital_labels: list[str] = Field(default_factory=list)
    capital_values: list[float] = Field(default_factory=list)
    scatter_points: list[dict] = Field(default_factory=list)


class PortfolioResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    warning: str | None = None
    exchange: str
    method: str
    risk_profile: str
    horizon_years: int
    budget: float
    tickers: list[str]
    allocations: list[PortfolioAllocationItem] = Field(default_factory=list)
    summary: PortfolioSummary
    chart_data: PortfolioChartData
    explanation: ExplanationSchema | None = None
    execution_ms: float

