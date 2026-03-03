from __future__ import annotations

import logging
import time

from fastapi import HTTPException

from app.api.schemas.portfolio import (
    PortfolioAllocationItem,
    PortfolioChartData,
    PortfolioResponse,
    PortfolioSummary,
)
from chatbot.constants import DEFAULT_SCREENER_WEIGHTS
from chatbot.data_loader import DataLoader
from chatbot.metrics_engine import MetricsEngine, ScoringEngine
from chatbot.portfolio_engine import PortfolioEngine

logger = logging.getLogger("stock_it_up.portfolio")


class PortfolioService:
    MARKET_OPTIONS = ["NSE", "BSE"]
    METHOD_OPTIONS = ["proportional", "softmax", "risk_adjusted"]
    RISK_OPTIONS = ["LOW", "MEDIUM", "HIGH"]
    MAX_TICKERS = 20

    def __init__(self) -> None:
        self.loader = DataLoader()

    @staticmethod
    def safe_int(raw: int | str, default: int, low: int, high: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    @staticmethod
    def safe_float(raw: float | str, default: float, low: float, high: float) -> float:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    def parse_tickers(self, raw_tickers: list[str] | str) -> list[str]:
        if isinstance(raw_tickers, str):
            tokens = [tok.strip().upper() for tok in raw_tickers.replace(",", " ").split()]
        else:
            tokens = [str(tok).strip().upper() for tok in raw_tickers]
        unique = []
        seen = set()
        for ticker in tokens:
            if ticker and ticker not in seen:
                unique.append(ticker)
                seen.add(ticker)
        return unique[: self.MAX_TICKERS]

    def _covariance_matrix(self, allocations: list[dict], rho: float = 0.2) -> list[list[float]]:
        vols = [max(0.0, float(a.get("volatility", 0.0))) for a in allocations]
        n = len(vols)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(vols[i] ** 2)
                else:
                    row.append(rho * vols[i] * vols[j])
            matrix.append(row)
        return matrix

    def _chart_data(self, allocations: list[PortfolioAllocationItem]) -> PortfolioChartData:
        return PortfolioChartData(
            allocation_labels=[a.ticker for a in allocations],
            allocation_values=[round(a.allocation * 100, 2) for a in allocations],
            capital_labels=[a.ticker for a in allocations],
            capital_values=[round(a.capital_amount, 2) for a in allocations],
            scatter_points=[
                {
                    "x": round(a.volatility * 100, 2),
                    "y": round(a.cagr * 100, 2),
                    "ticker": a.ticker,
                }
                for a in allocations
            ],
        )

    def build_payload(
        self,
        *,
        exchange: str,
        tickers: list[str] | str,
        budget: float,
        method: str,
        risk_profile: str,
        horizon_years: int,
    ) -> PortfolioResponse:
        started = time.perf_counter()
        exchange = exchange.upper()
        method = method.lower()
        risk_profile = risk_profile.upper()

        if exchange not in self.MARKET_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid market '{exchange}'.")
        if method not in self.METHOD_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid method '{method}'.")
        if risk_profile not in self.RISK_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid risk profile '{risk_profile}'.")

        clean_tickers = self.parse_tickers(tickers)
        if len(clean_tickers) < 2:
            raise HTTPException(status_code=400, detail="Provide at least 2 valid tickers.")

        budget_value = self.safe_float(budget, default=100000.0, low=1.0, high=1_000_000_000.0)
        years = self.safe_int(horizon_years, default=3, low=1, high=10)

        logger.info(
            "portfolio.query exchange=%s method=%s risk=%s tickers=%s budget=%s horizon_years=%s",
            exchange,
            method,
            risk_profile,
            ",".join(clean_tickers),
            budget_value,
            years,
        )

        metrics_dict: dict[str, dict] = {}
        missing: list[str] = []
        for ticker in clean_tickers:
            try:
                df = self.loader.load_stock(exchange, ticker)
                df = MetricsEngine.filter_by_horizon(df, years)
                metrics_dict[ticker] = MetricsEngine.compute_all(df)
            except Exception:
                missing.append(ticker)

        if len(metrics_dict) < 2:
            raise HTTPException(
                status_code=400,
                detail="Not enough valid tickers with data for the selected horizon.",
            )

        scored = ScoringEngine.compute_weighted_scores(metrics_dict, DEFAULT_SCREENER_WEIGHTS)

        scored_minimal: list[dict] = []
        for row in scored:
            ticker = row["ticker"]
            scored_minimal.append(
                {
                    "ticker": ticker,
                    "total_score": row["total_score"],
                    "cagr": metrics_dict[ticker].get("cagr", 0.0),
                    "volatility": metrics_dict[ticker].get("volatility", 0.0),
                    "sharpe": metrics_dict[ticker].get("sharpe", 0.0),
                    "max_drawdown": metrics_dict[ticker].get("max_drawdown", 0.0),
                    "sortino": metrics_dict[ticker].get("sortino", 0.0),
                    "expected_return": metrics_dict[ticker].get("cagr", 0.0),
                }
            )

        allocations = PortfolioEngine.allocate(
            scored_minimal,
            method=method,
            risk_profile=risk_profile,
        )

        allocation_map = {a["ticker"]: a for a in allocations}
        merged: list[dict] = []
        for row in scored_minimal:
            ticker = row["ticker"]
            alloc = allocation_map[ticker]
            merged.append(
                {
                    "ticker": ticker,
                    "allocation": alloc["allocation"],
                    "total_score": row["total_score"],
                    "cagr": row["cagr"],
                    "volatility": row["volatility"],
                    "sharpe": row["sharpe"],
                    "max_drawdown": row["max_drawdown"],
                    "sortino": row["sortino"],
                    "expected_return": row["expected_return"],
                }
            )

        alloc_total = sum(a["allocation"] for a in merged) or 1.0
        for a in merged:
            a["allocation"] = float(a["allocation"]) / alloc_total
        merged[-1]["allocation"] = 1.0 - sum(a["allocation"] for a in merged[:-1])

        merged = PortfolioEngine.compute_risk_decomposition(merged)
        merged = PortfolioEngine.allocate_capital(merged, budget_value)
        summary = PortfolioEngine.portfolio_summary(merged)

        covariance_matrix = self._covariance_matrix(merged)
        monte = PortfolioEngine.simulate_portfolio_monte_carlo(
            merged, covariance_matrix, num_simulations=2000, seed=42
        )

        typed_allocations = [
            PortfolioAllocationItem(
                ticker=a["ticker"],
                allocation=float(a["allocation"]),
                total_score=float(a["total_score"]),
                capital_amount=float(a.get("capital_amount", 0.0)),
                cagr=float(a.get("cagr", 0.0)),
                volatility=float(a.get("volatility", 0.0)),
                sharpe=float(a.get("sharpe", 0.0)),
                risk_share=float(a.get("risk_share", 0.0)),
            )
            for a in merged
        ]

        typed_summary = PortfolioSummary(
            portfolio_return=float(summary.get("portfolio_return", 0.0)),
            portfolio_volatility=float(summary.get("portfolio_volatility", 0.0)),
            portfolio_sharpe=float(summary.get("portfolio_sharpe", 0.0)),
            portfolio_mdd=float(summary.get("portfolio_mdd", 0.0)),
            portfolio_sortino=float(summary.get("portfolio_sortino", 0.0)),
            var_95=float(monte.get("var_95", 0.0)),
            cvar_95=float(monte.get("cvar_95", 0.0)),
            probability_of_loss=float(monte.get("probability_of_loss", 0.0)),
        )

        execution_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "portfolio.response exchange=%s valid_tickers=%s missing=%s execution_ms=%s",
            exchange,
            len(metrics_dict),
            len(missing),
            execution_ms,
        )

        warning = None
        if missing:
            warning = f"Skipped {len(missing)} ticker(s) due to missing/insufficient data: {', '.join(missing)}"

        return PortfolioResponse(
            ok=True,
            error=None,
            warning=warning,
            exchange=exchange,
            method=method,
            risk_profile=risk_profile,
            horizon_years=years,
            budget=budget_value,
            tickers=list(metrics_dict.keys()),
            allocations=typed_allocations,
            summary=typed_summary,
            chart_data=self._chart_data(typed_allocations),
            execution_ms=execution_ms,
        )


portfolio_service = PortfolioService()
