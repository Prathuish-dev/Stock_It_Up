from __future__ import annotations

import logging
import math
import time

from fastapi import HTTPException

from app.api.schemas.risk import (
    RiskAllocationItem,
    RiskChartData,
    RiskResponse,
    RiskSummary,
)
from chatbot.constants import DEFAULT_SCREENER_WEIGHTS
from chatbot.data_loader import DataLoader
from chatbot.metrics_engine import MetricsEngine, ScoringEngine
from chatbot.portfolio_engine import PortfolioEngine

logger = logging.getLogger("stock_it_up.risk")


class RiskService:
    MARKET_OPTIONS = ["NSE", "BSE"]
    METHOD_OPTIONS = ["proportional", "softmax", "risk_adjusted"]
    RISK_OPTIONS = ["LOW", "MEDIUM", "HIGH"]
    MAX_TICKERS = 25

    def __init__(self) -> None:
        self.loader = DataLoader()

    @staticmethod
    def safe_int(raw: int | str, default: int, low: int, high: int) -> int:
        try:
            value = int(raw)
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

    @staticmethod
    def _covariance_matrix(allocations: list[dict], rho: float = 0.2) -> list[list[float]]:
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

    @staticmethod
    def _histogram(values: list[float], bins: int = 30) -> tuple[list[str], list[float]]:
        if not values:
            return [], []
        low = min(values)
        high = max(values)
        if math.isclose(low, high):
            return [f"{low * 100:.2f}%"], [100.0]
        step = (high - low) / bins
        counts = [0 for _ in range(bins)]
        for v in values:
            idx = int((v - low) / step)
            if idx >= bins:
                idx = bins - 1
            counts[idx] += 1
        total = len(values)
        labels = []
        percents = []
        for i, c in enumerate(counts):
            start = low + i * step
            end = start + step
            labels.append(f"{start * 100:.1f}%..{end * 100:.1f}%")
            percents.append(round((c / total) * 100, 2))
        return labels, percents

    def build_payload(
        self,
        *,
        exchange: str,
        tickers: list[str] | str,
        method: str,
        risk_profile: str,
        horizon_years: int,
        num_simulations: int,
    ) -> RiskResponse:
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

        years = self.safe_int(horizon_years, default=3, low=1, high=10)
        sims = self.safe_int(num_simulations, default=3000, low=500, high=10000)

        logger.info(
            "risk.query exchange=%s method=%s risk=%s tickers=%s horizon=%s sims=%s",
            exchange,
            method,
            risk_profile,
            ",".join(clean_tickers),
            years,
            sims,
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
                detail="Not enough valid tickers with data for selected horizon.",
            )

        scored = ScoringEngine.compute_weighted_scores(metrics_dict, DEFAULT_SCREENER_WEIGHTS)
        minimal = []
        for row in scored:
            t = row["ticker"]
            minimal.append(
                {
                    "ticker": t,
                    "total_score": row["total_score"],
                    "cagr": metrics_dict[t].get("cagr", 0.0),
                    "volatility": metrics_dict[t].get("volatility", 0.0),
                    "sharpe": metrics_dict[t].get("sharpe", 0.0),
                    "max_drawdown": metrics_dict[t].get("max_drawdown", 0.0),
                    "sortino": metrics_dict[t].get("sortino", 0.0),
                    "expected_return": metrics_dict[t].get("cagr", 0.0),
                }
            )

        allocated = PortfolioEngine.allocate(minimal, method=method, risk_profile=risk_profile)
        alloc_map = {a["ticker"]: a["allocation"] for a in allocated}

        merged = []
        for row in minimal:
            ticker = row["ticker"]
            merged.append(
                {
                    **row,
                    "allocation": alloc_map.get(ticker, 0.0),
                }
            )

        alloc_total = sum(a["allocation"] for a in merged) or 1.0
        for a in merged:
            a["allocation"] = float(a["allocation"]) / alloc_total
        merged[-1]["allocation"] = 1.0 - sum(a["allocation"] for a in merged[:-1])

        merged = PortfolioEngine.compute_risk_decomposition(merged)
        summary = PortfolioEngine.portfolio_summary(merged)
        covariance = self._covariance_matrix(merged)
        monte = PortfolioEngine.simulate_portfolio_monte_carlo(
            merged, covariance, num_simulations=sims, seed=42
        )

        labels, values = self._histogram(monte.get("simulated_returns", []))

        allocations = [
            RiskAllocationItem(
                ticker=a["ticker"],
                allocation=float(a["allocation"]),
                cagr=float(a.get("cagr", 0.0)),
                volatility=float(a.get("volatility", 0.0)),
                sharpe=float(a.get("sharpe", 0.0)),
                risk_share=float(a.get("risk_share", 0.0)),
            )
            for a in merged
        ]

        typed_summary = RiskSummary(
            mean_return=float(monte.get("mean_return", 0.0)),
            std_dev=float(monte.get("std_dev", 0.0)),
            var_95=float(monte.get("var_95", 0.0)),
            cvar_95=float(monte.get("cvar_95", 0.0)),
            probability_of_loss=float(monte.get("probability_of_loss", 0.0)),
            portfolio_return=float(summary.get("portfolio_return", 0.0)),
            portfolio_volatility=float(summary.get("portfolio_volatility", 0.0)),
            portfolio_sharpe=float(summary.get("portfolio_sharpe", 0.0)),
            portfolio_mdd=float(summary.get("portfolio_mdd", 0.0)),
            portfolio_sortino=float(summary.get("portfolio_sortino", 0.0)),
        )

        chart_data = RiskChartData(
            histogram_labels=labels,
            histogram_values=values,
            risk_labels=[a.ticker for a in allocations],
            risk_values=[round(a.risk_share * 100, 2) for a in allocations],
            scatter_assets=[
                {
                    "x": round(a.volatility * 100, 2),
                    "y": round(a.cagr * 100, 2),
                    "ticker": a.ticker,
                }
                for a in allocations
            ],
            scatter_portfolio={
                "x": round(typed_summary.portfolio_volatility * 100, 2),
                "y": round(typed_summary.portfolio_return * 100, 2),
                "ticker": "PORTFOLIO",
            },
        )

        execution_ms = round((time.perf_counter() - started) * 1000, 2)
        warning = None
        if missing:
            warning = f"Skipped {len(missing)} ticker(s): {', '.join(missing)}"

        logger.info(
            "risk.response exchange=%s assets=%s missing=%s execution_ms=%s",
            exchange,
            len(allocations),
            len(missing),
            execution_ms,
        )

        return RiskResponse(
            ok=True,
            error=None,
            warning=warning,
            exchange=exchange,
            method=method,
            risk_profile=risk_profile,
            horizon_years=years,
            num_simulations=sims,
            allocations=allocations,
            summary=typed_summary,
            chart_data=chart_data,
            execution_ms=execution_ms,
        )


risk_service = RiskService()

