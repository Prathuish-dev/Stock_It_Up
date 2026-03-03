from __future__ import annotations

import logging
import math
import time

from fastapi import HTTPException

from app.api.schemas.ranking import (
    BarChartData,
    PieChartData,
    RankingChartData,
    RankingItem,
    RankingPagination,
    RankingResponse,
    RankingSelection,
    ScatterPoint,
)
from chatbot.constants import DEFAULT_HORIZON_YEARS, METRIC_REGISTRY
from chatbot.data_loader import DataLoader
from chatbot.metric_cache import MetricCache
from chatbot.metrics_engine import MetricsEngine
from chatbot.screener_engine import ScreenerEngine

logger = logging.getLogger("stock_it_up.ranking")


class RankingService:
    MARKET_OPTIONS = ["NSE", "BSE"]
    METRIC_OPTIONS = [
        "cagr",
        "volatility",
        "avg_volume",
        "latest_price",
        "sharpe",
        "max_drawdown",
        "sortino",
        "score",
    ]
    ORDER_OPTIONS = ["best", "worst"]
    MAX_LIMIT = 200

    def __init__(self) -> None:
        self.loader = DataLoader()
        self.cache = MetricCache(self.loader)

    @staticmethod
    def safe_int(raw: int | str, default: int, low: int, high: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(low, min(high, value))

    def validate_inputs(self, exchange: str, metric: str, order: str) -> None:
        if exchange not in self.MARKET_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid market '{exchange}'.")
        if metric not in self.METRIC_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid metric '{metric}'.")
        if order not in self.ORDER_OPTIONS:
            raise HTTPException(status_code=400, detail=f"Invalid order '{order}'.")

    @staticmethod
    def _direction_for(metric: str, order: str) -> str:
        higher_is_better = METRIC_REGISTRY.get(metric, {}).get("higher_is_better", True)
        if higher_is_better is None:
            higher_is_better = True
        if order == "best":
            return "desc" if higher_is_better else "asc"
        return "asc" if higher_is_better else "desc"

    def _enrich_metrics(self, exchange: str, horizon_years: int, rows: list[dict]) -> list[dict]:
        enriched: list[dict] = []
        for row in rows:
            merged = dict(row)
            metrics = merged.get("metrics")
            if metrics is None:
                try:
                    df = self.loader.load_stock(exchange, merged["ticker"])
                    horizon_df = MetricsEngine.filter_by_horizon(df, horizon_years)
                    metrics = MetricsEngine.compute_all(horizon_df)
                except (FileNotFoundError, ValueError):
                    metrics = {}
            merged["metrics"] = metrics
            enriched.append(merged)
        return enriched

    @staticmethod
    def _build_chart_payload(rows: list[RankingItem]) -> RankingChartData:
        scatter: list[ScatterPoint] = []
        cagr_labels: list[str] = []
        cagr_values: list[float] = []
        pie_labels: list[str] = []
        pie_weights: list[float] = []

        for row in rows:
            scatter.append(
                ScatterPoint(
                    x=round(row.volatility * 100, 2),
                    y=round(row.cagr * 100, 2),
                    ticker=row.ticker,
                )
            )
            cagr_labels.append(row.ticker)
            cagr_values.append(round(row.cagr * 100, 2))
            pie_labels.append(row.ticker)
            pie_weights.append(1.0 / max(1, row.rank))

        total_weight = sum(pie_weights) or 1.0
        pie_values = [round((w / total_weight) * 100, 2) for w in pie_weights]

        return RankingChartData(
            scatter=scatter,
            cagrBar=BarChartData(labels=cagr_labels, values=cagr_values),
            pie=PieChartData(labels=pie_labels, values=pie_values),
        )

    def build_payload(
        self,
        *,
        exchange: str,
        metric: str,
        order: str,
        limit: int,
        page: int,
        horizon_years: int = DEFAULT_HORIZON_YEARS,
    ) -> RankingResponse:
        started = time.perf_counter()
        self.validate_inputs(exchange, metric, order)

        warning = None
        safe_limit = self.safe_int(limit, default=10, low=1, high=self.MAX_LIMIT)
        if safe_limit >= self.MAX_LIMIT:
            warning = f"Showing capped maximum of {self.MAX_LIMIT} results for performance."
        safe_page = self.safe_int(page, default=1, low=1, high=10_000)
        safe_horizon = self.safe_int(horizon_years, default=DEFAULT_HORIZON_YEARS, low=1, high=10)

        direction = self._direction_for(metric, order)
        cache_hit = self.cache.is_valid(exchange, safe_horizon)

        logger.info(
            "ranking.query exchange=%s metric=%s order=%s limit=%s page=%s horizon_years=%s cache_hit=%s",
            exchange,
            metric,
            order,
            safe_limit,
            safe_page,
            safe_horizon,
            cache_hit,
        )

        raw_rows = ScreenerEngine.run(
            exchange=exchange,
            metric=metric,
            limit=safe_limit,
            horizon_years=safe_horizon,
            direction=direction,
            data_loader=self.loader,
            cache=self.cache,
        )
        raw_rows = self._enrich_metrics(exchange, safe_horizon, raw_rows)

        typed_rows: list[RankingItem] = []
        for row in raw_rows:
            metrics = row.get("metrics", {})
            typed_rows.append(
                RankingItem(
                    rank=int(row.get("rank", 0)),
                    rank_label=str(row.get("rank_label", "")),
                    ticker=str(row.get("ticker", "")),
                    metric_value=float(row.get("value", 0.0)),
                    display_value=str(row.get("display_value", "")),
                    cagr=float(metrics.get("cagr", 0.0)),
                    volatility=float(metrics.get("volatility", 0.0)),
                    sharpe=float(metrics.get("sharpe", 0.0)),
                )
            )

        page_size = 20 if safe_limit > 20 else max(1, safe_limit)
        total_results = len(typed_rows)
        total_pages = max(1, math.ceil(total_results / page_size))
        safe_page = max(1, min(safe_page, total_pages))
        start = (safe_page - 1) * page_size
        end = start + page_size
        paginated_rows = typed_rows[start:end]

        execution_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "ranking.response exchange=%s metric=%s rows=%s page=%s/%s execution_ms=%s",
            exchange,
            metric,
            total_results,
            safe_page,
            total_pages,
            execution_ms,
        )

        return RankingResponse(
            ok=True,
            error=None if paginated_rows else "No data available for the selected filters.",
            warning=warning,
            selected=RankingSelection(
                exchange=exchange,
                metric=metric,
                order=order,
                limit=safe_limit,
                horizon_years=safe_horizon,
            ),
            metric_display=METRIC_REGISTRY.get(metric, {}).get("display", metric.upper()),
            results=paginated_rows,
            pagination=RankingPagination(
                page=safe_page,
                total_pages=total_pages,
                total_results=total_results,
            ),
            chart_data=self._build_chart_payload(paginated_rows),
            execution_ms=execution_ms,
            cache_hit=cache_hit,
        )


ranking_service = RankingService()

