from __future__ import annotations

from pydantic import BaseModel, Field


class TickerSearchResponse(BaseModel):
    exchange: str
    query: str
    items: list[str] = Field(default_factory=list)

