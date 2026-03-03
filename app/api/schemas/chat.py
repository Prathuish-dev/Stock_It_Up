from __future__ import annotations

from pydantic import BaseModel, Field


class ChatStartRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)


class ChatMessageRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    state: str
    reply: str
    done: bool

