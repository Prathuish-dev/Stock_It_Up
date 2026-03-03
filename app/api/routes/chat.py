from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas.chat import ChatMessageRequest, ChatResponse, ChatStartRequest
from app.api.services.chat_service import chat_service

router = APIRouter(tags=["chat"])


@router.post("/api/chat/start", response_model=ChatResponse)
def start_chat(payload: ChatStartRequest) -> ChatResponse:
    return ChatResponse(**chat_service.start(payload.session_id))


@router.post("/api/chat/message", response_model=ChatResponse)
def send_chat_message(payload: ChatMessageRequest) -> ChatResponse:
    return ChatResponse(**chat_service.message(payload.session_id, payload.message))


@router.post("/api/chat/reset", response_model=ChatResponse)
def reset_chat(payload: ChatStartRequest) -> ChatResponse:
    return ChatResponse(**chat_service.reset(payload.session_id))

