from __future__ import annotations

from django.urls import path

from django_api import views

urlpatterns = [
    path("api/ranking", views.api_ranking, name="api_ranking"),
    path("api/portfolio", views.api_portfolio, name="api_portfolio"),
    path("api/risk", views.api_risk, name="api_risk"),
    path("api/chat/start", views.api_chat_start, name="api_chat_start"),
    path("api/chat/message", views.api_chat_message, name="api_chat_message"),
    path("api/chat/reset", views.api_chat_reset, name="api_chat_reset"),
    path("api/tickers/search", views.api_ticker_search, name="api_ticker_search"),
]

