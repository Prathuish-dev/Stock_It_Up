from __future__ import annotations

from django.urls import path

from django_webapp import views

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard", views.dashboard, name="dashboard"),
    path("ranking", views.ranking, name="ranking"),
    path("portfolio", views.portfolio, name="portfolio"),
    path("risk", views.risk, name="risk"),
]

