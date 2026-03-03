from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes.chat import router as chat_api_router
from app.api.routes.portfolio import router as portfolio_api_router
from app.api.routes.ranking import router as ranking_api_router
from app.api.routes.risk import router as risk_api_router
from app.api.routes.ticker import router as ticker_api_router
from app.api.schemas.ranking import ErrorResponse
from app.web.routes import router as web_router


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s",
    )


def create_app() -> FastAPI:
    configure_logging()

    base_dir = Path(__file__).resolve().parent.parent
    app = FastAPI(title="Stock It Up")
    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.mount("/static", StaticFiles(directory=base_dir / "app" / "static"), name="static")
    app.state.templates = Jinja2Templates(directory=str(base_dir / "app" / "templates"))

    app.include_router(web_router)
    app.include_router(ranking_api_router)
    app.include_router(portfolio_api_router)
    app.include_router(risk_api_router)
    app.include_router(ticker_api_router)
    app.include_router(chat_api_router)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        payload = ErrorResponse(error=str(exc.detail), status=exc.status_code)
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        payload = ErrorResponse(error="Invalid request parameters", status=422)
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def catch_all_handler(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("stock_it_up.app").exception("Unhandled exception: %s", exc)
        payload = ErrorResponse(error="Internal server error", status=500)
        return JSONResponse(status_code=500, content=payload.model_dump())

    return app


app = create_app()
