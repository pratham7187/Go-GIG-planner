"""
Application entrypoint.

Wires together: logging, DB initialization, routers, static-file serving,
CORS, and a global exception handler so no controller needs its own
try/except for predictable domain errors.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import image_controller, health_controller
from app.config.settings import get_settings
from app.database.session import init_db
from app.utils.exceptions import ImageNotFoundError
from app.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(settings.upload_dir)

# Ensure upload directory exists before Starlette's StaticFiles validates it.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("Startup complete. env=%s gemini_enabled=%s", settings.env, settings.gemini_enabled)
    yield


app = FastAPI(
    title=settings.app_name,
    description="Async backend pipeline for validating field-uploaded vehicle images.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- CORS -----------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global exception handlers -------------------------------------------

@app.exception_handler(ImageNotFoundError)
def handle_image_not_found(request: Request, exc: ImageNotFoundError):
    return JSONResponse(
        status_code=404,
        content={"error": "image_not_found", "detail": str(exc)},
    )


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
    )


# --- Routers ---------------------------------------------------------------

app.include_router(health_controller.router, tags=["health"])
app.include_router(image_controller.router, tags=["images"])


# --- Root route (SPA entry point) -----------------------------------------
# Registered BEFORE static-file mounts so Starlette's sequential route
# resolution finds it first — mounts are greedy and would otherwise shadow
# a late-registered "/" route.

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the SPA index.html at the root URL."""
    return FileResponse(STATIC_DIR / "index.html")


# --- Static file mounts ---------------------------------------------------

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
