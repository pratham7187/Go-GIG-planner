"""
Application-wide logging configuration.

Uses the standard library `logging` module with a consistent format so logs
are greppable in both local dev and container environments.
"""
import logging
import sys

from app.config.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())

    if root.handlers:
        # Already configured (e.g. reload) — avoid duplicate handlers.
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
