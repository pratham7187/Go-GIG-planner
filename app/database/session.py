"""
Database engine and session management.

SQLite is used for simplicity in this take-home context (single file, zero
external services, trivial to inspect). See README "Trade-offs" for the
reasoning and what would change for a production Postgres deployment.
"""
import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config.settings import get_settings

settings = get_settings()

# Ensure the sqlite file's parent directory exists before engine creation.
if settings.database_url.startswith("sqlite:///"):
    db_path = settings.database_url.replace("sqlite:///", "", 1)
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}

engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator:
    """FastAPI dependency — yields a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session_scope():
    """Context manager for use outside request scope (e.g. background tasks)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create tables and apply small backward-compatible SQLite additions."""
    from app.models import image, analysis_result  # noqa: F401 (register models)
    Base.metadata.create_all(bind=engine)

    # ``create_all`` intentionally does not alter existing tables.  Keep this
    # lightweight migration beside the SQLite demo schema so an existing local
    # database receives the column required by the current result payload.
    if engine.dialect.name == "sqlite":
        columns = {column["name"] for column in inspect(engine).get_columns("analysis_results")}
        if "duplicate_type" not in columns:
            with engine.begin() as connection:
                connection.execute(text(
                    "ALTER TABLE analysis_results ADD COLUMN duplicate_type VARCHAR(32)"
                ))
