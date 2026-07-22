"""
Test fixtures.

Points the app at a temporary SQLite file and upload directory *before*
`app.main` is imported, so tests never touch the real dev database or
uploads folder — and each test run starts from a clean slate.
"""
import os
import tempfile

import pytest

_tmp_dir = tempfile.mkdtemp(prefix="vehicle_pipeline_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_dir}/test.db"
os.environ["UPLOAD_DIR"] = os.path.join(_tmp_dir, "uploads")
os.environ["GEMINI_API_KEY"] = ""

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.database.session import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    init_db()


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c
