import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from vector_store import SearchResults


def make_results(docs=None, metas=None, error=None):
    docs = docs or []
    metas = metas or [{} for _ in docs]
    return SearchResults(documents=docs, metadata=metas, distances=[0.1] * len(docs), error=error)


@pytest.fixture
def mock_store():
    """A VectorStore stand-in whose search() result can be set per test."""
    from unittest.mock import MagicMock

    store = MagicMock()
    store.search.return_value = make_results(
        ["Lesson text about MCP."], [{"course_title": "MCP Course", "lesson_number": 2}]
    )
    store.get_lesson_link.return_value = "https://example.com/lesson2"
    return store


@pytest.fixture
def real_store(tmp_path):
    """A VectorStore over a COPY of the real backend/chroma_db (never mutates the original)."""
    src = BACKEND_DIR / "chroma_db"
    if not src.exists():
        pytest.skip("backend/chroma_db not present")
    dst = tmp_path / "chroma_db"
    shutil.copytree(src, dst)
    from config import config
    from vector_store import VectorStore

    return VectorStore(str(dst), config.EMBEDDING_MODEL, config.MAX_RESULTS)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name="search_course_content", input=None, id="toolu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=input or {"query": "x"}, id=id)


def api_response(content, stop_reason="end_turn"):
    return SimpleNamespace(content=content, stop_reason=stop_reason)
