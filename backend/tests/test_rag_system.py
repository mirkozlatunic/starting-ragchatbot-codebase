"""Tests for how RAGSystem handles content-related queries (backend/rag_system.py)."""

from unittest.mock import MagicMock, patch

import pytest
from config import config
from conftest import api_response, text_block, tool_use_block
from rag_system import RAGSystem


@pytest.fixture
def rag(tmp_path):
    """RAGSystem with VectorStore + Anthropic client mocked out."""
    with (
        patch("rag_system.VectorStore") as vs,
        patch("ai_generator.anthropic.Anthropic") as anth,
    ):
        vs.return_value.search.return_value = MagicMock(error=None)
        system = RAGSystem(config)
        system.mock_client = anth.return_value
        system.mock_vs = vs.return_value
        yield system


def test_registers_search_and_outline_tools(rag):
    names = {d["name"] for d in rag.tool_manager.get_tool_definitions()}
    assert names == {"search_course_content", "get_course_outline"}


def test_query_passes_tool_definitions_to_claude(rag):
    rag.mock_client.messages.create.return_value = api_response([text_block("hi")])
    rag.query("What is MCP?")
    sent = rag.mock_client.messages.create.call_args.kwargs["tools"]
    assert {t["name"] for t in sent} == {"search_course_content", "get_course_outline"}


def test_content_query_runs_search_and_returns_answer_and_sources(rag):
    from conftest import make_results

    rag.mock_vs.search.return_value = make_results(
        ["chunk text"], [{"course_title": "MCP Course", "lesson_number": 1}]
    )
    rag.mock_vs.get_lesson_link.return_value = "https://l/1"
    rag.mock_client.messages.create.side_effect = [
        api_response([tool_use_block(input={"query": "what is mcp"})], stop_reason="tool_use"),
        api_response([text_block("MCP is ...")]),
    ]
    answer, sources = rag.query("What is MCP?")
    assert answer == "MCP is ..."
    assert sources == [{"text": "MCP Course - Lesson 1", "url": "https://l/1"}]
    rag.mock_vs.search.assert_called_once()


def test_sources_reset_after_query(rag):
    from conftest import make_results

    rag.mock_vs.search.return_value = make_results(
        ["c"], [{"course_title": "C", "lesson_number": 1}]
    )
    rag.mock_client.messages.create.side_effect = [
        api_response([tool_use_block()], stop_reason="tool_use"),
        api_response([text_block("a")]),
    ]
    rag.query("q")
    assert rag.search_tool.last_sources == []


def test_session_history_recorded_and_used(rag):
    rag.mock_client.messages.create.return_value = api_response([text_block("first")])
    rag.query("one", session_id="s1")
    rag.query("two", session_id="s1")
    system_prompt = rag.mock_client.messages.create.call_args.kwargs["system"]
    assert "one" in system_prompt and "first" in system_prompt


def test_general_question_does_not_search(rag):
    rag.mock_client.messages.create.return_value = api_response([text_block("4")])
    answer, sources = rag.query("What is 2+2?")
    assert answer == "4" and sources == []
    rag.mock_vs.search.assert_not_called()


def test_search_error_is_fed_back_to_claude_not_raised(rag):
    from conftest import make_results

    rag.mock_vs.search.return_value = make_results(error="Search error: boom")
    rag.mock_client.messages.create.side_effect = [
        api_response([tool_use_block()], stop_reason="tool_use"),
        api_response([text_block("sorry")]),
    ]
    answer, _ = rag.query("q")
    assert answer == "sorry"
    tool_result = rag.mock_client.messages.create.call_args_list[1].kwargs["messages"][2]
    assert tool_result["content"][0]["content"] == "Search error: boom"


# --- Config / environment sanity: these are what the real app depends on ---


def test_api_key_is_configured():
    assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is empty (check .env)"


def test_real_vector_store_is_populated_and_retrievable(real_store):
    assert real_store.get_course_count() > 0
    res = real_store.search("what is covered in the lessons")
    assert res.error is None, res.error
    assert not res.is_empty()


def test_max_results_is_positive():
    assert config.MAX_RESULTS > 0, "MAX_RESULTS=0 makes every search return nothing"


def test_end_to_end_with_real_store_and_fake_claude(real_store):
    """Real VectorStore + real tools; only the Anthropic client is faked."""
    with (
        patch("rag_system.VectorStore", return_value=real_store),
        patch("ai_generator.anthropic.Anthropic") as anth,
    ):
        system = RAGSystem(config)
        client = anth.return_value
        client.messages.create.side_effect = [
            api_response([tool_use_block(input={"query": "lessons"})], stop_reason="tool_use"),
            api_response([text_block("ok")]),
        ]
        answer, sources = system.query("What do the lessons cover?")
    assert answer == "ok"
    fed_back = client.messages.create.call_args_list[1].kwargs["messages"][2]["content"][0]
    assert not fed_back["content"].startswith("Search error"), fed_back["content"]
    assert sources, "no sources captured from a successful search"


def test_configured_model_exists_on_anthropic_api():
    """Regression: a retired model ID gives a 404 on every query -> HTTP 500 -> 'Query failed'."""
    import anthropic

    if not config.ANTHROPIC_API_KEY:
        pytest.skip("no API key")
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    client.models.retrieve(config.ANTHROPIC_MODEL)  # raises NotFoundError if retired/misspelled
