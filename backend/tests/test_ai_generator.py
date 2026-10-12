"""Tests that AIGenerator correctly drives the CourseSearchTool (backend/ai_generator.py)."""

from unittest.mock import MagicMock, call, patch

import pytest
from ai_generator import AIGenerator
from conftest import api_response, text_block, tool_use_block
from search_tools import CourseSearchTool, ToolManager

TOOLS = [{"name": "search_course_content", "input_schema": {"type": "object"}}]


@pytest.fixture
def gen():
    with patch("ai_generator.anthropic.Anthropic") as cls:
        g = AIGenerator("key", "claude-test")
        g.client = cls.return_value
        yield g


def test_tools_and_auto_tool_choice_sent_to_api(gen):
    gen.client.messages.create.return_value = api_response([text_block("hi")])
    gen.generate_response("q", tools=TOOLS)
    kwargs = gen.client.messages.create.call_args.kwargs
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == {"type": "auto"}
    assert kwargs["model"] == "claude-test"


def test_no_tools_key_when_none_given(gen):
    gen.client.messages.create.return_value = api_response([text_block("hi")])
    gen.generate_response("q")
    assert "tools" not in gen.client.messages.create.call_args.kwargs


def test_direct_answer_returned_without_tool_execution(gen):
    gen.client.messages.create.return_value = api_response([text_block("direct")])
    tm = MagicMock()
    assert gen.generate_response("q", tools=TOOLS, tool_manager=tm) == "direct"
    tm.execute_tool.assert_not_called()
    assert gen.client.messages.create.call_count == 1


def test_tool_use_executes_search_tool_with_model_arguments(gen):
    args = {"query": "what is MCP", "course_name": "MCP", "lesson_number": 1}
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block(input=args)], stop_reason="tool_use"),
        api_response([text_block("final")]),
    ]
    tm = MagicMock()
    tm.execute_tool.return_value = "search output"
    assert gen.generate_response("q", tools=TOOLS, tool_manager=tm) == "final"
    tm.execute_tool.assert_called_once_with("search_course_content", **args)


def test_second_call_contains_tool_result_and_tools_still_offered(gen):
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block(id="toolu_9")], stop_reason="tool_use"),
        api_response([text_block("final")]),
    ]
    tm = MagicMock()
    tm.execute_tool.return_value = "search output"
    gen.generate_response("q", tools=TOOLS, tool_manager=tm)

    second = gen.client.messages.create.call_args_list[1].kwargs
    assert second["tools"] == TOOLS
    assert second["messages"][0] == {"role": "user", "content": "q"}
    assert second["messages"][1]["role"] == "assistant"
    assert second["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "search output"}],
    }


def test_one_round_then_text_stops_early(gen):
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block()], stop_reason="tool_use"),
        api_response([text_block("final")]),
    ]
    tm = MagicMock()
    tm.execute_tool.return_value = "out"
    assert gen.generate_response("q", tools=TOOLS, tool_manager=tm) == "final"
    assert gen.client.messages.create.call_count == 2
    assert tm.execute_tool.call_count == 1


OUTLINE_ARGS = {"course_title": "X"}
SEARCH_ARGS = {"query": "lesson 4 topic", "course_name": "Y"}


def two_round_responses(final="answer"):
    return [
        api_response(
            [tool_use_block("get_course_outline", OUTLINE_ARGS, "toolu_a")], stop_reason="tool_use"
        ),
        api_response(
            [tool_use_block("search_course_content", SEARCH_ARGS, "toolu_b")],
            stop_reason="tool_use",
        ),
        api_response([text_block(final)]),
    ]


def test_two_sequential_rounds_outline_then_search(gen):
    gen.client.messages.create.side_effect = two_round_responses()
    tm = MagicMock()
    tm.execute_tool.side_effect = ["outline: lesson 4 = Topic", "search output"]

    assert gen.generate_response("q", tools=TOOLS, tool_manager=tm) == "answer"

    assert tm.execute_tool.call_args_list == [
        call("get_course_outline", **OUTLINE_ARGS),
        call("search_course_content", **SEARCH_ARGS),
    ]
    calls = gen.client.messages.create.call_args_list
    assert len(calls) == 3
    assert [len(c.kwargs["messages"]) for c in calls] == [1, 3, 5]
    for c in calls[:2]:
        assert c.kwargs["tools"] == TOOLS
        assert c.kwargs["tool_choice"] == {"type": "auto"}
    # round-1 result is visible when Claude decides on round 2 and when it answers
    assert calls[1].kwargs["messages"][2]["content"][0]["content"] == "outline: lesson 4 = Topic"
    last = calls[2].kwargs["messages"]
    assert last[2]["content"][0]["tool_use_id"] == "toolu_a"
    assert last[4]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_b",
        "content": "search output",
    }


def test_round_cap_forces_final_call_without_more_tools(gen):
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block(id="t1")], stop_reason="tool_use"),
        api_response([tool_use_block(id="t2")], stop_reason="tool_use"),
        api_response([text_block("forced answer")]),
    ]
    tm = MagicMock()
    tm.execute_tool.return_value = "out"

    assert gen.generate_response("q", tools=TOOLS, tool_manager=tm) == "forced answer"

    assert gen.client.messages.create.call_count == 3
    assert tm.execute_tool.call_count == 2
    assert gen.client.messages.create.call_args_list[2].kwargs["tool_choice"] == {"type": "none"}


def test_tool_exception_returned_as_error_and_ends_rounds(gen):
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block(id="t1")], stop_reason="tool_use"),
        api_response([text_block("sorry")]),
    ]
    tm = MagicMock()
    tm.execute_tool.side_effect = RuntimeError("boom")

    assert gen.generate_response("q", tools=TOOLS, tool_manager=tm) == "sorry"

    assert gen.client.messages.create.call_count == 2
    second = gen.client.messages.create.call_args_list[1].kwargs
    assert second["tool_choice"] == {"type": "none"}
    result = second["messages"][2]["content"][0]
    assert result["tool_use_id"] == "t1"
    assert result["is_error"] is True
    assert "boom" in result["content"]


def test_tool_exception_in_second_round_ends_with_final_call(gen):
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block(id="t1")], stop_reason="tool_use"),
        api_response([tool_use_block(id="t2")], stop_reason="tool_use"),
        api_response([text_block("partial answer")]),
    ]
    tm = MagicMock()
    tm.execute_tool.side_effect = ["ok", RuntimeError("down")]

    assert gen.generate_response("q", tools=TOOLS, tool_manager=tm) == "partial answer"

    final = gen.client.messages.create.call_args_list[2].kwargs
    assert final["messages"][2]["content"][0]["content"] == "ok"
    assert final["messages"][4]["content"][0]["is_error"] is True


def test_parallel_tool_use_all_answered_even_if_one_fails(gen):
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block(id="t1"), tool_use_block(id="t2")], stop_reason="tool_use"),
        api_response([text_block("final")]),
    ]
    tm = MagicMock()
    tm.execute_tool.side_effect = [RuntimeError("bad"), "fine"]

    gen.generate_response("q", tools=TOOLS, tool_manager=tm)

    results = gen.client.messages.create.call_args_list[1].kwargs["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert results[0]["is_error"] is True
    assert results[1]["content"] == "fine"


def test_tool_use_without_tool_manager_returns_text_not_crash(gen):
    gen.client.messages.create.return_value = api_response(
        [text_block("preamble"), tool_use_block()], stop_reason="tool_use"
    )
    assert gen.generate_response("q", tools=TOOLS) == "preamble"
    assert gen.client.messages.create.call_count == 1


def test_response_without_text_returns_empty_string(gen):
    gen.client.messages.create.return_value = api_response([])
    assert gen.generate_response("q") == ""


def test_history_system_prompt_identical_across_rounds(gen):
    gen.client.messages.create.side_effect = two_round_responses()
    tm = MagicMock()
    tm.execute_tool.return_value = "out"
    gen.generate_response("q", conversation_history="User: a", tools=TOOLS, tool_manager=tm)
    systems = {c.kwargs["system"] for c in gen.client.messages.create.call_args_list}
    assert len(systems) == 1
    assert "Previous conversation:\nUser: a" in systems.pop()


def test_conversation_history_goes_into_system_prompt(gen):
    gen.client.messages.create.return_value = api_response([text_block("hi")])
    gen.generate_response("q", conversation_history="User: a\nAssistant: b")
    assert (
        "Previous conversation:\nUser: a" in gen.client.messages.create.call_args.kwargs["system"]
    )


def test_real_tool_manager_round_trip(gen, mock_store):
    """tool_use block -> real ToolManager -> real CourseSearchTool -> follow-up call."""
    tm = ToolManager()
    tm.register_tool(CourseSearchTool(mock_store))
    gen.client.messages.create.side_effect = [
        api_response([tool_use_block(input={"query": "mcp"})], stop_reason="tool_use"),
        api_response([text_block("final")]),
    ]
    gen.generate_response("q", tools=tm.get_tool_definitions(), tool_manager=tm)
    second = gen.client.messages.create.call_args_list[1].kwargs
    assert "Lesson text about MCP." in second["messages"][2]["content"][0]["content"]


def test_system_prompt_points_model_at_search_tool():
    assert "search_course_content" in AIGenerator.SYSTEM_PROMPT


def test_system_prompt_allows_sequential_calls():
    assert "One tool call per query maximum" not in AIGenerator.SYSTEM_PROMPT
    assert "2 sequential tool calls" in AIGenerator.SYSTEM_PROMPT


def test_api_error_propagates(gen):
    """Documents current behaviour: API failures are not caught (-> HTTP 500 -> 'Query failed')."""
    gen.client.messages.create.side_effect = RuntimeError("api down")
    with pytest.raises(RuntimeError):
        gen.generate_response("q", tools=TOOLS)
