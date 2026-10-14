"""Tests for CourseSearchTool.execute (backend/search_tools.py)."""

from conftest import make_results
from search_tools import CourseSearchTool, ToolManager


class TestExecuteWithMockedStore:
    def test_formats_results_with_course_and_lesson_header(self, mock_store):
        out = CourseSearchTool(mock_store).execute(query="mcp")
        assert out.startswith("[MCP Course - Lesson 2]")
        assert "Lesson text about MCP." in out

    def test_passes_filters_through_to_store(self, mock_store):
        CourseSearchTool(mock_store).execute(query="q", course_name="MCP", lesson_number=3)
        mock_store.search.assert_called_once_with(query="q", course_name="MCP", lesson_number=3)

    def test_store_error_is_returned_verbatim(self, mock_store):
        mock_store.search.return_value = make_results(error="Search error: boom")
        assert CourseSearchTool(mock_store).execute(query="q") == "Search error: boom"

    def test_empty_results_message(self, mock_store):
        mock_store.search.return_value = make_results()
        assert CourseSearchTool(mock_store).execute(query="q") == "No relevant content found."

    def test_empty_results_message_includes_filters(self, mock_store):
        mock_store.search.return_value = make_results()
        out = CourseSearchTool(mock_store).execute(query="q", course_name="MCP", lesson_number=4)
        assert out == "No relevant content found in course 'MCP' in lesson 4."

    def test_multiple_results_are_separated(self, mock_store):
        mock_store.search.return_value = make_results(
            ["a", "b"],
            [
                {"course_title": "C", "lesson_number": 1},
                {"course_title": "C", "lesson_number": 2},
            ],
        )
        out = CourseSearchTool(mock_store).execute(query="q")
        assert "[C - Lesson 1]\na\n\n[C - Lesson 2]\nb" == out

    def test_sources_tracked_with_links_and_deduplicated(self, mock_store):
        mock_store.search.return_value = make_results(
            ["a", "b"], [{"course_title": "C", "lesson_number": 1}] * 2
        )
        tool = CourseSearchTool(mock_store)
        tool.execute(query="q")
        assert tool.last_sources == [{"text": "C - Lesson 1", "url": "https://example.com/lesson2"}]

    def test_result_without_lesson_number_has_no_lesson_in_header(self, mock_store):
        mock_store.search.return_value = make_results(["a"], [{"course_title": "C"}])
        assert CourseSearchTool(mock_store).execute(query="q").startswith("[C]\n")


class TestToolDefinition:
    def test_definition_shape(self, mock_store):
        d = CourseSearchTool(mock_store).get_tool_definition()
        assert d["name"] == "search_course_content"
        assert d["input_schema"]["required"] == ["query"]
        assert set(d["input_schema"]["properties"]) == {"query", "course_name", "lesson_number"}

    def test_tool_manager_executes_by_name(self, mock_store):
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(mock_store))
        assert "MCP Course" in tm.execute_tool("search_course_content", query="q")
        assert tm.execute_tool("nope") == "Tool 'nope' not found"


class TestExecuteAgainstRealChroma:
    """Uses a copy of the real chroma_db to catch store-level breakage."""

    def test_content_search_returns_results_not_error(self, real_store):
        out = CourseSearchTool(real_store).execute(query="What is covered in the lessons?")
        assert not out.startswith("Search error"), out
        assert not out.startswith("No relevant content"), out

    def test_course_name_filter_resolves(self, real_store):
        title = real_store.get_existing_course_titles()[0]
        out = CourseSearchTool(real_store).execute(query="introduction", course_name=title)
        assert not out.startswith(("Search error", "No course found")), out

    def test_course_and_lesson_filter_does_not_error(self, real_store):
        title = real_store.get_existing_course_titles()[0]
        out = CourseSearchTool(real_store).execute(
            query="introduction", course_name=title, lesson_number=1
        )
        assert not out.startswith("Search error"), out
