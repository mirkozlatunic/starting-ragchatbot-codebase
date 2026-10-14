import pytest


class TestQueryEndpoint:
    def test_creates_session_when_missing(self, client, mock_rag_system, sample_sources):
        r = client.post("/api/query", json={"query": "What is MCP?"})

        assert r.status_code == 200
        assert r.json() == {
            "answer": "MCP is a protocol.",
            "sources": sample_sources,
            "session_id": "session_1",
        }
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is MCP?", "session_1")

    def test_reuses_provided_session(self, client, mock_rag_system):
        r = client.post("/api/query", json={"query": "More?", "session_id": "abc"})

        assert r.status_code == 200
        assert r.json()["session_id"] == "abc"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("More?", "abc")

    def test_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("General answer.", [])

        r = client.post("/api/query", json={"query": "hi"})

        assert r.json()["sources"] == []

    @pytest.mark.parametrize(
        "body",
        [{}, {"session_id": "abc"}, {"query": 123}, {"query": None}],
    )
    def test_invalid_body_returns_422(self, client, body):
        assert client.post("/api/query", json=body).status_code == 422

    def test_non_json_body_returns_422(self, client):
        assert client.post("/api/query", content="not json").status_code == 422

    def test_rag_failure_returns_500_with_detail(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("boom")

        r = client.post("/api/query", json={"query": "x"})

        assert r.status_code == 500
        assert r.json() == {"detail": "boom"}


class TestCoursesEndpoint:
    def test_returns_stats(self, client):
        r = client.get("/api/courses")

        assert r.status_code == 200
        assert r.json() == {
            "total_courses": 2,
            "course_titles": ["MCP Course", "Chroma Course"],
        }

    def test_no_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        assert client.get("/api/courses").json() == {"total_courses": 0, "course_titles": []}

    def test_failure_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("chroma down")

        r = client.get("/api/courses")

        assert r.status_code == 500
        assert r.json() == {"detail": "chroma down"}


class TestSessionEndpoint:
    def test_delete_session(self, client, mock_rag_system):
        r = client.delete("/api/session/abc")

        assert r.status_code == 200
        assert r.json() == {"status": "ok"}
        mock_rag_system.session_manager.delete_session.assert_called_once_with("abc")


class TestRootEndpoint:
    def test_serves_index(self, client):
        r = client.get("/")

        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "RAG Chatbot" in r.text

    def test_unknown_static_path_404(self, client):
        assert client.get("/nope.js").status_code == 404

    def test_unknown_api_path_404(self, client):
        assert client.get("/api/unknown").status_code == 404
