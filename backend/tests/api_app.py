"""A test-only FastAPI app mirroring the endpoints in backend/app.py.

backend/app.py builds a real RAGSystem and mounts ../frontend at import time, which breaks in a
test environment. This factory re-declares the same routes against an injected RAG system and a
temporary static directory. Keep it in sync with app.py when endpoints change.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class Source(BaseModel):
    text: str
    url: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: list[str]


def create_test_app(rag_system, static_dir: Path) -> FastAPI:
    app = FastAPI(title="Course Materials RAG System (test)")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        rag_system.session_manager.delete_session(session_id)
        return {"status": "ok"}

    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
    return app
