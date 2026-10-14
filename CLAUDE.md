# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running

- Setup: `uv sync`, then copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.
- Start: `./run.sh` (runs `uvicorn app:app --reload --port 8000` from `backend/`). App at http://localhost:8000, API docs at `/docs`.
- Always use `uv` to run the server and any Python (`uv run ...`, `uv add ...`, `uv sync`). Never call `pip`, `python`, or `uvicorn` directly.
- Run Python files with `uv run <file>.py` (e.g. `uv run python backend/foo.py` or `uv run backend/foo.py`), never `python <file>.py`.
- Manage ALL dependencies with uv: `uv add <pkg>` / `uv add --dev <pkg>` / `uv remove <pkg>`, which update `pyproject.toml` and `uv.lock`. Never edit dependency lists by hand, use `pip install`, or create a `requirements.txt`.
- There are no tests. `main.py` is an unused stub.
- First start downloads the `all-MiniLM-L6-v2` embedding model.

## Architecture

FastAPI backend (`backend/`) serving a vanilla JS frontend (`frontend/`). `RAGSystem` (`rag_system.py`) wires everything together.

Query flow: `POST /api/query` → `RAGSystem.query` → `AIGenerator` calls Claude with the `search_course_content` tool. If Claude calls it, `CourseSearchTool` → `VectorStore.search` (ChromaDB) runs, then a second Claude call **without tools** writes the answer. Retrieval is tool-based, not always-on, so general questions never touch the vector store.

## Gotchas

- Course docs in `docs/*.txt` must follow the parser format in `document_processor.py`: `Course Title:`, `Course Link:`, `Course Instructor:` on the first lines, then `Lesson N: Title` markers, each optionally followed by `Lesson Link:`.
- Ingestion skips any course whose title is already in ChromaDB. Edited docs are NOT re-ingested. Delete `backend/chroma_db/` (or call `add_course_folder(..., clear_existing=True)`) to rebuild.
- `add_course_folder` accepts `.pdf`/`.docx`, but `read_file` only reads plain text.
- Chunk context prefixes differ between the last lesson (every chunk gets "Course X Lesson N content:") and earlier lessons (first chunk only, "Lesson N content:").
- Session history is in memory only. `CourseSearchTool.last_sources` is shared state, so concurrent requests can mix up sources.
- Chroma has two collections: `course_catalog` (course-name resolution and lesson metadata) and `course_content` (chunks).
- Tunables (model, chunk size/overlap, max results, history length) are in `backend/config.py`.

## Style

- Format with black (line length 100, configured in `pyproject.toml`) and lint with ruff. `./scripts/format.sh` formats and applies ruff fixes; `./scripts/check.sh` runs black --check, ruff check and pytest without modifying files.
- A hook runs `ruff format` + `ruff check --fix` on edited Python files; its output matches black closely, but run `./scripts/format.sh` before committing.
