---
name: verify
description: Start the RAG server and smoke-test /api/courses and /api/query end to end. Use after changing anything in backend/ or frontend/.
disable-model-invocation: true
---

The repo has no automated tests, so verify changes against the running app.

1. Confirm `.env` exists with a real `ANTHROPIC_API_KEY`. If not, stop and tell the user.
2. Start the server in the background: `cd backend && uv run uvicorn app:app --port 8000`. Wait until `curl -s localhost:8000/api/courses` responds (the first start downloads the embedding model and ingests `docs/`, which can take a while).
3. Check `GET /api/courses`: expect 4 courses with titles listed.
4. Check a course-specific query that should trigger the search tool, and confirm `sources` is non-empty:
   `curl -s -X POST localhost:8000/api/query -H 'Content-Type: application/json' -d '{"query":"What is covered in lesson 1 of the computer use course?"}'`
5. Check a general-knowledge query (e.g. "What is a vector database?") and confirm `sources` is empty.
6. Reuse the returned `session_id` for a follow-up query to confirm conversation history works.
7. Stop the server and report each check as pass/fail with the evidence. Do not claim success for steps you did not run.
