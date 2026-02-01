# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the application (backend serves frontend)
./run.sh
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000

# Run tests
cd backend && uv run pytest

# Run a single test file
cd backend && uv run pytest tests/test_api.py

# Run a specific test
cd backend && uv run pytest tests/test_api.py::test_query_response_with_sources
```

**Windows**: Use Git Bash for shell commands.

**Environment**: Create `.env` in root with `ANTHROPIC_API_KEY=your_key`

**Access**: Web UI at `http://localhost:8000`, API docs at `http://localhost:8000/docs`

## Architecture

This is a RAG (Retrieval-Augmented Generation) system for querying course materials.

### Data Flow
```
Course docs (docs/*.txt) → DocumentProcessor → VectorStore (ChromaDB)
                                                      ↓
User Query → FastAPI → RAGSystem → Claude + Tool Calling → CourseSearchTool
                                                      ↓
                                          Response with Sources → Frontend
```

### Backend Components (`backend/`)

| File | Purpose |
|------|---------|
| `app.py` | FastAPI endpoints: `POST /api/query`, `GET /api/courses`. Serves static frontend. |
| `rag_system.py` | Main orchestrator. Coordinates document loading, queries, and session management. |
| `document_processor.py` | Parses course documents, extracts metadata (title, instructor, lesson links), chunks text. |
| `vector_store.py` | ChromaDB wrapper. Two collections: `course_catalog` (metadata) and `course_content` (chunks). |
| `ai_generator.py` | Claude API wrapper with tool calling. Uses claude-sonnet-4-20250514, temperature 0. |
| `search_tools.py` | Defines `course_search` tool for Claude. Handles semantic search with course/lesson filtering. |
| `session_manager.py` | Tracks conversation history per session (last 2 exchanges). |
| `config.py` | Settings: embeddings model (`all-MiniLM-L6-v2`), chunk size (800), max results (5). |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk`, `Source`. |

### Frontend (`frontend/`)

Vanilla HTML/CSS/JS single-page app. Session state managed client-side (`currentSessionId`). API calls go to `/api/query` with session ID for conversation continuity.

### Key Patterns

- **Tool-based search**: Claude decides when to search via function calling, improving accuracy over naive RAG
- **Dual vector collections**: Catalog for course-level queries, content for detailed lesson queries
- **Semantic course matching**: Partial course name matches work (e.g., "MCP course" finds full title)
- **Source citations**: Search results include lesson links displayed in collapsible UI section
