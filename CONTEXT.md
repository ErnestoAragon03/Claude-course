# Project Context & Memory

This file serves as the persistent memory and architectural context for the codebase.

## 🎯 Project Goal
A Retrieval-Augmented Generation (RAG) system designed to query and retrieve information from specific DeepLearning.AI short courses. The system uses semantic search to find relevant course material and uses Claude to generate answers.

## 📚 Course Material (Knowledge Base)
The system currently ingests transcripts from the following courses (located in `docs/`):
1. **Building Towards Computer Use with Anthropic** (Anthropic API, Computer Use, Tool Calling)
2. **MCP: Build Rich-Context AI Apps with Anthropic** (Model Context Protocol, Servers/Clients)
3. **Advanced Retrieval for AI with Chroma** (RAG, Embeddings, Query Expansion)
4. **Prompt Compression and Query Optimization** (MongoDB, Optimization)

## 🏗️ Architecture

### Data Pipeline
`Raw Text (docs/)` -> `DocumentProcessor` -> `Chunks` -> `ChromaDB (Vector Store)`

### Backend (`backend/`)
- **Framework**: FastAPI (`app.py`)
- **Orchestrator**: `RAGSystem` (`rag_system.py`) coordinates retrieval and generation.
- **Database**: ChromaDB (`vector_store.py`) with two collections:
  - `course_catalog`: Metadata and course-level info.
  - `course_content`: Actual text chunks for retrieval.
- **AI Model**: Anthropic Claude 3.5 Sonnet (`ai_generator.py`).

### Frontend (`frontend/`)
- **Tech**: Vanilla HTML, CSS, JavaScript.
- **State**: Client-side session management.
- **Features**: 
  - Chat interface.
  - Source citation (collapsible).
  - Dark/Light theme toggle.

## 🔄 Data Flow
1. User sends query -> `POST /api/query`.
2. Claude analyzes query -> Decides to use `course_search` tool.
3. `VectorStore` retrieves relevant chunks.
4. Claude generates response using chunks as context.
5. Response + Sources sent back to Frontend.

## 🛠️ Key Dependencies
- **Python**: `fastapi`, `uvicorn`, `chromadb`, `anthropic`, `pydantic`.
- **Environment**: Managed by `uv`.
- **Testing**: `pytest`.

See `GEMINI.md` for coding rules and operational norms.