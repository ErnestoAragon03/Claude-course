"""Pytest configuration and shared fixtures for RAG system tests"""
import sys
import os

# Ensure backend module is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from models import Source, Course, Lesson, CourseChunk


# --- Pydantic models for test app (mirrors app.py) ---

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# --- Fixtures for mock objects ---

@pytest.fixture
def mock_session_manager():
    """Mock SessionManager that returns predictable session IDs"""
    manager = Mock()
    manager.create_session.return_value = "test-session-123"
    manager.get_history.return_value = []
    manager.add_exchange.return_value = None
    return manager


@pytest.fixture
def mock_rag_system(mock_session_manager):
    """Mock RAGSystem with configurable behavior"""
    rag = Mock()
    rag.session_manager = mock_session_manager
    rag.query.return_value = ("Default test answer", [])
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"]
    }
    return rag


@pytest.fixture
def sample_sources():
    """Sample Source objects for testing"""
    return [
        Source(text="Introduction to Python - Lesson 1", link="https://example.com/python/1"),
        Source(text="Introduction to Python - Lesson 2", link="https://example.com/python/2"),
        Source(text="Advanced Topics - Lesson 5", link=None),
    ]


@pytest.fixture
def sample_courses():
    """Sample Course objects for testing"""
    return [
        Course(
            title="Introduction to Python",
            course_link="https://example.com/python",
            instructor="John Doe",
            lessons=[
                Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/python/1"),
                Lesson(lesson_number=2, title="Variables", lesson_link="https://example.com/python/2"),
            ]
        ),
        Course(
            title="Advanced Topics",
            instructor="Jane Smith",
            lessons=[
                Lesson(lesson_number=1, title="Decorators"),
                Lesson(lesson_number=2, title="Metaclasses"),
            ]
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Sample CourseChunk objects for testing"""
    return [
        CourseChunk(
            content="Python is a versatile programming language.",
            course_title="Introduction to Python",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables store data values.",
            course_title="Introduction to Python",
            lesson_number=2,
            chunk_index=0
        ),
    ]


# --- Test App Factory ---

def create_test_app(rag_system_mock: Mock) -> FastAPI:
    """
    Create a FastAPI test app with mocked RAG system.

    This avoids importing app.py directly, which would:
    1. Try to mount static files from ../frontend (doesn't exist in tests)
    2. Initialize the real RAG system
    """
    app = FastAPI(title="Test Course Materials RAG System")

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system_mock.session_manager.create_session()

            answer, sources = rag_system_mock.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system_mock.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Course Materials RAG System"}

    return app


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies"""
    return create_test_app(mock_rag_system)


@pytest.fixture
def client(test_app):
    """Create a TestClient for the test app"""
    return TestClient(test_app)


@pytest.fixture
def client_with_rag(mock_rag_system):
    """
    Create a TestClient with access to the mock RAG system.
    Returns tuple of (client, mock_rag_system) for configuring mock behavior.
    """
    app = create_test_app(mock_rag_system)
    return TestClient(app), mock_rag_system


# --- Helper fixtures for common test scenarios ---

@pytest.fixture
def rag_with_sources(mock_rag_system, sample_sources):
    """Configure mock RAG to return sample sources"""
    mock_rag_system.query.return_value = (
        "Here is information about the courses.",
        sample_sources
    )
    return mock_rag_system


@pytest.fixture
def rag_with_error(mock_rag_system):
    """Configure mock RAG to raise an error"""
    mock_rag_system.query.side_effect = Exception("Database connection failed")
    return mock_rag_system
