"""Tests for FastAPI API endpoints"""
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient

from models import Source


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_returns_200_with_valid_request(self, client):
        """Test successful query returns 200 status"""
        response = client.post("/api/query", json={"query": "What is Python?"})

        assert response.status_code == 200

    def test_query_returns_answer_and_session_id(self, client):
        """Test response contains required fields"""
        response = client.post("/api/query", json={"query": "What is Python?"})
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_query_with_existing_session_id(self, client_with_rag):
        """Test query with provided session_id uses it"""
        client, mock_rag = client_with_rag

        response = client.post(
            "/api/query",
            json={"query": "Follow up question", "session_id": "existing-session"}
        )
        data = response.json()

        assert response.status_code == 200
        assert data["session_id"] == "existing-session"
        mock_rag.query.assert_called_once_with("Follow up question", "existing-session")

    def test_query_without_session_id_creates_new_session(self, client_with_rag):
        """Test query without session_id creates a new one"""
        client, mock_rag = client_with_rag

        response = client.post("/api/query", json={"query": "New question"})
        data = response.json()

        assert response.status_code == 200
        assert data["session_id"] == "test-session-123"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_query_returns_sources(self, client_with_rag, sample_sources):
        """Test query returns sources in response"""
        client, mock_rag = client_with_rag
        mock_rag.query.return_value = ("Answer with sources", sample_sources)

        response = client.post("/api/query", json={"query": "Tell me about courses"})
        data = response.json()

        assert response.status_code == 200
        assert len(data["sources"]) == 3
        assert data["sources"][0]["text"] == "Introduction to Python - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/python/1"
        assert data["sources"][2]["link"] is None  # Third source has no link

    def test_query_with_empty_sources(self, client_with_rag):
        """Test query handles empty sources list"""
        client, mock_rag = client_with_rag
        mock_rag.query.return_value = ("General answer", [])

        response = client.post("/api/query", json={"query": "General question"})
        data = response.json()

        assert response.status_code == 200
        assert data["sources"] == []

    def test_query_error_returns_500(self, client_with_rag):
        """Test query error returns 500 with error detail"""
        client, mock_rag = client_with_rag
        mock_rag.query.side_effect = Exception("Database connection failed")

        response = client.post("/api/query", json={"query": "Test query"})

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]

    def test_query_missing_query_field_returns_422(self, client):
        """Test missing required query field returns validation error"""
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_query_empty_query_string(self, client):
        """Test empty query string is accepted"""
        response = client.post("/api/query", json={"query": ""})

        # Empty string is valid per schema, behavior depends on RAG system
        assert response.status_code == 200


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_returns_200(self, client):
        """Test courses endpoint returns 200"""
        response = client.get("/api/courses")

        assert response.status_code == 200

    def test_courses_returns_stats(self, client):
        """Test courses endpoint returns expected stats"""
        response = client.get("/api/courses")
        data = response.json()

        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Course A" in data["course_titles"]

    def test_courses_error_returns_500(self, client_with_rag):
        """Test courses endpoint error returns 500"""
        client, mock_rag = client_with_rag
        mock_rag.get_course_analytics.side_effect = Exception("Failed to get analytics")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "Failed to get analytics" in response.json()["detail"]


class TestRootEndpoint:
    """Tests for GET / endpoint"""

    def test_root_returns_200(self, client):
        """Test root endpoint returns 200"""
        response = client.get("/")

        assert response.status_code == 200

    def test_root_returns_message(self, client):
        """Test root endpoint returns welcome message"""
        response = client.get("/")
        data = response.json()

        assert "message" in data


class TestQueryResponseSerialization:
    """Tests for verifying Source serialization in QueryResponse"""

    def test_source_objects_serialize_correctly(self):
        """Test Source objects serialize to dict properly"""
        sources = [
            Source(text="Course A - Lesson 1", link="https://example.com/1"),
            Source(text="Course B - Lesson 2", link=None),
        ]

        serialized = [s.model_dump() for s in sources]

        assert serialized[0]["text"] == "Course A - Lesson 1"
        assert serialized[0]["link"] == "https://example.com/1"
        assert serialized[1]["link"] is None

    def test_query_response_with_sources(self):
        """Test QueryResponse model with Source objects"""
        from pydantic import BaseModel
        from typing import List

        class QueryResponse(BaseModel):
            answer: str
            sources: List[Source]
            session_id: str

        sources = [
            Source(text="Test Course - Lesson 1", link="https://example.com"),
        ]

        response = QueryResponse(
            answer="Test answer",
            sources=sources,
            session_id="test-session"
        )
        response_dict = response.model_dump()

        assert response_dict["answer"] == "Test answer"
        assert len(response_dict["sources"]) == 1
        assert response_dict["sources"][0]["text"] == "Test Course - Lesson 1"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_query_with_special_characters(self, client):
        """Test query with special characters"""
        response = client.post(
            "/api/query",
            json={"query": "What about <script>alert('xss')</script>?"}
        )

        assert response.status_code == 200

    def test_query_with_unicode(self, client):
        """Test query with unicode characters"""
        response = client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200

    def test_query_with_long_text(self, client):
        """Test query with very long text"""
        long_query = "What is " + "Python " * 1000 + "?"
        response = client.post("/api/query", json={"query": long_query})

        assert response.status_code == 200

    def test_multiple_sequential_queries(self, client_with_rag):
        """Test multiple queries in sequence"""
        client, mock_rag = client_with_rag

        responses = []
        for i in range(3):
            mock_rag.query.return_value = (f"Answer {i}", [])
            response = client.post("/api/query", json={"query": f"Question {i}"})
            responses.append(response)

        assert all(r.status_code == 200 for r in responses)
        assert mock_rag.query.call_count == 3
