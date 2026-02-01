"""Tests for FastAPI endpoints and serialization"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class TestQueryResponseSerialization:
    """Tests for verifying Source serialization in QueryResponse"""

    def test_source_objects_work_with_query_response(self):
        """Test that Source objects work correctly with QueryResponse"""
        from models import Source
        from pydantic import BaseModel
        from typing import List, Optional

        # QueryResponse now uses Source directly (after the fix)
        class QueryResponse(BaseModel):
            answer: str
            sources: List[Source]
            session_id: str

        # Create Source objects
        sources = [
            Source(text="Course A - Lesson 1", link="https://example.com/1"),
            Source(text="Course B - Lesson 2", link=None),
        ]

        # This should work now that QueryResponse uses Source directly
        try:
            response = QueryResponse(
                answer="Test answer",
                sources=sources,
                session_id="test-session"
            )
            # Verify serialization works
            response_dict = response.model_dump()
            assert len(response_dict["sources"]) == 2
            assert response_dict["sources"][0]["text"] == "Course A - Lesson 1"
            assert response_dict["sources"][0]["link"] == "https://example.com/1"
            assert response_dict["sources"][1]["link"] is None
        except Exception as e:
            pytest.fail(f"QueryResponse creation failed: {e}")


class TestAPIEndpoint:
    """Tests for the actual API endpoint"""

    @patch('app.rag_system')
    def test_query_endpoint_with_source_objects(self, mock_rag_system):
        """Test that /api/query endpoint handles Source objects correctly"""
        from app import app
        from models import Source

        # Setup mock
        mock_rag_system.session_manager.create_session.return_value = "new-session"
        mock_rag_system.query.return_value = (
            "This is the answer",
            [Source(text="Test Course - Lesson 1", link="https://example.com")]
        )

        client = TestClient(app)

        # Act
        response = client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        # Assert
        assert response.status_code == 200, f"Failed with: {response.text}"
        data = response.json()
        assert data["answer"] == "This is the answer"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com"

    @patch('app.rag_system')
    def test_query_endpoint_with_empty_sources(self, mock_rag_system):
        """Test that /api/query endpoint handles empty sources list"""
        from app import app

        mock_rag_system.session_manager.create_session.return_value = "new-session"
        mock_rag_system.query.return_value = ("General answer", [])

        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "General question"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []

    @patch('app.rag_system')
    def test_query_endpoint_error_handling(self, mock_rag_system):
        """Test that /api/query endpoint handles errors correctly"""
        from app import app

        mock_rag_system.session_manager.create_session.return_value = "new-session"
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        client = TestClient(app)

        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


class TestStartupInitialization:
    """Tests related to app initialization"""

    def test_rag_system_import_succeeds(self):
        """Test that RAGSystem can be imported without errors"""
        try:
            from rag_system import RAGSystem
        except Exception as e:
            pytest.fail(f"Failed to import RAGSystem: {e}")

    def test_config_import_succeeds(self):
        """Test that config can be imported"""
        try:
            from config import config
            assert hasattr(config, 'ANTHROPIC_API_KEY')
            assert hasattr(config, 'CHROMA_PATH')
        except Exception as e:
            pytest.fail(f"Failed to import config: {e}")
