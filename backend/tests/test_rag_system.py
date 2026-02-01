"""Tests for RAGSystem end-to-end query handling"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional


@dataclass
class MockConfig:
    """Mock configuration for testing"""
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    CHROMA_PATH: str = "./test_chroma_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_RESULTS: int = 5
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    MAX_HISTORY: int = 2


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_rag_system_initializes_all_components(
        self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that RAGSystem initializes all required components"""
        from rag_system import RAGSystem

        # Arrange
        config = MockConfig()

        # Act
        rag = RAGSystem(config)

        # Assert
        mock_doc_proc.assert_called_once_with(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        mock_vector_store.assert_called_once_with(
            config.CHROMA_PATH,
            config.EMBEDDING_MODEL,
            config.MAX_RESULTS
        )
        mock_ai_gen.assert_called_once_with(
            config.ANTHROPIC_API_KEY,
            config.ANTHROPIC_MODEL
        )
        mock_session.assert_called_once_with(config.MAX_HISTORY)

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_rag_system_registers_search_tool(
        self, mock_session, mock_doc_proc, mock_ai_gen, mock_vector_store
    ):
        """Test that search tool is registered with tool manager"""
        from rag_system import RAGSystem

        # Arrange
        config = MockConfig()

        # Act
        rag = RAGSystem(config)

        # Assert
        assert rag.search_tool is not None
        assert "search_course_content" in rag.tool_manager.tools


class TestRAGSystemQuery:
    """Tests for RAGSystem.query() method"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_calls_ai_generator_with_tools(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that query passes tools to AI generator"""
        from rag_system import RAGSystem
        from models import Source

        # Arrange
        config = MockConfig()
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "AI response"

        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = None

        rag = RAGSystem(config)

        # Act
        response, sources = rag.query("What is MCP?", session_id="test-session")

        # Assert
        mock_ai_generator.generate_response.assert_called_once()
        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tool_manager"] == rag.tool_manager

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_retrieves_sources_from_tool_manager(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that sources are retrieved from tool manager after query"""
        from rag_system import RAGSystem
        from models import Source

        # Arrange
        config = MockConfig()
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response"

        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = None

        rag = RAGSystem(config)

        # Simulate that tool was used and sources were set
        test_sources = [Source(text="Course A - Lesson 1", link="https://example.com")]
        rag.search_tool.last_sources = test_sources

        # Act
        response, sources = rag.query("Question", session_id="test")

        # Assert
        assert sources == test_sources

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_resets_sources_after_retrieval(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that sources are reset after being retrieved"""
        from rag_system import RAGSystem
        from models import Source

        # Arrange
        config = MockConfig()
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response"

        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = None

        rag = RAGSystem(config)
        rag.search_tool.last_sources = [Source(text="Test", link=None)]

        # Act
        rag.query("Question", session_id="test")

        # Assert - Sources should be reset
        assert rag.search_tool.last_sources == []

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_updates_session_history(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that query updates conversation history"""
        from rag_system import RAGSystem

        # Arrange
        config = MockConfig()
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "AI Response"

        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = None

        rag = RAGSystem(config)

        # Act
        rag.query("User question", session_id="session-123")

        # Assert
        mock_session.add_exchange.assert_called_once_with(
            "session-123",
            "User question",
            "AI Response"
        )

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_includes_history_in_request(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that conversation history is passed to AI generator"""
        from rag_system import RAGSystem

        # Arrange
        config = MockConfig()
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.return_value = "Response"

        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = "Previous conversation..."

        rag = RAGSystem(config)

        # Act
        rag.query("Follow up question", session_id="session-456")

        # Assert
        call_kwargs = mock_ai_generator.generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == "Previous conversation..."


class TestRAGSystemErrorHandling:
    """Tests for error handling in RAGSystem"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_query_propagates_ai_generator_errors(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store
    ):
        """Test that AI generator errors propagate to caller"""
        from rag_system import RAGSystem

        # Arrange
        config = MockConfig()
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator
        mock_ai_generator.generate_response.side_effect = Exception("API Error")

        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = None

        rag = RAGSystem(config)

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            rag.query("Question")

        assert "API Error" in str(exc_info.value)


class TestRAGSystemIntegration:
    """Integration-style tests with minimal mocking"""

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.SessionManager')
    def test_full_query_flow_with_tool_execution(
        self, mock_session_class, mock_doc_proc, mock_ai_gen_class, mock_vector_store_class
    ):
        """Test complete query flow including tool execution"""
        from rag_system import RAGSystem
        from vector_store import SearchResults
        from models import Source

        # Arrange
        config = MockConfig()

        # Setup mock AI generator that simulates tool use
        mock_ai_generator = Mock()
        mock_ai_gen_class.return_value = mock_ai_generator

        # Setup mock vector store
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        mock_vector_store.search.return_value = SearchResults(
            documents=["MCP is a protocol for..."],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.2],
            error=None
        )
        mock_vector_store.get_lesson_link.return_value = "https://example.com/mcp/lesson1"

        mock_session = Mock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = None

        rag = RAGSystem(config)

        # Simulate AI generator calling the tool
        def simulate_ai_with_tool(query, conversation_history, tools, tool_manager):
            # AI decides to use the search tool
            tool_result = tool_manager.execute_tool(
                "search_course_content",
                query="MCP protocol"
            )
            return f"Based on the search: {tool_result[:50]}..."

        mock_ai_generator.generate_response.side_effect = simulate_ai_with_tool

        # Act
        response, sources = rag.query("What is MCP?", session_id="test")

        # Assert
        assert "MCP Course" in response or "MCP" in response
        assert len(sources) == 1
        assert sources[0].text == "MCP Course - Lesson 1"
        assert sources[0].link == "https://example.com/mcp/lesson1"


class TestSourceSerialization:
    """Tests to verify Source objects serialize correctly for API responses"""

    def test_source_object_has_required_fields(self):
        """Test Source object structure matches API expectations"""
        from models import Source

        source = Source(text="Course - Lesson 1", link="https://example.com")

        # Check fields exist
        assert hasattr(source, 'text')
        assert hasattr(source, 'link')
        assert source.text == "Course - Lesson 1"
        assert source.link == "https://example.com"

    def test_source_object_allows_none_link(self):
        """Test Source object allows None for link"""
        from models import Source

        source = Source(text="Course - Lesson 1", link=None)

        assert source.link is None

    def test_source_object_serializes_to_dict(self):
        """Test Source can be converted to dict for JSON serialization"""
        from models import Source

        source = Source(text="Course - Lesson 1", link="https://example.com")

        # Pydantic models have model_dump() method
        source_dict = source.model_dump()

        assert source_dict == {
            "text": "Course - Lesson 1",
            "link": "https://example.com"
        }
