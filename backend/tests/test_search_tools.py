"""Tests for CourseSearchTool.execute() method"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from models import Source


class TestCourseSearchToolExecute:
    """Tests for the execute method of CourseSearchTool"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_with_valid_results(self):
        """Test execute returns formatted results when search succeeds"""
        # Arrange
        mock_results = SearchResults(
            documents=["This is course content about Python basics"],
            metadata=[{"course_title": "Python 101", "lesson_number": 1}],
            distances=[0.5],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        # Act
        result = self.tool.execute(query="Python basics")

        # Assert
        assert "[Python 101 - Lesson 1]" in result
        assert "Python basics" in result
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_error_result(self):
        """Test execute returns error message when search fails"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: ChromaDB connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute(query="test query")

        # Assert
        assert result == "Search error: ChromaDB connection failed"

    def test_execute_with_empty_results(self):
        """Test execute returns appropriate message when no results found"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute(query="nonexistent topic")

        # Assert
        assert "No relevant content found" in result

    def test_execute_with_course_filter(self):
        """Test execute passes course filter to search"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "MCP Course", "lesson_number": 2}],
            distances=[0.3],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.tool.execute(query="MCP basics", course_name="MCP")

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="MCP basics",
            course_name="MCP",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self):
        """Test execute passes lesson filter to search"""
        # Arrange
        mock_results = SearchResults(
            documents=["Lesson content"],
            metadata=[{"course_title": "Course", "lesson_number": 3}],
            distances=[0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        result = self.tool.execute(query="test", lesson_number=3)

        # Assert
        self.mock_vector_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=3
        )

    def test_execute_empty_results_with_filters_shows_filter_info(self):
        """Test that empty results message includes filter information"""
        # Arrange
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute(query="test", course_name="MCP", lesson_number=2)

        # Assert
        assert "in course 'MCP'" in result
        assert "in lesson 2" in result


class TestCourseSearchToolFormatResults:
    """Tests for the _format_results method and source tracking"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.tool = CourseSearchTool(self.mock_vector_store)

    def test_format_results_creates_source_objects(self):
        """Test that _format_results creates proper Source objects"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content 1"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson"

        # Act
        self.tool.execute(query="test")

        # Assert
        assert len(self.tool.last_sources) == 1
        source = self.tool.last_sources[0]
        assert isinstance(source, Source)
        assert source.text == "Test Course - Lesson 1"
        assert source.link == "https://example.com/lesson"

    def test_format_results_deduplicates_sources(self):
        """Test that duplicate sources are not added"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content 1", "Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 1},  # Duplicate
                {"course_title": "Course A", "lesson_number": 2},  # Different lesson
            ],
            distances=[0.1, 0.2, 0.3],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        self.mock_vector_store.get_lesson_link.return_value = None

        # Act
        self.tool.execute(query="test")

        # Assert
        assert len(self.tool.last_sources) == 2  # Should be 2, not 3
        source_texts = [s.text for s in self.tool.last_sources]
        assert "Course A - Lesson 1" in source_texts
        assert "Course A - Lesson 2" in source_texts

    def test_format_results_handles_missing_lesson_number(self):
        """Test handling of results without lesson number"""
        # Arrange
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Course", "lesson_number": None}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Act
        result = self.tool.execute(query="test")

        # Assert
        assert len(self.tool.last_sources) == 1
        assert self.tool.last_sources[0].text == "Course"
        assert self.tool.last_sources[0].link is None
        # get_lesson_link should not be called when lesson_num is None
        self.mock_vector_store.get_lesson_link.assert_not_called()


class TestToolManager:
    """Tests for ToolManager"""

    def test_register_and_execute_tool(self):
        """Test registering and executing a tool"""
        # Arrange
        manager = ToolManager()
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = None

        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)

        # Act
        result = manager.execute_tool("search_course_content", query="test")

        # Assert
        assert "[Test - Lesson 1]" in result

    def test_execute_unknown_tool_returns_error(self):
        """Test that executing unknown tool returns error"""
        # Arrange
        manager = ToolManager()

        # Act
        result = manager.execute_tool("nonexistent_tool", query="test")

        # Assert
        assert "not found" in result

    def test_get_last_sources_returns_source_objects(self):
        """Test that get_last_sources returns Source objects"""
        # Arrange
        manager = ToolManager()
        mock_store = Mock()
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        mock_store.search.return_value = mock_results
        mock_store.get_lesson_link.return_value = "https://link.com"

        tool = CourseSearchTool(mock_store)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="test")

        # Act
        sources = manager.get_last_sources()

        # Assert
        assert len(sources) == 1
        assert isinstance(sources[0], Source)
        assert sources[0].text == "Test - Lesson 1"
        assert sources[0].link == "https://link.com"


class TestSearchResultsDataclass:
    """Tests for SearchResults dataclass"""

    def test_is_empty_returns_true_for_empty_documents(self):
        """Test is_empty returns True when documents list is empty"""
        results = SearchResults(documents=[], metadata=[], distances=[])
        assert results.is_empty() is True

    def test_is_empty_returns_false_for_non_empty_documents(self):
        """Test is_empty returns False when documents exist"""
        results = SearchResults(
            documents=["content"],
            metadata=[{}],
            distances=[0.1]
        )
        assert results.is_empty() is False

    def test_empty_constructor_sets_error(self):
        """Test SearchResults.empty() creates results with error"""
        results = SearchResults.empty("Test error message")
        assert results.error == "Test error message"
        assert results.is_empty() is True
