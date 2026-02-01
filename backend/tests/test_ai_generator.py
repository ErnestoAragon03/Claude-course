"""Tests for AIGenerator tool calling functionality"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock, patch
from ai_generator import AIGenerator


class MockContentBlock:
    """Mock for Anthropic content blocks"""
    def __init__(self, block_type, text=None, tool_name=None, tool_input=None, tool_id=None):
        self.type = block_type
        self.text = text
        self.name = tool_name
        self.input = tool_input or {}
        self.id = tool_id


class MockResponse:
    """Mock for Anthropic API response"""
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class TestAIGeneratorBasic:
    """Basic tests for AIGenerator initialization and simple responses"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_initialization(self, mock_anthropic_class):
        """Test AIGenerator initializes with correct parameters"""
        # Act
        generator = AIGenerator(api_key="test-key", model="test-model")

        # Assert
        mock_anthropic_class.assert_called_once_with(api_key="test-key")
        assert generator.model == "test-model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test generate_response returns text when no tools needed"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="This is the answer")]
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(query="What is Python?")

        # Assert
        assert result == "This is the answer"
        mock_client.messages.create.assert_called_once()


class TestAIGeneratorToolCalling:
    """Tests for tool calling functionality"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_passes_tools_to_api(self, mock_anthropic_class):
        """Test that tools are passed correctly to the API"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Answer")]
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")
        tools = [{"name": "search_tool", "description": "Searches content"}]

        # Act
        generator.generate_response(query="test", tools=tools)

        # Assert
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_handles_tool_use(self, mock_anthropic_class):
        """Test that tool_use stop_reason triggers tool execution"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: Claude wants to use a tool
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "MCP basics"},
                    tool_id="tool_123"
                )
            ]
        )

        # Second response: Claude's final answer
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Final answer about MCP")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: MCP is a protocol..."

        generator = AIGenerator(api_key="test-key", model="test-model")
        tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="What is MCP?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Final answer about MCP"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="MCP basics"
        )
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_results_passed_back_to_claude(self, mock_anthropic_class):
        """Test that tool results are correctly passed back to Claude"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_456"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Final answer")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        generator.generate_response(
            query="test query",
            tools=[{"name": "test"}],
            tool_manager=mock_tool_manager
        )

        # Assert - Check second call includes tool results
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Should have: user message, assistant tool_use, user tool_result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "tool_456"
        assert messages[2]["content"][0]["content"] == "Tool execution result"

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_tool_execution_without_tool_manager(self, mock_anthropic_class):
        """Test that tool_use is not handled if no tool_manager provided"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Response wants to use tool but no manager provided
        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(block_type="text", text="I need to search"),
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="search",
                    tool_input={},
                    tool_id="123"
                )
            ]
        )
        mock_client.messages.create.return_value = tool_use_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act - No tool_manager provided
        result = generator.generate_response(query="test", tools=[{"name": "search"}])

        # Assert - Should return the text content, not execute tool
        assert result == "I need to search"
        assert mock_client.messages.create.call_count == 1


class TestAIGeneratorConversationHistory:
    """Tests for conversation history handling"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_history_included_in_system(self, mock_anthropic_class):
        """Test that conversation history is appended to system prompt"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Answer")]
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")
        history = "User: Previous question\nAssistant: Previous answer"

        # Act
        generator.generate_response(query="Follow up", conversation_history=history)

        # Assert
        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" in system_content
        assert history in system_content

    @patch('ai_generator.anthropic.Anthropic')
    def test_no_history_prefix_when_history_is_none(self, mock_anthropic_class):
        """Test that system prompt is clean when no history provided"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Answer")]
        )
        mock_client.messages.create.return_value = mock_response

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        generator.generate_response(query="Question", conversation_history=None)

        # Assert
        call_args = mock_client.messages.create.call_args
        system_content = call_args.kwargs["system"]
        assert "Previous conversation:" not in system_content


class TestAIGeneratorErrorHandling:
    """Tests for error scenarios"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_propagates(self, mock_anthropic_class):
        """Test that API errors propagate correctly"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error: Rate limit exceeded")

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            generator.generate_response(query="test")

        assert "Rate limit exceeded" in str(exc_info.value)

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_passed_to_claude(self, mock_anthropic_class):
        """Test that tool execution errors are passed back to Claude"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_use_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_789"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="I couldn't find results")]
        )

        mock_client.messages.create.side_effect = [tool_use_response, final_response]

        # Tool manager returns an error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search error: ChromaDB connection failed"

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Assert - Error was passed to Claude and it responded
        second_call_messages = mock_client.messages.create.call_args_list[1].kwargs["messages"]
        tool_result_content = second_call_messages[2]["content"][0]["content"]
        assert "Search error:" in tool_result_content
