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


class TestAIGeneratorSequentialToolCalling:
    """Tests for sequential tool calling (up to 2 rounds)"""

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calls_two_rounds(self, mock_anthropic_class):
        """Test that Claude can make 2 sequential tool calls"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Round 1: Claude wants to get course outline
        first_tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "MCP Course"},
                    tool_id="tool_1"
                )
            ]
        )

        # Round 2: Claude wants to search based on outline results
        second_tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "tool calling patterns"},
                    tool_id="tool_2"
                )
            ]
        )

        # Final: Claude provides answer
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Final answer combining both results")]
        )

        mock_client.messages.create.side_effect = [
            first_tool_response,
            second_tool_response,
            final_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 3 covers tool calling patterns",
            "Search results: Multiple courses discuss tool calling..."
        ]

        generator = AIGenerator(api_key="test-key", model="test-model")
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        # Act
        result = generator.generate_response(
            query="What topic does lesson 3 cover and what other courses discuss it?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Final answer combining both results"
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_terminates_after_max_rounds(self, mock_anthropic_class):
        """Test that tool calling stops after MAX_TOOL_ROUNDS even if Claude wants more"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # Claude keeps requesting tools (would be infinite without limit)
        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_loop"
                )
            ]
        )

        # After max rounds, final call without tools returns text
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Answer after max rounds")]
        )

        # 1 initial + 2 rounds = 3 API calls total
        mock_client.messages.create.side_effect = [
            tool_response,  # Initial response wants tool
            tool_response,  # Round 1 still wants tool
            final_response  # Round 2 (max reached, no tools) -> text
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Assert - Should stop after 2 tool execution rounds
        assert result == "Answer after max rounds"
        assert mock_client.messages.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_terminates_early_when_no_tool_use(self, mock_anthropic_class):
        """Test that loop exits early if Claude doesn't request another tool"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        # First response: tool use
        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "Test"},
                    tool_id="tool_1"
                )
            ]
        )

        # Second response: Claude is satisfied, no more tools needed
        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Got enough info")]
        )

        mock_client.messages.create.side_effect = [tool_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Complete course outline"

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act
        result = generator.generate_response(
            query="test",
            tools=[{"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )

        # Assert - Only 2 API calls (initial + 1 round), not 3
        assert result == "Got enough info"
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_exception_handled_gracefully(self, mock_anthropic_class):
        """Test that tool exceptions are caught and passed to Claude"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="search_course_content",
                    tool_input={"query": "test"},
                    tool_id="tool_err"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Handled the error")]
        )

        mock_client.messages.create.side_effect = [tool_response, final_response]

        # Tool manager raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        generator = AIGenerator(api_key="test-key", model="test-model")

        # Act - Should not raise, should handle gracefully
        result = generator.generate_response(
            query="test",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Assert - Error message passed to Claude
        assert result == "Handled the error"
        second_call = mock_client.messages.create.call_args_list[1]
        tool_result = second_call.kwargs["messages"][2]["content"][0]["content"]
        assert "Tool execution error:" in tool_result
        assert "Database connection failed" in tool_result

    @patch('ai_generator.anthropic.Anthropic')
    def test_tools_included_in_followup_calls(self, mock_anthropic_class):
        """Test that tools are included in follow-up API calls (not removed)"""
        # Arrange
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        tool_response = MockResponse(
            stop_reason="tool_use",
            content=[
                MockContentBlock(
                    block_type="tool_use",
                    tool_name="get_course_outline",
                    tool_input={"course_name": "Test"},
                    tool_id="tool_1"
                )
            ]
        )

        final_response = MockResponse(
            stop_reason="end_turn",
            content=[MockContentBlock(block_type="text", text="Done")]
        )

        mock_client.messages.create.side_effect = [tool_response, final_response]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"

        generator = AIGenerator(api_key="test-key", model="test-model")
        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        # Act
        generator.generate_response(query="test", tools=tools, tool_manager=mock_tool_manager)

        # Assert - Second API call should include tools
        second_call = mock_client.messages.create.call_args_list[1]
        assert "tools" in second_call.kwargs
        assert second_call.kwargs["tools"] == tools
        assert second_call.kwargs["tool_choice"] == {"type": "auto"}
