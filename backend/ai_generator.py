import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool call rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for course information.

Available Tools:
1. **search_course_content**: Search within course content for specific topics or information
2. **get_course_outline**: Get the complete structure/outline of a course with all lessons

Tool Usage Guidelines:
- Use **get_course_outline** when users ask about:
  - Course structure, outline, or organization
  - What lessons are in a course
  - What topics a course covers
  - How many lessons are in a course

- Use **search_course_content** when users ask about:
  - Specific topics, concepts, or information within course content
  - Detailed explanations from course materials

- **Up to 2 sequential tool calls per query** - Use this for complex questions requiring information from multiple sources (e.g., get course outline first, then search based on lesson title)
- If no results are found, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the search results" or "according to the outline"

When presenting course outlines:
- Include the course title and link
- List lessons with their numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls with support for sequential rounds.

        Executes tools and allows Claude to make follow-up tool calls based on results.
        Terminates when: max rounds reached, no tool_use in response, or tool error.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters (including tools)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        messages = base_params["messages"].copy()
        current_response = initial_response
        round_count = 0

        while round_count < self.MAX_TOOL_ROUNDS:
            # Add assistant's tool use response to messages
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in current_response.content:
                if content_block.type == "tool_use":
                    try:
                        tool_result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                    except Exception as e:
                        tool_result = f"Tool execution error: {str(e)}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

            # Add tool results as user message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            round_count += 1

            # Make follow-up API call WITH tools available for potential next round
            followup_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"]
            }

            # Keep tools available unless we've hit max rounds
            if round_count < self.MAX_TOOL_ROUNDS and "tools" in base_params:
                followup_params["tools"] = base_params["tools"]
                followup_params["tool_choice"] = {"type": "auto"}

            current_response = self.client.messages.create(**followup_params)

            # If Claude doesn't want to use tools, we're done
            if current_response.stop_reason != "tool_use":
                break

        # Extract text from final response
        for content_block in current_response.content:
            if hasattr(content_block, "text"):
                return content_block.text

        return ""