from typing import Any

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Available Tools:
- `search_course_content`: for questions about specific course content or detailed educational materials
- `get_course_outline`: for outline-related queries (course outline, syllabus, structure, "what lessons does X have")

Outline Queries:
- Use `get_course_outline`; the response must include the course title, the course link, and the number and title of every lesson

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may make up to 2 sequential tool calls per query, each after seeing the previous result
- Make a second call only when it depends on the first result (e.g. fetch a course outline, then search a specific lesson; or search one course, then another to compare). Prefer a single call
- Never repeat an identical search; after the final tool result, answer directly
- If a tool returns an error, say you couldn't retrieve that information
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first (up to 2 calls), then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    MAX_TOOL_ROUNDS = 2

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: str | None = None,
        tools: list | None = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Claude may chain up to MAX_TOOL_ROUNDS tool calls, each in its own API
        request so it can reason over earlier results. A final request with tool
        use disabled produces the answer when the cap is hit or a tool fails.

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

        api_params = {**self.base_params, "system": system_content}
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = self.client.messages.create(**api_params, messages=list(messages))
            if not tool_manager or not self._has_tool_use(response):
                return self._extract_text(response)

            messages.append({"role": "assistant", "content": response.content})
            tool_results, failed = self._execute_tools(response, tool_manager)
            messages.append({"role": "user", "content": tool_results})
            if failed:
                break

        # Cap reached or a tool failed: force a final text answer
        final_params = dict(api_params)
        if tools:
            final_params["tool_choice"] = {"type": "none"}
        final_response = self.client.messages.create(**final_params, messages=list(messages))
        return self._extract_text(final_response)

    @staticmethod
    def _has_tool_use(response) -> bool:
        return any(block.type == "tool_use" for block in response.content)

    @staticmethod
    def _extract_text(response) -> str:
        """Join the text blocks of a response; empty string if there are none."""
        return "".join(block.text for block in response.content if block.type == "text")

    @staticmethod
    def _execute_tools(response, tool_manager) -> tuple[list[dict[str, Any]], bool]:
        """
        Execute every tool_use block in the response.

        Every tool_use gets a tool_result (the API requires it). A raised
        exception becomes an is_error result and marks the batch as failed.

        Returns:
            (tool_result blocks, whether any tool raised)
        """
        results = []
        failed = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output = tool_manager.execute_tool(block.name, **block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            except Exception as e:  # noqa: BLE001 - reported to Claude as an is_error result
                failed = True
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error executing {block.name}: {e}",
                        "is_error": True,
                    }
                )
        return results, failed
