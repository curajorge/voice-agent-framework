"""Abstract Agent base class definition.

All agents in the framework must inherit from AbstractAgent and implement
the required abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import structlog

from src.framework.core.context import GlobalContext, HandoffData
from src.framework.core.signals import Response, Signal

logger = structlog.get_logger(__name__)


class Tool:
    """Wrapper for agent tools/functions.

    Tools are callable functions that agents can invoke to perform actions.
    """

    def __init__(
        self,
        name: str,
        description: str,
        function: Callable[..., Any],
        parameters: dict[str, Any],
        is_slow: bool = False,
    ) -> None:
        """Initialize the tool.

        Args:
            name: Unique name for the tool.
            description: Human-readable description.
            function: The callable to execute.
            parameters: JSON Schema for the parameters.
            is_slow: Whether this tool is slow (triggers filler audio).
        """
        self.name = name
        self.description = description
        self.function = function
        self.parameters = parameters
        self.is_slow = is_slow

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments to pass to the function.

        Returns:
            The result of the function execution.
        """
        logger.info("executing_tool", tool_name=self.name, arguments=kwargs)
        result = self.function(**kwargs)
        # Handle both sync and async functions
        if hasattr(result, "__await__"):
            return await result
        return result

    def to_gemini_schema(self) -> dict[str, Any]:
        """Convert tool to Gemini function declaration format.

        Returns:
            Dictionary in Gemini's function declaration format.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ModelConfig:
    """Configuration for the LLM model used by an agent."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-preview-native-audio-dialog",
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
        voice_name: str = "Kore",
        response_modality: str = "AUDIO",
        language: str = "en-US",
    ) -> None:
        """Initialize model configuration.

        Args:
            model_name: The Gemini model to use.
            temperature: Sampling temperature.
            max_output_tokens: Maximum tokens in response.
            voice_name: Voice for audio output.
            response_modality: "AUDIO" or "TEXT".
            language: Language code for speech.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.voice_name = voice_name
        self.response_modality = response_modality
        self.language = language


class AbstractAgent(ABC):
    """Abstract base class for all agents in the framework.

    Agents are the primary units of intelligence in the system. Each agent
    has a specific purpose, its own system prompt, and access to tools.
    """

    def __init__(
        self,
        name: str,
        system_prompt_path: Path | str,
        model_config: ModelConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        """Initialize the agent.

        Args:
            name: Unique identifier for the agent.
            system_prompt_path: Path to the system prompt file.
            model_config: Configuration for the LLM.
            tools: List of tools available to this agent.
        """
        self.name = name
        self.system_prompt_path = Path(system_prompt_path)
        self.model_config = model_config or ModelConfig()
        self.tools = tools or []
        self._system_prompt: str | None = None
        self._logger = structlog.get_logger(__name__).bind(agent=name)
        self._handoff_context: str | None = None

    @property
    def system_prompt(self) -> str:
        """Load and return the system prompt.

        Returns:
            The system prompt string.

        Raises:
            FileNotFoundError: If the prompt file doesn't exist.
        """
        if self._system_prompt is None:
            self._system_prompt = self._load_prompt()
        return self._system_prompt

    def _load_prompt(self) -> str:
        """Load the system prompt from file.

        Returns:
            The prompt content.
        """
        if self.system_prompt_path.exists():
            return self.system_prompt_path.read_text(encoding="utf-8")
        self._logger.warning(
            "prompt_file_not_found",
            path=str(self.system_prompt_path),
        )
        return f"You are {self.name}, an AI assistant."

    def render_prompt(self, context: GlobalContext) -> str:
        """Render the system prompt with context variables.

        Args:
            context: The global context for template variables.

        Returns:
            The rendered prompt string.
        """
        template_vars = context.to_template_vars()
        prompt = self.system_prompt
        for key, value in template_vars.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
        
        # Inject handoff context if present
        if self._handoff_context:
            prompt = f"{prompt}\n\n{self._handoff_context}"
            
        return prompt

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: The tool name to find.

        Returns:
            The tool if found, None otherwise.
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get all tools in Gemini schema format.

        Returns:
            List of tool schemas.
        """
        return [tool.to_gemini_schema() for tool in self.tools]

    def has_slow_tools(self) -> bool:
        """Check if agent has any slow tools.
        
        Returns:
            True if any tool is marked as slow.
        """
        return any(tool.is_slow for tool in self.tools)

    def is_tool_slow(self, tool_name: str) -> bool:
        """Check if a specific tool is slow.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            True if tool is marked as slow.
        """
        tool = self.get_tool(tool_name)
        return tool.is_slow if tool else False

    @abstractmethod
    async def process_signal(
        self,
        signal: Signal,
        context: GlobalContext,
    ) -> Response:
        """Process an incoming signal and produce a response.

        This is the core method that must be implemented by all agents.
        It receives user input and produces the agent's response.

        Args:
            signal: The input signal (audio or text).
            context: The global context.

        Returns:
            The agent's response.
        """
        pass

    async def on_enter(
        self,
        context: GlobalContext,
        handoff_data: HandoffData | None = None,
    ) -> None:
        """Lifecycle hook called when this agent becomes active.

        Override this to perform setup when the agent is routed to.
        Supports warm handoffs by accepting handoff data.

        Args:
            context: The global context.
            handoff_data: Optional data from the previous agent.
        """
        self._logger.info(
            "agent_activated",
            session_id=context.session.session_id,
            has_handoff=handoff_data is not None,
        )
        
        # Process handoff data for warm transition
        if handoff_data:
            self._handoff_context = handoff_data.to_context_injection()
            self._logger.info(
                "warm_handoff_received",
                source_agent=handoff_data.source_agent,
                user_intent=handoff_data.user_intent,
                greeting_completed=handoff_data.greeting_completed,
            )
        else:
            self._handoff_context = None

    async def on_exit(self, context: GlobalContext) -> None:
        """Lifecycle hook called when this agent is deactivated.

        Override this to perform cleanup when routing away from this agent.

        Args:
            context: The global context.
        """
        self._logger.info("agent_deactivated", session_id=context.session.session_id)
        # Clear handoff context on exit
        self._handoff_context = None

    async def handle_tool_result(
        self,
        tool_name: str,
        result: Any,
        context: GlobalContext,
    ) -> Response | None:
        """Handle the result of a tool execution.

        Override this to process tool results before sending to user.

        Args:
            tool_name: Name of the executed tool.
            result: The tool's return value.
            context: The global context.

        Returns:
            Optional response to send to user.
        """
        self._logger.debug(
            "tool_result_received",
            tool_name=tool_name,
            result_type=type(result).__name__,
        )
        return None

    def __repr__(self) -> str:
        """String representation of the agent.

        Returns:
            A string describing the agent.
        """
        return f"<{self.__class__.__name__}(name={self.name})>"
