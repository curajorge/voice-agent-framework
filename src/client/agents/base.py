"""Base client agent with LLM integration.

Extends the framework's AbstractAgent with Gemini LLM capabilities
specific to the TaskMaster application.
"""

from pathlib import Path
from typing import Any

import structlog

from src.framework.core.agent import AbstractAgent, ModelConfig, Tool
from src.framework.core.context import GlobalContext, HandoffData
from src.framework.core.signals import (
    AudioSignal,
    Response,
    ResponseType,
    RoutingDecision,
    Signal,
    TextSignal,
    ToolCall,
)
from src.infrastructure.llm.gemini_audio import GeminiAudioClient, GeminiLiveSession
from src.infrastructure.llm.provider import LLMConfig, LLMMessage

logger = structlog.get_logger(__name__)


class BaseClientAgent(AbstractAgent):
    """Base class for client agents with Gemini integration.

    Provides common functionality for all TaskMaster agents including
    LLM client management and response generation.
    """

    def __init__(
        self,
        name: str,
        system_prompt_path: Path | str,
        gemini_client: GeminiAudioClient,
        model_config: ModelConfig | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        """Initialize the client agent.

        Args:
            name: Agent name.
            system_prompt_path: Path to system prompt.
            gemini_client: The Gemini audio client.
            model_config: Optional model configuration.
            tools: Optional list of tools.
        """
        super().__init__(name, system_prompt_path, model_config, tools)
        self.gemini_client = gemini_client
        self._live_session: GeminiLiveSession | None = None
        self._conversation: list[LLMMessage] = []

    async def process_signal(
        self,
        signal: Signal,
        context: GlobalContext,
    ) -> Response:
        """Process a signal using the Gemini LLM.

        Args:
            signal: The input signal.
            context: The global context.

        Returns:
            The agent's response.
        """
        # Render the system prompt with context
        rendered_prompt = self.render_prompt(context)

        # Build the message
        if isinstance(signal, AudioSignal):
            # Process audio directly
            llm_response = await self.gemini_client.process_audio(
                audio_data=signal.audio_data,
                system_prompt=rendered_prompt,
                sample_rate=signal.sample_rate,
                tools=self.get_tools_schema() if self.tools else None,
            )

            # Handle tool calls
            if llm_response.tool_calls:
                return Response.tool_response(
                    session_id=signal.session_id,
                    agent_name=self.name,
                    tool_calls=llm_response.tool_calls,
                )

            # Return audio response
            if llm_response.audio_data:
                return Response.audio_response(
                    session_id=signal.session_id,
                    agent_name=self.name,
                    audio_data=llm_response.audio_data,
                )

            # Fallback to text
            return Response.text_response(
                session_id=signal.session_id,
                agent_name=self.name,
                content=llm_response.text or "I'm sorry, I couldn't process that.",
            )

        elif isinstance(signal, TextSignal):
            # Add to conversation history
            self._conversation.append(
                LLMMessage(role="user", content=signal.content)
            )

            # Generate response
            llm_response = await self.gemini_client.generate(
                messages=self._conversation,
                system_prompt=rendered_prompt,
                tools=self.get_tools_schema() if self.tools else None,
            )

            # Handle tool calls
            if llm_response.tool_calls:
                return Response.tool_response(
                    session_id=signal.session_id,
                    agent_name=self.name,
                    tool_calls=llm_response.tool_calls,
                )

            # Add assistant response to history
            response_text = llm_response.text or "I understand."
            self._conversation.append(
                LLMMessage(role="assistant", content=response_text)
            )

            return Response.text_response(
                session_id=signal.session_id,
                agent_name=self.name,
                content=response_text,
            )

        else:
            return Response.text_response(
                session_id=signal.session_id,
                agent_name=self.name,
                content="I received an unsupported signal type.",
            )

    async def start_live_session(self, context: GlobalContext) -> None:
        """Start a live audio session.

        Args:
            context: The global context.
        """
        if self._live_session is None:
            rendered_prompt = self.render_prompt(context)
            self._live_session = await self.gemini_client.create_live_session(
                system_prompt=rendered_prompt,
                tools=self.get_tools_schema() if self.tools else None,
            )
            await self._live_session.__aenter__()
            self._logger.info("live_session_started")

    async def end_live_session(self) -> None:
        """End the live audio session."""
        if self._live_session:
            await self._live_session.__aexit__(None, None, None)
            self._live_session = None
            self._logger.info("live_session_ended")

    async def on_enter(
        self,
        context: GlobalContext,
        handoff_data: HandoffData | None = None,
    ) -> None:
        """Called when this agent becomes active.

        Args:
            context: The global context.
            handoff_data: Optional handoff data from previous agent.
        """
        await super().on_enter(context, handoff_data)
        # Clear conversation for fresh start (or keep for continuity)
        # self._conversation.clear()

    async def on_exit(self, context: GlobalContext) -> None:
        """Called when this agent is deactivated.

        Args:
            context: The global context.
        """
        await super().on_exit(context)
        await self.end_live_session()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation.clear()
