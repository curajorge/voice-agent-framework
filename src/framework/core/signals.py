"""Signal and Response definitions for agent communication.

Signals represent inputs to agents (audio or text), while Responses
represent the outputs from agents back to the user.
"""

from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """Types of signals that can be sent to agents."""

    AUDIO = "audio"
    TEXT = "text"
    SYSTEM = "system"


class ResponseType(str, Enum):
    """Types of responses that agents can produce."""

    AUDIO = "audio"
    TEXT = "text"
    TOOL_CALL = "tool_call"
    ROUTING = "routing"
    ERROR = "error"


class Signal(BaseModel, ABC):
    """Base class for all signals sent to agents.

    Signals are the input mechanism for agents. They can be audio bytes,
    text strings, or system events.
    """

    id: UUID = Field(default_factory=uuid4)
    signal_type: SignalType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        frozen = False


class AudioSignal(Signal):
    """Signal containing audio data from the user.

    This is the primary signal type when connected via phone/Twilio.
    Audio is in raw PCM format suitable for Gemini 2.5 Flash.
    """

    signal_type: SignalType = SignalType.AUDIO
    audio_data: bytes
    sample_rate: int = 16000
    channels: int = 1
    encoding: str = "LINEAR16"
    duration_ms: int | None = None

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class TextSignal(Signal):
    """Signal containing text input from the user.

    Used for CLI testing or text-based interfaces.
    """

    signal_type: SignalType = SignalType.TEXT
    content: str
    language: str = "en-US"


class SystemSignal(Signal):
    """Signal for system events (session start, end, etc.)."""

    signal_type: SignalType = SignalType.SYSTEM
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Represents a tool call requested by an agent."""

    tool_name: str
    arguments: dict[str, Any]
    call_id: str = Field(default_factory=lambda: str(uuid4()))


class RoutingDecision(BaseModel):
    """Represents a routing decision from the router agent."""

    thought_process: str
    route_to: str
    handover_context: str | None = None
    priority: int = 0


class Response(BaseModel):
    """Response from an agent after processing a signal.

    Responses can contain audio, text, tool calls, or routing decisions.
    """

    id: UUID = Field(default_factory=uuid4)
    response_type: ResponseType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str
    agent_name: str

    # Content fields (mutually exclusive based on response_type)
    audio_data: bytes | None = None
    text_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    routing_decision: RoutingDecision | None = None
    error_message: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    requires_tool_execution: bool = False
    is_final: bool = True

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True

    @classmethod
    def audio_response(
        cls,
        session_id: str,
        agent_name: str,
        audio_data: bytes,
        **kwargs: Any,
    ) -> "Response":
        """Create an audio response.

        Args:
            session_id: The session identifier.
            agent_name: Name of the responding agent.
            audio_data: The audio bytes to send.
            **kwargs: Additional metadata.

        Returns:
            A Response configured for audio output.
        """
        return cls(
            response_type=ResponseType.AUDIO,
            session_id=session_id,
            agent_name=agent_name,
            audio_data=audio_data,
            metadata=kwargs,
        )

    @classmethod
    def text_response(
        cls,
        session_id: str,
        agent_name: str,
        content: str,
        **kwargs: Any,
    ) -> "Response":
        """Create a text response.

        Args:
            session_id: The session identifier.
            agent_name: Name of the responding agent.
            content: The text content to send.
            **kwargs: Additional metadata.

        Returns:
            A Response configured for text output.
        """
        return cls(
            response_type=ResponseType.TEXT,
            session_id=session_id,
            agent_name=agent_name,
            text_content=content,
            metadata=kwargs,
        )

    @classmethod
    def tool_response(
        cls,
        session_id: str,
        agent_name: str,
        tool_calls: list[ToolCall],
        **kwargs: Any,
    ) -> "Response":
        """Create a tool call response.

        Args:
            session_id: The session identifier.
            agent_name: Name of the responding agent.
            tool_calls: List of tools to execute.
            **kwargs: Additional metadata.

        Returns:
            A Response configured for tool execution.
        """
        return cls(
            response_type=ResponseType.TOOL_CALL,
            session_id=session_id,
            agent_name=agent_name,
            tool_calls=tool_calls,
            requires_tool_execution=True,
            is_final=False,
            metadata=kwargs,
        )

    @classmethod
    def routing_response(
        cls,
        session_id: str,
        agent_name: str,
        decision: RoutingDecision,
        **kwargs: Any,
    ) -> "Response":
        """Create a routing response.

        Args:
            session_id: The session identifier.
            agent_name: Name of the responding agent.
            decision: The routing decision.
            **kwargs: Additional metadata.

        Returns:
            A Response configured for routing.
        """
        return cls(
            response_type=ResponseType.ROUTING,
            session_id=session_id,
            agent_name=agent_name,
            routing_decision=decision,
            is_final=False,
            metadata=kwargs,
        )
