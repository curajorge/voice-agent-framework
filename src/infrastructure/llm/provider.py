"""Abstract LLM Provider interface.

This module defines the contract for LLM integrations, ensuring
agents don't depend directly on specific LLM implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from src.framework.core.signals import Response, ToolCall


@dataclass
class LLMMessage:
    """A message in the LLM conversation."""

    role: str  # "user", "assistant", "system"
    content: str | bytes
    is_audio: bool = False
    metadata: dict[str, Any] | None = None


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""

    model_name: str
    temperature: float = 0.7
    max_output_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: list[str] | None = None

    # Audio-specific config
    voice_name: str = "Kore"
    response_modality: str = "AUDIO"  # "TEXT" or "AUDIO"
    language: str = "en-US"


@dataclass
class LLMResponse:
    """Response from the LLM."""

    text: str | None = None
    audio_data: bytes | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = "stop"
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must handle:
    - Text and audio input/output
    - Tool/function calling
    - Streaming responses
    """

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the provider.

        Args:
            config: The LLM configuration.
        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: The conversation history.
            system_prompt: The system prompt.
            tools: Optional tool definitions.

        Returns:
            The LLM response.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[LLMMessage],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from the LLM.

        Args:
            messages: The conversation history.
            system_prompt: The system prompt.
            tools: Optional tool definitions.

        Yields:
            Partial LLM responses as they are generated.
        """
        pass

    @abstractmethod
    async def process_audio(
        self,
        audio_data: bytes,
        system_prompt: str,
        sample_rate: int = 16000,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Process audio input and generate a response.

        Args:
            audio_data: The input audio bytes.
            system_prompt: The system prompt.
            sample_rate: Audio sample rate in Hz.
            tools: Optional tool definitions.

        Returns:
            The LLM response (may include audio output).
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the provider and release resources."""
        pass
