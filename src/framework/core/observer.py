"""Intervention Observer for background stream analysis.

The observer runs concurrently with the active agent, analyzing the
input stream for hotwords, sentiment spikes, or other intervention triggers.
"""

import asyncio
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

import structlog

from src.framework.core.exceptions import PriorityIntervention
from src.framework.core.signals import AudioSignal, Signal, TextSignal

logger = structlog.get_logger(__name__)


class InterventionType(str, Enum):
    """Types of interventions the observer can trigger."""

    HOTWORD = "HOTWORD"
    SENTIMENT = "SENTIMENT"
    TIMEOUT = "TIMEOUT"
    EMERGENCY = "EMERGENCY"


class HotwordConfig:
    """Configuration for hotword detection."""

    def __init__(
        self,
        hotwords: list[str] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize hotword configuration.

        Args:
            hotwords: List of trigger words/phrases.
            case_sensitive: Whether matching is case-sensitive.
        """
        self.hotwords = hotwords or [
            "stop",
            "cancel",
            "operator",
            "help",
            "emergency",
            "nevermind",
            "never mind",
        ]
        self.case_sensitive = case_sensitive

    def matches(self, text: str) -> str | None:
        """Check if text contains a hotword.

        Args:
            text: The text to check.

        Returns:
            The matched hotword or None.
        """
        check_text = text if self.case_sensitive else text.lower()
        for word in self.hotwords:
            check_word = word if self.case_sensitive else word.lower()
            if check_word in check_text:
                return word
        return None


class InterventionObserver:
    """Background observer that monitors input streams for intervention triggers.

    The observer runs in parallel with the active agent, analyzing each
    input signal for hotwords, sentiment changes, or other conditions
    that warrant immediate intervention.
    """

    def __init__(
        self,
        hotword_config: HotwordConfig | None = None,
        timeout_seconds: float = 30.0,
        enable_sentiment: bool = False,
    ) -> None:
        """Initialize the intervention observer.

        Args:
            hotword_config: Configuration for hotword detection.
            timeout_seconds: Inactivity timeout threshold.
            enable_sentiment: Whether to analyze sentiment (requires model).
        """
        self.hotword_config = hotword_config or HotwordConfig()
        self.timeout_seconds = timeout_seconds
        self.enable_sentiment = enable_sentiment
        self._last_activity = asyncio.get_event_loop().time()
        self._intervention_triggered = False
        self._cancel_event = asyncio.Event()
        self._logger = structlog.get_logger(__name__)

    async def watch(
        self,
        signal_stream: AsyncGenerator[Signal, None],
    ) -> AsyncGenerator[Signal, None]:
        """Watch the signal stream for intervention triggers.

        This method wraps the original signal stream, analyzing each
        signal as it passes through.

        Args:
            signal_stream: The original signal stream to monitor.

        Yields:
            The original signals (pass-through).

        Raises:
            PriorityIntervention: When an intervention is triggered.
        """
        async for signal in signal_stream:
            if self._cancel_event.is_set():
                break

            # Update activity timestamp
            self._last_activity = asyncio.get_event_loop().time()

            # Analyze signal for intervention triggers
            await self._analyze_signal(signal)

            # Pass through the signal
            yield signal

    async def _analyze_signal(self, signal: Signal) -> None:
        """Analyze a signal for intervention triggers.

        Args:
            signal: The signal to analyze.

        Raises:
            PriorityIntervention: If an intervention is triggered.
        """
        text_content: str | None = None

        if isinstance(signal, TextSignal):
            text_content = signal.content
        elif isinstance(signal, AudioSignal):
            # For audio, we would need STT transcription
            # In production, this could use a lightweight ASR model
            # For now, check metadata for any transcription
            text_content = signal.metadata.get("transcription")

        if text_content:
            # Check for hotwords
            matched = self.hotword_config.matches(text_content)
            if matched:
                self._logger.info(
                    "hotword_detected",
                    hotword=matched,
                    session_id=signal.session_id,
                )
                self._intervention_triggered = True
                raise PriorityIntervention(
                    message=f"Hotword detected: {matched}",
                    intervention_type=InterventionType.HOTWORD.value,
                    target_agent=self._get_target_for_hotword(matched),
                    details={"hotword": matched, "original_text": text_content},
                )

            # Check for sentiment (if enabled)
            if self.enable_sentiment:
                sentiment_score = await self._analyze_sentiment(text_content)
                if sentiment_score < -0.7:  # Very negative sentiment
                    self._logger.info(
                        "negative_sentiment_detected",
                        score=sentiment_score,
                        session_id=signal.session_id,
                    )
                    raise PriorityIntervention(
                        message="Negative sentiment detected",
                        intervention_type=InterventionType.SENTIMENT.value,
                        target_agent="human_intervention",
                        details={"sentiment_score": sentiment_score},
                    )

    def _get_target_for_hotword(self, hotword: str) -> str | None:
        """Determine the target agent for a hotword.

        Args:
            hotword: The detected hotword.

        Returns:
            The target agent name or None for default handling.
        """
        hotword_lower = hotword.lower()
        if hotword_lower in ("operator", "help", "emergency"):
            return "human_intervention"
        if hotword_lower in ("stop", "cancel", "nevermind", "never mind"):
            return "router"
        return None

    async def _analyze_sentiment(self, text: str) -> float:
        """Analyze the sentiment of text.

        Args:
            text: The text to analyze.

        Returns:
            Sentiment score from -1.0 (negative) to 1.0 (positive).
        """
        # Simple keyword-based sentiment for demonstration
        # In production, use a proper sentiment model
        negative_words = [
            "angry", "frustrated", "terrible", "awful", "hate",
            "worst", "horrible", "disgusting", "furious", "upset",
        ]
        positive_words = [
            "great", "wonderful", "excellent", "amazing", "love",
            "best", "fantastic", "happy", "pleased", "thank",
        ]

        text_lower = text.lower()
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)

        total = neg_count + pos_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    async def check_timeout(self) -> None:
        """Check if the inactivity timeout has been exceeded.

        Raises:
            PriorityIntervention: If timeout is exceeded.
        """
        current_time = asyncio.get_event_loop().time()
        elapsed = current_time - self._last_activity

        if elapsed > self.timeout_seconds:
            self._logger.info(
                "timeout_detected",
                elapsed_seconds=elapsed,
            )
            raise PriorityIntervention(
                message=f"Inactivity timeout after {elapsed:.1f} seconds",
                intervention_type=InterventionType.TIMEOUT.value,
                target_agent="router",
                details={"elapsed_seconds": elapsed},
            )

    def cancel(self) -> None:
        """Signal the observer to stop watching."""
        self._cancel_event.set()

    def reset(self) -> None:
        """Reset the observer state."""
        self._intervention_triggered = False
        self._cancel_event.clear()
        self._last_activity = asyncio.get_event_loop().time()
