"""VUI Metrics Instrumentation.

Provides structured logging for Voice User Interface metrics:
- TTFA (Time to First Audio)
- Routing Latency
- Silence Duration
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MetricType(str, Enum):
    """Types of VUI metrics."""

    TTFA = "ttfa"  # Time to First Audio
    ROUTING_LATENCY = "routing_latency"
    SILENCE_DURATION = "silence_duration"
    TOOL_EXECUTION = "tool_execution"
    FILLER_PLAYED = "filler_played"


@dataclass
class MetricEvent:
    """A single metric event."""

    metric_type: MetricType
    value_ms: float
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class VUIMetrics:
    """VUI Metrics collector and logger.
    
    Tracks timing metrics for voice interactions to identify
    and optimize latency sources.
    """

    # Thresholds for warnings
    TTFA_WARNING_MS = 500.0
    SILENCE_WARNING_MS = 1000.0
    ROUTING_WARNING_MS = 200.0

    def __init__(self, session_id: str) -> None:
        """Initialize metrics collector.
        
        Args:
            session_id: The session identifier.
        """
        self.session_id = session_id
        self._logger = structlog.get_logger(__name__).bind(session_id=session_id)
        self._timers: dict[str, float] = {}
        self._last_audio_sent: float | None = None
        self._silence_start: float | None = None

    def start_timer(self, name: str) -> None:
        """Start a named timer.
        
        Args:
            name: Timer name (e.g., 'ttfa', 'routing').
        """
        self._timers[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed milliseconds.
        
        Args:
            name: Timer name.
            
        Returns:
            Elapsed time in milliseconds.
        """
        if name not in self._timers:
            return 0.0
        elapsed = (time.perf_counter() - self._timers[name]) * 1000
        del self._timers[name]
        return elapsed

    def record_user_speech_end(self) -> None:
        """Record when user speech ends (for TTFA calculation)."""
        self.start_timer("ttfa")
        self._logger.debug("user_speech_end_recorded")

    def record_first_audio_sent(self) -> None:
        """Record when first audio packet is sent."""
        ttfa = self.stop_timer("ttfa")
        self._last_audio_sent = time.perf_counter()
        
        if ttfa > 0:
            self._log_metric(
                MetricType.TTFA,
                ttfa,
                warning_threshold=self.TTFA_WARNING_MS,
            )

    def record_routing_start(self) -> None:
        """Record when routing decision starts."""
        self.start_timer("routing")

    def record_routing_complete(self, target_agent: str) -> None:
        """Record when target agent becomes active.
        
        Args:
            target_agent: Name of the agent routed to.
        """
        latency = self.stop_timer("routing")
        self._log_metric(
            MetricType.ROUTING_LATENCY,
            latency,
            warning_threshold=self.ROUTING_WARNING_MS,
            metadata={"target_agent": target_agent},
        )

    def record_tool_execution(self, tool_name: str, duration_ms: float) -> None:
        """Record tool execution time.
        
        Args:
            tool_name: Name of the executed tool.
            duration_ms: Execution duration in milliseconds.
        """
        self._log_metric(
            MetricType.TOOL_EXECUTION,
            duration_ms,
            metadata={"tool_name": tool_name},
        )

    def record_filler_played(self, filler_type: str, duration_ms: float) -> None:
        """Record when filler audio is played.
        
        Args:
            filler_type: Type of filler (ROUTING, TOOL, etc.).
            duration_ms: Duration of filler in milliseconds.
        """
        self._log_metric(
            MetricType.FILLER_PLAYED,
            duration_ms,
            metadata={"filler_type": filler_type},
        )

    def check_silence(self) -> None:
        """Check for silence and log if threshold exceeded."""
        if self._last_audio_sent is None:
            return

        silence_duration = (time.perf_counter() - self._last_audio_sent) * 1000
        
        if silence_duration > self.SILENCE_WARNING_MS:
            if self._silence_start is None:
                self._silence_start = self._last_audio_sent
                self._log_metric(
                    MetricType.SILENCE_DURATION,
                    silence_duration,
                    warning_threshold=self.SILENCE_WARNING_MS,
                )

    def reset_silence_tracker(self) -> None:
        """Reset silence tracking after audio is sent."""
        self._silence_start = None
        self._last_audio_sent = time.perf_counter()

    def _log_metric(
        self,
        metric_type: MetricType,
        value_ms: float,
        warning_threshold: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log a metric event.
        
        Args:
            metric_type: Type of metric.
            value_ms: Value in milliseconds.
            warning_threshold: Optional threshold for warning log level.
            metadata: Optional additional metadata.
        """
        log_data = {
            "metric": metric_type.value,
            "value_ms": round(value_ms, 2),
            "session_id": self.session_id,
            **(metadata or {}),
        }

        if warning_threshold and value_ms > warning_threshold:
            self._logger.warning("vui_metric_exceeded_threshold", **log_data)
        else:
            self._logger.info("vui_metric", **log_data)
