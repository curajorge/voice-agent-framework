"""Framework exception definitions.

This module defines the exception hierarchy used throughout the framework
for handling errors, interventions, and routing decisions.
"""

from typing import Any


class FrameworkException(Exception):
    """Base exception for all framework-related errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        """Initialize the framework exception.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class PriorityIntervention(FrameworkException):
    """Exception raised by InterventionObserver to force context switch.

    This exception is thrown when a hotword is detected or when sentiment
    analysis indicates the need for immediate intervention.
    """

    def __init__(
        self,
        message: str,
        intervention_type: str,
        target_agent: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the priority intervention.

        Args:
            message: Description of why intervention was triggered.
            intervention_type: Type of intervention (HOTWORD, SENTIMENT, TIMEOUT).
            target_agent: Optional agent to route to.
            details: Optional additional context.
        """
        super().__init__(message, details)
        self.intervention_type = intervention_type
        self.target_agent = target_agent


class RoutingException(FrameworkException):
    """Exception raised when routing to an agent fails."""

    def __init__(
        self,
        message: str,
        source_agent: str,
        target_agent: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the routing exception.

        Args:
            message: Description of the routing failure.
            source_agent: Agent that initiated the routing.
            target_agent: Agent that was the routing target.
            details: Optional additional context.
        """
        super().__init__(message, details)
        self.source_agent = source_agent
        self.target_agent = target_agent


class AgentException(FrameworkException):
    """Exception raised when an agent encounters an error during processing."""

    def __init__(
        self,
        message: str,
        agent_name: str,
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the agent exception.

        Args:
            message: Description of the agent error.
            agent_name: Name of the agent that raised the exception.
            recoverable: Whether the error is recoverable.
            details: Optional additional context.
        """
        super().__init__(message, details)
        self.agent_name = agent_name
        self.recoverable = recoverable


class ToolExecutionException(FrameworkException):
    """Exception raised when a tool fails to execute."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        arguments: dict[str, Any],
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tool execution exception.

        Args:
            message: Description of the tool failure.
            tool_name: Name of the tool that failed.
            arguments: Arguments passed to the tool.
            details: Optional additional context.
        """
        super().__init__(message, details)
        self.tool_name = tool_name
        self.arguments = arguments


class AuthenticationException(FrameworkException):
    """Exception raised when authentication fails or is required."""

    def __init__(
        self,
        message: str = "Authentication required",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the authentication exception.

        Args:
            message: Description of the authentication issue.
            details: Optional additional context.
        """
        super().__init__(message, details)


class SessionExpiredException(FrameworkException):
    """Exception raised when a session has expired."""

    def __init__(
        self,
        session_id: str,
        message: str = "Session has expired",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the session expired exception.

        Args:
            session_id: The expired session identifier.
            message: Description of the expiration.
            details: Optional additional context.
        """
        super().__init__(message, details)
        self.session_id = session_id
