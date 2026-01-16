"""Context management for stateful agent interactions.

The context system provides a hierarchical state management approach:
- GlobalContext: Application-wide state
- SessionContext: Per-conversation state
- UserContext: Authenticated user data
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Supported platforms for voice interactions."""

    TWILIO = "twilio"
    WEB = "web"
    CLI = "cli"
    TEST = "test"


class TaskStatus(str, Enum):
    """Status values for tasks."""

    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class VoicePreferences(BaseModel):
    """User preferences for voice interaction."""

    voice_name: str = "Kore"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    language: str = "en-US"


class UserContext(BaseModel):
    """Authenticated user context.

    Contains all user-specific data retrieved after authentication.
    """

    user_id: str
    phone_number: str
    full_name: str | None = None
    is_authenticated: bool = True
    voice_preferences: VoicePreferences = Field(default_factory=VoicePreferences)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def anonymous(cls) -> "UserContext":
        """Create an anonymous/unauthenticated user context.

        Returns:
            A UserContext with is_authenticated=False.
        """
        return cls(
            user_id="anonymous",
            phone_number="unknown",
            is_authenticated=False,
        )


class Scratchpad(BaseModel):
    """Temporary storage for multi-turn slot filling.

    Agents can use this to store partial form data during conversations.
    """

    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the scratchpad.

        Args:
            key: The key to store the value under.
            value: The value to store.
        """
        self.data[key] = value
        self.updated_at = datetime.utcnow()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the scratchpad.

        Args:
            key: The key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            The stored value or default.
        """
        return self.data.get(key, default)

    def clear(self) -> None:
        """Clear all data from the scratchpad."""
        self.data.clear()
        self.updated_at = datetime.utcnow()

    def has(self, key: str) -> bool:
        """Check if a key exists in the scratchpad.

        Args:
            key: The key to check.

        Returns:
            True if key exists, False otherwise.
        """
        return key in self.data


class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""

    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str  # "user" or "assistant" or "system"
    content: str
    agent_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HandoffData(BaseModel):
    """Data passed during agent handoffs for warm transitions.
    
    Contains context from the previous agent to enable seamless
    conversation continuity.
    """

    source_agent: str
    target_agent: str
    last_user_turn: str | None = None
    user_intent: str | None = None
    user_name: str | None = None
    greeting_completed: bool = False
    scratchpad_snapshot: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_context_injection(self) -> str:
        """Generate a context injection string for the new agent's prompt.
        
        Returns:
            A formatted context string for prompt injection.
        """
        parts = []
        
        if self.user_name:
            parts.append(f"User Name: {self.user_name}")
        
        if self.user_intent:
            parts.append(f"Previous Intent: {self.user_intent}")
            
        if self.last_user_turn:
            parts.append(f"Last User Message: \"{self.last_user_turn}\"")
            
        if self.greeting_completed:
            parts.append("Note: Greeting already completed. Do NOT re-greet the user.")
            
        if self.reason:
            parts.append(f"Handoff Reason: {self.reason}")
            
        if not parts:
            return ""
            
        return "[HANDOFF CONTEXT]\n" + "\n".join(parts) + "\n[END CONTEXT]"


class SessionContext(BaseModel):
    """Per-session context for a single conversation.

    Maintains conversation history and session-specific state.
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    platform: Platform = Platform.CLI
    active_agent: str = "router"
    previous_agent: str | None = None
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    scratchpad: Scratchpad = Field(default_factory=Scratchpad)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # Warm handoff support
    handoff_data: HandoffData | None = None
    greeting_completed: bool = False

    def add_turn(
        self,
        role: str,
        content: str,
        agent_name: str | None = None,
        **metadata: Any,
    ) -> ConversationTurn:
        """Add a turn to the conversation history.

        Args:
            role: The role (user/assistant/system).
            content: The content of the turn.
            agent_name: Name of the agent if role is assistant.
            **metadata: Additional metadata.

        Returns:
            The created ConversationTurn.
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            agent_name=agent_name,
            metadata=metadata,
        )
        self.conversation_history.append(turn)
        self.last_activity = datetime.utcnow()
        return turn

    def get_recent_history(self, limit: int = 10) -> list[ConversationTurn]:
        """Get the most recent conversation turns.

        Args:
            limit: Maximum number of turns to return.

        Returns:
            List of recent conversation turns.
        """
        return self.conversation_history[-limit:]

    def get_last_user_turn(self) -> str | None:
        """Get the content of the last user turn.
        
        Returns:
            The last user message or None.
        """
        for turn in reversed(self.conversation_history):
            if turn.role == "user":
                return turn.content
        return None

    def switch_agent(self, new_agent: str) -> None:
        """Switch the active agent.

        Args:
            new_agent: Name of the agent to switch to.
        """
        self.previous_agent = self.active_agent
        self.active_agent = new_agent
        self.last_activity = datetime.utcnow()

    def prepare_handoff(
        self,
        target_agent: str,
        reason: str | None = None,
        user_intent: str | None = None,
    ) -> HandoffData:
        """Prepare handoff data for agent transition.
        
        Args:
            target_agent: The agent being handed off to.
            reason: Reason for the handoff.
            user_intent: Detected user intent.
            
        Returns:
            HandoffData for the transition.
        """
        self.handoff_data = HandoffData(
            source_agent=self.active_agent,
            target_agent=target_agent,
            last_user_turn=self.get_last_user_turn(),
            user_intent=user_intent,
            greeting_completed=self.greeting_completed,
            scratchpad_snapshot=dict(self.scratchpad.data),
            reason=reason,
        )
        return self.handoff_data

    def consume_handoff(self) -> HandoffData | None:
        """Consume and clear the handoff data.
        
        Returns:
            The handoff data if present, None otherwise.
        """
        data = self.handoff_data
        self.handoff_data = None
        return data

    def mark_greeting_completed(self) -> None:
        """Mark that the initial greeting has been completed."""
        self.greeting_completed = True


class GlobalContext(BaseModel):
    """Application-wide context managing all sessions and state.

    This is the primary state container injected into agents and the orchestrator.
    """

    app_name: str = "Kura-Next"
    version: str = "1.0.0"
    environment: str = "development"
    current_time: datetime = Field(default_factory=datetime.utcnow)

    # Active session and user
    session: SessionContext = Field(default_factory=SessionContext)
    user: UserContext = Field(default_factory=UserContext.anonymous)

    # Agent registry
    available_agents: list[str] = Field(
        default_factory=lambda: ["router", "task_manager", "identity"]
    )

    # Global metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def refresh_time(self) -> None:
        """Update the current time."""
        self.current_time = datetime.utcnow()

    def set_user(self, user: UserContext) -> None:
        """Set the authenticated user.

        Args:
            user: The authenticated user context.
        """
        self.user = user
        # Update handoff data with user name if pending
        if self.session.handoff_data:
            self.session.handoff_data.user_name = user.full_name

    def clear_user(self) -> None:
        """Clear the user context (logout)."""
        self.user = UserContext.anonymous()

    def is_authenticated(self) -> bool:
        """Check if the current user is authenticated.

        Returns:
            True if user is authenticated, False otherwise.
        """
        return self.user.is_authenticated

    def get_platform(self) -> Platform:
        """Get the current platform.

        Returns:
            The platform enum value.
        """
        return self.session.platform

    def to_template_vars(self) -> dict[str, Any]:
        """Convert context to template variables for prompt rendering.

        Returns:
            Dictionary of template variables.
        """
        return {
            "user_name": self.user.full_name or "Guest",
            "current_time": self.current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform_source": self.session.platform.value,
            "session_id": self.session.session_id,
            "is_authenticated": self.user.is_authenticated,
            "greeting_completed": self.session.greeting_completed,
        }
