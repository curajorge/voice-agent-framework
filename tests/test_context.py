"""Tests for context management."""

import pytest

from src.framework.core.context import (
    GlobalContext,
    SessionContext,
    UserContext,
    Platform,
    Scratchpad,
)


class TestUserContext:
    """Tests for UserContext."""

    def test_create_user_context(self):
        """Test creating a user context."""
        user = UserContext(
            user_id="test-123",
            phone_number="+1234567890",
            full_name="Test User",
        )

        assert user.user_id == "test-123"
        assert user.phone_number == "+1234567890"
        assert user.full_name == "Test User"
        assert user.is_authenticated is True

    def test_anonymous_user(self):
        """Test anonymous user context."""
        user = UserContext.anonymous()

        assert user.user_id == "anonymous"
        assert user.is_authenticated is False


class TestScratchpad:
    """Tests for Scratchpad."""

    def test_set_and_get(self):
        """Test setting and getting values."""
        pad = Scratchpad()
        pad.set("key1", "value1")

        assert pad.get("key1") == "value1"
        assert pad.get("nonexistent", "default") == "default"

    def test_has(self):
        """Test checking for key existence."""
        pad = Scratchpad()
        pad.set("exists", True)

        assert pad.has("exists") is True
        assert pad.has("not_exists") is False

    def test_clear(self):
        """Test clearing the scratchpad."""
        pad = Scratchpad()
        pad.set("key1", "value1")
        pad.clear()

        assert pad.has("key1") is False


class TestSessionContext:
    """Tests for SessionContext."""

    def test_create_session(self):
        """Test creating a session context."""
        session = SessionContext(platform=Platform.CLI)

        assert session.session_id is not None
        assert session.platform == Platform.CLI
        assert session.active_agent == "router"

    def test_switch_agent(self):
        """Test switching agents."""
        session = SessionContext()
        session.switch_agent("task_manager")

        assert session.active_agent == "task_manager"
        assert session.previous_agent == "router"

    def test_add_turn(self):
        """Test adding conversation turns."""
        session = SessionContext()
        turn = session.add_turn("user", "Hello")

        assert len(session.conversation_history) == 1
        assert turn.role == "user"
        assert turn.content == "Hello"


class TestGlobalContext:
    """Tests for GlobalContext."""

    def test_create_context(self):
        """Test creating global context."""
        context = GlobalContext(environment="test")

        assert context.environment == "test"
        assert context.is_authenticated() is False

    def test_set_user(self):
        """Test setting authenticated user."""
        context = GlobalContext()
        user = UserContext(
            user_id="test-123",
            phone_number="+1234567890",
            full_name="Test User",
        )
        context.set_user(user)

        assert context.is_authenticated() is True
        assert context.user.full_name == "Test User"

    def test_clear_user(self):
        """Test clearing user (logout)."""
        context = GlobalContext()
        context.set_user(
            UserContext(
                user_id="test",
                phone_number="+1",
                is_authenticated=True,
            )
        )
        context.clear_user()

        assert context.is_authenticated() is False

    def test_template_vars(self):
        """Test generating template variables."""
        context = GlobalContext()
        context.set_user(
            UserContext(
                user_id="test",
                phone_number="+1",
                full_name="John Doe",
            )
        )

        vars = context.to_template_vars()

        assert vars["user_name"] == "John Doe"
        assert vars["is_authenticated"] is True
