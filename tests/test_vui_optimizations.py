"""Tests for VUI optimizations: latency masking, warm handoffs, and metrics."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.framework.core.context import (
    GlobalContext,
    SessionContext,
    UserContext,
    HandoffData,
    Platform,
)
from src.framework.core.io_handler import FillerType, FILLER_PHRASES
from src.framework.core.metrics import VUIMetrics, MetricType
from src.framework.core.signals import RoutingDecision, ToolCall


class TestHandoffData:
    """Tests for HandoffData warm handoff support."""

    def test_create_handoff_data(self):
        """Test creating handoff data."""
        handoff = HandoffData(
            source_agent="identity",
            target_agent="task_manager",
            last_user_turn="Create a task for me",
            user_intent="task creation",
            user_name="John Doe",
            greeting_completed=True,
            reason="User authenticated",
        )

        assert handoff.source_agent == "identity"
        assert handoff.target_agent == "task_manager"
        assert handoff.greeting_completed is True

    def test_context_injection_generation(self):
        """Test generating context injection string."""
        handoff = HandoffData(
            source_agent="identity",
            target_agent="task_manager",
            last_user_turn="Add a reminder",
            user_intent="create reminder",
            user_name="Jane",
            greeting_completed=True,
            reason="Auth complete",
        )

        injection = handoff.to_context_injection()

        assert "[HANDOFF CONTEXT]" in injection
        assert "User Name: Jane" in injection
        assert "Previous Intent: create reminder" in injection
        assert "Do NOT re-greet" in injection
        assert "[END CONTEXT]" in injection

    def test_empty_handoff_returns_empty_string(self):
        """Test that empty handoff returns empty injection."""
        handoff = HandoffData(
            source_agent="router",
            target_agent="task_manager",
        )

        injection = handoff.to_context_injection()
        assert injection == ""


class TestSessionContextHandoff:
    """Tests for SessionContext handoff methods."""

    def test_prepare_handoff(self):
        """Test preparing handoff data."""
        session = SessionContext(platform=Platform.TWILIO)
        session.add_turn("user", "Create a task")
        session.active_agent = "router"
        session.greeting_completed = True

        handoff = session.prepare_handoff(
            target_agent="task_manager",
            reason="Task intent detected",
            user_intent="create task",
        )

        assert handoff.source_agent == "router"
        assert handoff.target_agent == "task_manager"
        assert handoff.last_user_turn == "Create a task"
        assert handoff.greeting_completed is True
        assert session.handoff_data is handoff

    def test_consume_handoff(self):
        """Test consuming and clearing handoff data."""
        session = SessionContext()
        handoff = session.prepare_handoff(
            target_agent="task_manager",
            reason="Test",
        )

        consumed = session.consume_handoff()

        assert consumed is handoff
        assert session.handoff_data is None

    def test_get_last_user_turn(self):
        """Test getting last user turn."""
        session = SessionContext()
        session.add_turn("user", "First message")
        session.add_turn("assistant", "Response")
        session.add_turn("user", "Second message")

        last_turn = session.get_last_user_turn()
        assert last_turn == "Second message"


class TestVUIMetrics:
    """Tests for VUI metrics instrumentation."""

    def test_ttfa_measurement(self):
        """Test Time to First Audio measurement."""
        metrics = VUIMetrics(session_id="test-session")

        metrics.record_user_speech_end()
        # Simulate some processing time
        metrics.record_first_audio_sent()

        # Timer should be stopped
        assert "ttfa" not in metrics._timers

    def test_routing_latency_measurement(self):
        """Test routing latency measurement."""
        metrics = VUIMetrics(session_id="test-session")

        metrics.record_routing_start()
        metrics.record_routing_complete("task_manager")

        assert "routing" not in metrics._timers

    def test_silence_detection(self):
        """Test silence duration detection."""
        metrics = VUIMetrics(session_id="test-session")

        # Simulate audio sent
        metrics.reset_silence_tracker()
        
        # Check silence (should not trigger warning yet)
        metrics.check_silence()

    def test_tool_execution_recording(self):
        """Test tool execution time recording."""
        metrics = VUIMetrics(session_id="test-session")

        # Should not raise
        metrics.record_tool_execution("create_task", 150.0)

    def test_filler_recording(self):
        """Test filler audio recording."""
        metrics = VUIMetrics(session_id="test-session")

        metrics.record_filler_played("ROUTING", 500.0)


class TestFillerPhrases:
    """Tests for filler phrases configuration."""

    def test_all_filler_types_have_phrases(self):
        """Test that all filler types have phrases defined."""
        for filler_type in FillerType:
            assert filler_type in FILLER_PHRASES
            assert len(FILLER_PHRASES[filler_type]) > 0

    def test_routing_filler_phrases(self):
        """Test routing filler phrases exist."""
        phrases = FILLER_PHRASES[FillerType.ROUTING]
        assert "One moment please." in phrases

    def test_creating_filler_phrases(self):
        """Test creating filler phrases exist."""
        phrases = FILLER_PHRASES[FillerType.CREATING]
        assert "Let me add that for you." in phrases


class TestTransferAgentTool:
    """Tests for transfer_agent tool routing."""

    def test_routing_decision_creation(self):
        """Test creating routing decision."""
        decision = RoutingDecision(
            thought_process="User wants to create a task",
            route_to="task_manager",
            handover_context="Create a task for tomorrow",
        )

        assert decision.route_to == "task_manager"
        assert "task" in decision.thought_process.lower()

    def test_tool_call_creation(self):
        """Test creating transfer_agent tool call."""
        tool_call = ToolCall(
            tool_name="transfer_agent",
            arguments={
                "target_agent_name": "task_manager",
                "reason": "Task management request",
            },
        )

        assert tool_call.tool_name == "transfer_agent"
        assert tool_call.arguments["target_agent_name"] == "task_manager"


class TestGlobalContextHandoff:
    """Tests for GlobalContext handoff integration."""

    def test_set_user_updates_handoff(self):
        """Test that setting user updates pending handoff data."""
        context = GlobalContext()
        
        # Prepare a handoff
        handoff = context.session.prepare_handoff(
            target_agent="task_manager",
            reason="Test",
        )
        
        # Set user
        user = UserContext(
            user_id="user-123",
            phone_number="+1234567890",
            full_name="Test User",
        )
        context.set_user(user)
        
        # Handoff should have user name
        assert context.session.handoff_data.user_name == "Test User"

    def test_template_vars_include_greeting_state(self):
        """Test that template vars include greeting state."""
        context = GlobalContext()
        context.session.mark_greeting_completed()
        
        vars = context.to_template_vars()
        
        assert "greeting_completed" in vars
        assert vars["greeting_completed"] is True
