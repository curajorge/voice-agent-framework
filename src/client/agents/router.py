"""Router Agent - The Receptionist with Tool-Based Routing.

The Router Agent is responsible for intent classification and routing
incoming requests to the appropriate specialized agent using tool calls.
"""

import json
from pathlib import Path
from typing import Any

import structlog

from src.client.agents.base import BaseClientAgent
from src.framework.core.agent import ModelConfig, Tool
from src.framework.core.context import GlobalContext
from src.framework.core.signals import (
    AudioSignal,
    Response,
    ResponseType,
    RoutingDecision,
    Signal,
    TextSignal,
    ToolCall,
)
from src.infrastructure.llm.gemini_audio import GeminiAudioClient
from src.infrastructure.llm.provider import LLMMessage

logger = structlog.get_logger(__name__)


class RouterAgent(BaseClientAgent):
    """The Receptionist - Central Orchestrator for intent routing.

    Uses tool-based routing via the transfer_agent tool for immediate
    agent switching without text generation latency.
    """

    # Valid routing targets
    VALID_TARGETS = ["identity", "task_manager", "router"]

    def __init__(
        self,
        gemini_client: GeminiAudioClient,
        system_prompt_path: Path | str | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        """Initialize the Router Agent.

        Args:
            gemini_client: The Gemini audio client.
            system_prompt_path: Optional custom prompt path.
            model_config: Optional model configuration.
        """
        # Create the transfer_agent tool
        tools = [
            Tool(
                name="transfer_agent",
                description="Transfer the conversation to a specialized agent. Use this to route the user to the appropriate agent based on their intent.",
                function=self._transfer_agent,
                parameters={
                    "type": "object",
                    "properties": {
                        "target_agent_name": {
                            "type": "string",
                            "description": "The name of the agent to transfer to",
                            "enum": ["identity", "task_manager"],
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for the transfer (e.g., 'User wants to create a task', 'Authentication required')",
                        },
                    },
                    "required": ["target_agent_name", "reason"],
                },
            ),
        ]

        super().__init__(
            name="router",
            # Point to the updated file containing the optimized prompt
            system_prompt_path=system_prompt_path or Path("resources/prompts/router/v1_system.txt"),
            gemini_client=gemini_client,
            model_config=model_config or ModelConfig(
                temperature=0.3,  # Lower temperature for consistent routing
                response_modality="TEXT",  # Router uses text for decisions
            ),
            tools=tools,
        )

    def _transfer_agent(
        self,
        target_agent_name: str,
        reason: str,
    ) -> dict[str, Any]:
        """Transfer agent tool implementation.
        
        Note: This is a marker function. The actual routing is handled
        by the Orchestrator when it intercepts this tool call.
        
        Args:
            target_agent_name: Target agent to route to.
            reason: Reason for the transfer.
            
        Returns:
            Routing metadata (not used, Orchestrator intercepts).
        """
        return {
            "action": "transfer",
            "target": target_agent_name,
            "reason": reason,
        }

    async def process_signal(
        self,
        signal: Signal,
        context: GlobalContext,
    ) -> Response:
        """Process a signal and determine routing.

        Uses tool-based routing for immediate agent switching.

        Args:
            signal: The input signal.
            context: The global context.

        Returns:
            A routing response indicating the target agent.
        """
        # Quick check: if not authenticated, route to identity immediately
        if not context.is_authenticated():
            self._logger.info("unauthenticated_user_routing_to_identity")
            return Response.routing_response(
                session_id=signal.session_id,
                agent_name=self.name,
                decision=RoutingDecision(
                    thought_process="User is not authenticated",
                    route_to="identity",
                    handover_context="New session, authentication required",
                ),
            )

        # Get the text content for analysis
        text_content = self._extract_text(signal)

        # Quick keyword-based routing for common patterns
        if text_content:
            quick_route = self._quick_route(text_content.lower(), context)
            if quick_route:
                return Response.routing_response(
                    session_id=signal.session_id,
                    agent_name=self.name,
                    decision=quick_route,
                )

        # Use LLM with transfer_agent tool for complex routing
        return await self._llm_route(signal, context)

    def _extract_text(self, signal: Signal) -> str | None:
        """Extract text content from a signal."""
        if isinstance(signal, TextSignal):
            return signal.content
        if isinstance(signal, AudioSignal):
            return signal.metadata.get("transcription")
        return None

    def _quick_route(
        self,
        text: str,
        context: GlobalContext,
    ) -> RoutingDecision | None:
        """Quick keyword-based routing for common patterns.

        Args:
            text: Lowercase user input.
            context: The global context.

        Returns:
            RoutingDecision if pattern matched, None otherwise.
        """
        # Task-related keywords -> task_manager
        task_keywords = [
            "task", "todo", "remind", "schedule", "add", "create",
            "list", "show", "what's on", "what do i have", "meeting",
            "appointment", "deadline", "priority", "due", "mark",
            "complete", "done", "finish", "delete", "remove",
        ]

        for keyword in task_keywords:
            if keyword in text:
                return RoutingDecision(
                    thought_process=f"Detected task intent: '{keyword}'",
                    route_to="task_manager",
                    handover_context=text,
                )

        # Authentication keywords -> identity (even if authenticated, for re-auth)
        auth_keywords = ["who am i", "my name", "identify"]
        for keyword in auth_keywords:
            if keyword in text:
                return RoutingDecision(
                    thought_process=f"User asking about identity: '{keyword}'",
                    route_to="identity",
                )

        # Default to task_manager for authenticated users
        if context.is_authenticated():
            return RoutingDecision(
                thought_process="Authenticated user, defaulting to task manager",
                route_to="task_manager",
                handover_context=text,
            )

        return None

    async def _llm_route(
        self,
        signal: Signal,
        context: GlobalContext,
    ) -> Response:
        """Use LLM with transfer_agent tool for routing decisions.

        Args:
            signal: The input signal.
            context: The global context.

        Returns:
            Routing response.
        """
        rendered_prompt = self.render_prompt(context)
        
        # Build message
        text_content = self._extract_text(signal) or "[audio input]"
        self._conversation.append(LLMMessage(role="user", content=text_content))

        try:
            llm_response = await self.gemini_client.generate(
                messages=self._conversation,
                system_prompt=rendered_prompt,
                tools=self.get_tools_schema(),
            )

            # Check for transfer_agent tool call
            if llm_response.tool_calls:
                for tool_call in llm_response.tool_calls:
                    if tool_call.tool_name == "transfer_agent":
                        target = tool_call.arguments.get("target_agent_name", "task_manager")
                        reason = tool_call.arguments.get("reason", "")
                        
                        # Validate target
                        if target not in self.VALID_TARGETS:
                            target = "task_manager"
                        
                        return Response.routing_response(
                            session_id=signal.session_id,
                            agent_name=self.name,
                            decision=RoutingDecision(
                                thought_process=reason,
                                route_to=target,
                                handover_context=text_content,
                            ),
                        )

            # Fallback: parse text response for routing
            if llm_response.text:
                return self._parse_text_routing(llm_response.text, signal.session_id)

        except Exception as e:
            self._logger.error("llm_routing_error", error=str(e))

        # Default fallback
        return Response.routing_response(
            session_id=signal.session_id,
            agent_name=self.name,
            decision=RoutingDecision(
                thought_process="Fallback routing",
                route_to="task_manager",
            ),
        )

    def _parse_text_routing(
        self,
        text: str,
        session_id: str,
    ) -> Response:
        """Parse text response for routing decision (fallback)."""
        text_lower = text.lower()
        
        if "identity" in text_lower or "auth" in text_lower:
            target = "identity"
        else:
            target = "task_manager"
            
        return Response.routing_response(
            session_id=session_id,
            agent_name=self.name,
            decision=RoutingDecision(
                thought_process=f"Parsed from text: {text[:100]}",
                route_to=target,
            ),
        )

    async def on_enter(self, context: GlobalContext, handoff_data=None) -> None:
        """Called when router becomes active."""
        await super().on_enter(context, handoff_data)
        self._logger.info(
            "router_active",
            user_authenticated=context.is_authenticated(),
        )
        # Clear conversation for fresh routing
        self._conversation.clear()
