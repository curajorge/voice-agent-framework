"""Orchestrator - The core event loop and agent coordinator.

The Orchestrator is the kernel of the framework. It manages the event loop,
coordinates agents, handles routing, and processes tool executions.

Includes VUI optimizations:
- Tool-based routing interception for immediate agent switching
- Bridge audio for latency masking
- Warm handoff context propagation
- VUI metrics instrumentation
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import structlog

from src.framework.core.agent import AbstractAgent, Tool
from src.framework.core.context import GlobalContext, HandoffData
from src.framework.core.exceptions import (
    AgentException,
    PriorityIntervention,
    RoutingException,
    ToolExecutionException,
)
from src.framework.core.io_handler import IOHandler, FillerType
from src.framework.core.metrics import VUIMetrics
from src.framework.core.observer import InterventionObserver
from src.framework.core.signals import (
    Response,
    ResponseType,
    Signal,
    ToolCall,
)

logger = structlog.get_logger(__name__)


# Special tool name for routing
TRANSFER_AGENT_TOOL = "transfer_agent"


class Orchestrator:
    """The central coordination engine for the agent framework.

    The Orchestrator manages:
    - The async event loop
    - Agent registry and routing
    - Tool execution with latency masking
    - Context management and warm handoffs
    - Intervention handling
    - VUI metrics collection
    """

    def __init__(
        self,
        context: GlobalContext | None = None,
        observer: InterventionObserver | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            context: The global context (created if not provided).
            observer: The intervention observer (created if not provided).
        """
        self.context = context or GlobalContext()
        self.observer = observer or InterventionObserver()
        self._agents: dict[str, AbstractAgent] = {}
        self._active_agent: AbstractAgent | None = None
        self._io_handler: IOHandler | None = None
        self._running = False
        self._metrics: VUIMetrics | None = None
        self._logger = structlog.get_logger(__name__)
        
        # Routing state for transfer_agent interception
        self._pending_transfer: dict[str, Any] | None = None

    def register_agent(self, agent: AbstractAgent) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent: The agent to register.

        Raises:
            ValueError: If an agent with the same name is already registered.
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' is already registered")

        self._agents[agent.name] = agent
        self._logger.info("agent_registered", agent_name=agent.name)

        # Update available agents in context
        if agent.name not in self.context.available_agents:
            self.context.available_agents.append(agent.name)

    def get_agent(self, name: str) -> AbstractAgent | None:
        """Get a registered agent by name.

        Args:
            name: The agent name.

        Returns:
            The agent if found, None otherwise.
        """
        return self._agents.get(name)

    async def set_active_agent(
        self,
        agent_name: str,
        handoff_data: HandoffData | None = None,
    ) -> None:
        """Switch the active agent with warm handoff support.

        Args:
            agent_name: Name of the agent to activate.
            handoff_data: Optional handoff data for warm transition.

        Raises:
            RoutingException: If the agent is not registered.
        """
        if agent_name not in self._agents:
            raise RoutingException(
                message=f"Agent '{agent_name}' not found",
                source_agent=self.context.session.active_agent,
                target_agent=agent_name,
            )

        # Start routing metrics
        if self._metrics:
            self._metrics.record_routing_start()

        # Call lifecycle hooks
        if self._active_agent:
            await self._active_agent.on_exit(self.context)

        previous_agent = self._active_agent
        self._active_agent = self._agents[agent_name]
        self.context.session.switch_agent(agent_name)

        # Warm handoff: pass context to new agent
        await self._active_agent.on_enter(self.context, handoff_data)

        # Record routing completion
        if self._metrics:
            self._metrics.record_routing_complete(agent_name)

        self._logger.info(
            "agent_switched",
            from_agent=previous_agent.name if previous_agent else None,
            to_agent=agent_name,
            has_handoff=handoff_data is not None,
        )

    async def run(self, io_handler: IOHandler) -> None:
        """Run the main event loop.

        This is the primary entry point for starting the orchestrator.
        It processes input signals and coordinates agent responses.

        Args:
            io_handler: The IO handler for input/output.
        """
        self._io_handler = io_handler
        self._running = True
        self._metrics = VUIMetrics(self.context.session.session_id)

        # Ensure we have a default agent
        if not self._active_agent:
            if "router" in self._agents:
                await self.set_active_agent("router")
            elif self._agents:
                first_agent = next(iter(self._agents.keys()))
                await self.set_active_agent(first_agent)
            else:
                raise RuntimeError("No agents registered")

        self._logger.info(
            "orchestrator_started",
            session_id=self.context.session.session_id,
            active_agent=self._active_agent.name,
        )

        try:
            await self._event_loop(io_handler)
        except Exception as e:
            self._logger.error("orchestrator_error", error=str(e))
            raise
        finally:
            self._running = False
            await io_handler.close()
            self._logger.info("orchestrator_stopped")

    async def _event_loop(self, io_handler: IOHandler) -> None:
        """The main event processing loop.

        Args:
            io_handler: The IO handler for input/output.
        """
        # Create the signal stream with observer wrapping
        signal_stream = self.observer.watch(io_handler.stream_input())

        # Start timeout checker
        timeout_task = asyncio.create_task(self._timeout_checker())
        
        # Start silence monitor
        silence_task = asyncio.create_task(self._silence_monitor())

        try:
            async for signal in signal_stream:
                if not self._running:
                    break

                try:
                    # Record user speech end for TTFA
                    if self._metrics:
                        self._metrics.record_user_speech_end()

                    # Process the signal through the active agent
                    response = await self._process_signal(signal)

                    # Handle the response
                    await self._handle_response(response, io_handler)

                except PriorityIntervention as intervention:
                    await self._handle_intervention(intervention, io_handler)

                except AgentException as e:
                    self._logger.error(
                        "agent_error",
                        agent=e.agent_name,
                        error=e.message,
                    )
                    if e.recoverable:
                        await io_handler.send_text(
                            "I encountered an issue. Let me try again.",
                            agent_name="system",
                        )
                    else:
                        raise

        except PriorityIntervention as intervention:
            await self._handle_intervention(intervention, io_handler)

        finally:
            timeout_task.cancel()
            silence_task.cancel()
            try:
                await asyncio.gather(timeout_task, silence_task, return_exceptions=True)
            except asyncio.CancelledError:
                pass

    async def _process_signal(self, signal: Signal) -> Response:
        """Process a signal through the active agent.

        Args:
            signal: The input signal.

        Returns:
            The agent's response.
        """
        if not self._active_agent:
            raise RuntimeError("No active agent")

        # Update context
        self.context.refresh_time()

        # Add to conversation history
        if hasattr(signal, "content"):
            self.context.session.add_turn(
                role="user",
                content=getattr(signal, "content", "[audio]"),
            )

        # Check authentication requirement
        if not self.context.is_authenticated():
            if self._active_agent.name != "identity":
                if "identity" in self._agents:
                    # Prepare handoff data
                    handoff = self.context.session.prepare_handoff(
                        target_agent="identity",
                        reason="Authentication required",
                    )
                    await self.set_active_agent("identity", handoff)

        # Process signal
        response = await self._active_agent.process_signal(signal, self.context)

        return response

    async def _handle_response(
        self,
        response: Response,
        io_handler: IOHandler,
    ) -> None:
        """Handle an agent response.

        Args:
            response: The response to handle.
            io_handler: The IO handler for output.
        """
        # Handle tool calls with transfer_agent interception
        if response.requires_tool_execution and response.tool_calls:
            # Check for transfer_agent tool call (routing)
            for tool_call in response.tool_calls:
                if tool_call.tool_name == TRANSFER_AGENT_TOOL:
                    await self._handle_transfer_agent(
                        tool_call,
                        response,
                        io_handler,
                    )
                    return
            
            # Regular tool execution
            response = await self._execute_tools(response, io_handler)

        # Handle routing decisions
        if response.response_type == ResponseType.ROUTING and response.routing_decision:
            await self._handle_routing_decision(response, io_handler)
            return

        # Send response to user
        if response.text_content or response.audio_data:
            # Record first audio for TTFA
            if response.audio_data and self._metrics:
                self._metrics.record_first_audio_sent()
                self._metrics.reset_silence_tracker()

            await io_handler.stream_output(response)

            # Mark greeting as completed after first assistant response
            if not self.context.session.greeting_completed:
                self.context.session.mark_greeting_completed()

            # Add to conversation history
            content = response.text_content or "[audio response]"
            self.context.session.add_turn(
                role="assistant",
                content=content,
                agent_name=response.agent_name,
            )

    async def _handle_transfer_agent(
        self,
        tool_call: ToolCall,
        response: Response,
        io_handler: IOHandler,
    ) -> None:
        """Handle transfer_agent tool call for immediate routing.
        
        This intercepts the transfer_agent tool call and triggers
        immediate agent switching without completing the tool loop.
        
        Args:
            tool_call: The transfer_agent tool call.
            response: The original response.
            io_handler: The IO handler.
        """
        target = tool_call.arguments.get("target_agent_name", "task_manager")
        reason = tool_call.arguments.get("reason", "")
        
        self._logger.info(
            "transfer_agent_intercepted",
            target=target,
            reason=reason,
        )
        
        # Send filler audio for latency masking
        await io_handler.send_filler(FillerType.ROUTING, self._active_agent.name if self._active_agent else "system")
        
        # Prepare warm handoff data
        user_intent = self.context.session.get_last_user_turn()
        handoff = self.context.session.prepare_handoff(
            target_agent=target,
            reason=reason,
            user_intent=user_intent,
        )
        
        # Update handoff with user name if authenticated
        if self.context.is_authenticated():
            handoff.user_name = self.context.user.full_name
        
        # Cancel filler before switching
        await io_handler.cancel_filler()
        
        # Switch agent with warm handoff
        if target in self._agents:
            await self.set_active_agent(target, handoff)
        else:
            self._logger.warning("invalid_transfer_target", target=target)
            await self.set_active_agent("task_manager", handoff)

    async def _handle_routing_decision(
        self,
        response: Response,
        io_handler: IOHandler,
    ) -> None:
        """Handle routing decision from response.
        
        Args:
            response: The routing response.
            io_handler: The IO handler.
        """
        decision = response.routing_decision
        if not decision:
            return
            
        target = decision.route_to
        self._logger.info(
            "routing_decision",
            target=target,
            thought=decision.thought_process,
        )

        if target in self._agents:
            # Send filler for latency masking
            await io_handler.send_filler(
                FillerType.ROUTING,
                response.agent_name,
            )
            
            # Prepare handoff
            handoff = self.context.session.prepare_handoff(
                target_agent=target,
                reason=decision.thought_process,
                user_intent=decision.handover_context,
            )
            
            if self.context.is_authenticated():
                handoff.user_name = self.context.user.full_name
            
            # Cancel filler and switch
            await io_handler.cancel_filler()
            await self.set_active_agent(target, handoff)

    async def _execute_tools(
        self,
        response: Response,
        io_handler: IOHandler,
    ) -> Response:
        """Execute tool calls from an agent response with latency masking.

        Args:
            response: The response containing tool calls.
            io_handler: The IO handler for filler audio.

        Returns:
            Updated response after tool execution.
        """
        if not self._active_agent:
            raise RuntimeError("No active agent")

        results: list[dict[str, Any]] = []

        for tool_call in response.tool_calls:
            # Skip transfer_agent (handled separately)
            if tool_call.tool_name == TRANSFER_AGENT_TOOL:
                continue
                
            try:
                # Check if tool is slow and send filler
                if self._active_agent.is_tool_slow(tool_call.tool_name):
                    filler_type = self._get_filler_type_for_tool(tool_call.tool_name)
                    await io_handler.send_filler(
                        filler_type,
                        self._active_agent.name,
                    )
                
                # Execute tool with timing
                start_time = time.perf_counter()
                result = await self._execute_single_tool(tool_call)
                duration_ms = (time.perf_counter() - start_time) * 1000
                
                # Record metrics
                if self._metrics:
                    self._metrics.record_tool_execution(
                        tool_call.tool_name,
                        duration_ms,
                    )
                
                # Cancel filler
                await io_handler.cancel_filler()
                
                results.append({
                    "tool_name": tool_call.tool_name,
                    "call_id": tool_call.call_id,
                    "success": True,
                    "result": result,
                })

                # Let agent handle the result
                handler_response = await self._active_agent.handle_tool_result(
                    tool_call.tool_name,
                    result,
                    self.context,
                )
                if handler_response:
                    return handler_response

            except ToolExecutionException as e:
                # Cancel filler on error
                await io_handler.cancel_filler()
                
                results.append({
                    "tool_name": tool_call.tool_name,
                    "call_id": tool_call.call_id,
                    "success": False,
                    "error": e.message,
                })
                
                # Send apology for failed tool
                await io_handler.send_text(
                    "I'm having trouble with that. Let me try something else.",
                    agent_name=self._active_agent.name,
                )

        # Return response with tool results in metadata
        response.metadata["tool_results"] = results
        response.is_final = True
        return response

    def _get_filler_type_for_tool(self, tool_name: str) -> FillerType:
        """Get appropriate filler type for a tool.
        
        Args:
            tool_name: Name of the tool.
            
        Returns:
            Appropriate FillerType.
        """
        if "create" in tool_name or "add" in tool_name:
            return FillerType.CREATING
        if "search" in tool_name or "get" in tool_name or "list" in tool_name:
            return FillerType.SEARCHING
        return FillerType.TOOL_EXECUTION

    async def _execute_single_tool(self, tool_call: ToolCall) -> Any:
        """Execute a single tool call.

        Args:
            tool_call: The tool call to execute.

        Returns:
            The tool execution result.

        Raises:
            ToolExecutionException: If execution fails.
        """
        if not self._active_agent:
            raise RuntimeError("No active agent")

        tool = self._active_agent.get_tool(tool_call.tool_name)
        if not tool:
            raise ToolExecutionException(
                message=f"Tool '{tool_call.tool_name}' not found",
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
            )

        try:
            return await tool.execute(**tool_call.arguments)
        except Exception as e:
            raise ToolExecutionException(
                message=str(e),
                tool_name=tool_call.tool_name,
                arguments=tool_call.arguments,
            )

    async def _handle_intervention(
        self,
        intervention: PriorityIntervention,
        io_handler: IOHandler,
    ) -> None:
        """Handle a priority intervention.

        Args:
            intervention: The intervention that was raised.
            io_handler: The IO handler for output.
        """
        self._logger.info(
            "intervention_handling",
            type=intervention.intervention_type,
            target=intervention.target_agent,
        )

        # Cancel any filler audio
        await io_handler.cancel_filler()

        # Clear any pending audio
        if hasattr(io_handler, "clear_audio"):
            await io_handler.clear_audio()

        # Route to target agent if specified
        if intervention.target_agent and intervention.target_agent in self._agents:
            await self.set_active_agent(intervention.target_agent)
        elif "router" in self._agents:
            await self.set_active_agent("router")

        # Notify user
        await io_handler.send_text(
            "I understand. How can I help you?",
            agent_name=self._active_agent.name if self._active_agent else "system",
        )

        # Reset observer
        self.observer.reset()

    async def _timeout_checker(self) -> None:
        """Background task to check for inactivity timeout."""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                await self.observer.check_timeout()
            except PriorityIntervention:
                # Timeout will be handled in main loop
                pass
            except asyncio.CancelledError:
                break

    async def _silence_monitor(self) -> None:
        """Background task to monitor for excessive silence."""
        while self._running:
            try:
                await asyncio.sleep(1)  # Check every second
                if self._metrics:
                    self._metrics.check_silence()
            except asyncio.CancelledError:
                break

    def stop(self) -> None:
        """Stop the orchestrator."""
        self._running = False
        self.observer.cancel()
