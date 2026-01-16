"""Twilio Voice Handler with VUI Optimizations.

Includes:
- Bridge audio for latency masking
- Warm handoffs with context propagation
- VUI metrics instrumentation
- Session persistence
"""

import asyncio
import audioop
import base64
import json
import time
import urllib.parse
from typing import Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect

from src.framework.core.context import GlobalContext, Platform, SessionContext, UserContext, HandoffData
from src.framework.core.io_handler import TwilioMediaStreamHandler, FillerType, FILLER_PHRASES
from src.framework.core.metrics import VUIMetrics
from src.framework.core.orchestrator import Orchestrator
from src.infrastructure.llm.gemini_audio import GeminiAudioClient, GeminiLiveSession
from src.infrastructure.database.service import DatabaseService

logger = structlog.get_logger(__name__)


class TwilioVoiceHandler:
    """Handler for Twilio Voice WebSocket connections with VUI optimizations."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        gemini_client: GeminiAudioClient,
        database_service: DatabaseService,
    ) -> None:
        self.orchestrator = orchestrator
        self.gemini_client = gemini_client
        self.database = database_service
        self._logger = structlog.get_logger(__name__)
        self._active_sessions: dict[str, GeminiLiveSession] = {}
        self._switch_agent_requested = False
        self._next_agent_name: str | None = None
        self._handoff_data: HandoffData | None = None
        self._start_event_received = asyncio.Event()
        self._restart_count = 0
        self._metrics: VUIMetrics | None = None
        self._session_stop_event = asyncio.Event()

    async def handle_call(
        self,
        websocket: WebSocket,
        call_sid: str,
        from_number: str,
    ) -> None:
        await websocket.accept()

        decoded_number = urllib.parse.unquote(from_number)
        self._logger.info("call_connected", call_sid=call_sid, initial_number=decoded_number)

        # Initialize metrics
        self._metrics = VUIMetrics(call_sid)

        self.orchestrator.context.clear_user()
        self._switch_agent_requested = False
        self._next_agent_name = None
        self._handoff_data = None
        self._start_event_received.clear()
        self._session_stop_event.clear()
        self._restart_count = 0

        session_context = SessionContext(
            session_id=call_sid,
            platform=Platform.TWILIO,
            metadata={
                "phone_number": decoded_number,
                "call_sid": call_sid,
                "stream_sid": None
            },
        )
        self.orchestrator.context.session = session_context

        io_handler = TwilioMediaStreamHandler(
            session_id=call_sid,
            websocket=websocket,
        )

        # Wait for start event
        receive_task = asyncio.create_task(self._receive_loop_for_start(websocket))
        try:
            await asyncio.wait_for(self._start_event_received.wait(), timeout=2.0)
            real_number = self.orchestrator.context.session.metadata.get("phone_number", decoded_number)
            self._logger.info("final_caller_id_resolved", number=real_number)
        except asyncio.TimeoutError:
            self._logger.warning("start_event_timeout", call_sid=call_sid)
            real_number = decoded_number

        # Determine initial agent based on authentication
        initial_agent = "identity"
        if real_number and real_number != "unknown":
            try:
                async with self.database.repositories() as (users, _):
                    user = await users.get_by_phone(real_number)
                    if user:
                        self.orchestrator.context.set_user(UserContext(
                            user_id=user.user_id,
                            phone_number=real_number,
                            full_name=user.full_name,
                            is_authenticated=True
                        ))
                        initial_agent = "task_manager"
                        self._logger.info("user_recognized", user=user.full_name)
            except Exception as e:
                self._logger.error("db_lookup_failed", error=str(e))

        await self.orchestrator.set_active_agent(initial_agent)

        try:
            while True:
                # Reset stop event for new session
                self._session_stop_event.clear()
                
                # Run the session
                await self._run_agent_session(
                    call_sid,
                    io_handler,
                    websocket,
                    receive_task
                )
                receive_task = None

                # If agent switch requested, continue with new agent
                if self._switch_agent_requested:
                    if self._next_agent_name:
                        # Pass handoff data for warm transition
                        await self.orchestrator.set_active_agent(
                            self._next_agent_name,
                            self._handoff_data
                        )
                    self._switch_agent_requested = False
                    self._next_agent_name = None
                    self._handoff_data = None
                    self._restart_count = 0
                    self._logger.info("switching_agent_session")
                    await asyncio.sleep(0.3)
                    continue
                
                # Session ended normally (call disconnected)
                self._logger.info("session_ended_normally")
                break

        except WebSocketDisconnect:
            self._logger.info("call_disconnected")
        except Exception as e:
            self._logger.error("call_error", error=str(e))
        finally:
            await io_handler.close()
            self._logger.info("call_ended")

    async def _receive_loop_for_start(self, websocket):
        try:
            while not self._start_event_received.is_set():
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("event") == "start":
                    stream_sid = msg["start"]["streamSid"]
                    self.orchestrator.context.session.metadata["stream_sid"] = stream_sid
                    custom_params = msg["start"].get("customParameters", {})
                    if "caller" in custom_params:
                        self.orchestrator.context.session.metadata["phone_number"] = custom_params["caller"]
                    self._start_event_received.set()
        except Exception:
            pass

    async def _run_agent_session(
        self,
        call_sid: str,
        io_handler: TwilioMediaStreamHandler,
        websocket: WebSocket,
        existing_receive_task: asyncio.Task | None = None
    ) -> None:
        """Run a single agent session until switch or disconnect."""
        live_session: GeminiLiveSession | None = None
        try:
            active_agent = self.orchestrator._active_agent
            if not active_agent: 
                return

            if hasattr(active_agent, "_current_context"):
                active_agent._current_context = self.orchestrator.context

            # Build system prompt with warm handoff context
            system_prompt = active_agent.render_prompt(self.orchestrator.context)
            tools = active_agent.get_tools_schema() if active_agent.tools else None

            self._logger.info("starting_gemini_session", agent=active_agent.name)
            
            # Record routing metrics
            if self._metrics:
                self._metrics.record_routing_start()
                
            live_session = await self.gemini_client.create_live_session(system_prompt, tools)
            await live_session.__aenter__()
            self._active_sessions[call_sid] = live_session
            
            if self._metrics:
                self._metrics.record_routing_complete(active_agent.name)

            # Send appropriate trigger message based on state
            if active_agent.name == "identity":
                await live_session.send_text("User connected. Greet them warmly and ask for their name to create an account.")
            elif active_agent.name == "task_manager":
                user_name = self.orchestrator.context.user.full_name
                greeting_done = self.orchestrator.context.session.greeting_completed
                
                # Pre-fetch task count for better UX
                task_count = 0
                try:
                    if self.orchestrator.context.is_authenticated():
                        async with self.database.repositories() as (_, tasks):
                            task_count = await tasks.get_open_count(self.orchestrator.context.user.user_id)
                except Exception as e:
                    self._logger.error("failed_to_get_task_count", error=str(e))

                task_info = f"They have {task_count} active tasks."
                if task_count == 0:
                    task_info = "They have no active tasks."
                
                if greeting_done:
                    # Warm handoff - don't re-greet
                    await live_session.send_text(
                        f"User {user_name} has been handed off to you. {task_info} "
                        f"Do NOT greet them again. Mention the task count briefly and ask if they need help with them."
                    )
                else:
                    await live_session.send_text(
                        f"User {user_name} connected. {task_info} "
                        f"Greet them, mention the {task_count} tasks they have, and ask if they need help with them."
                    )

            # Process the call stream until disconnect or agent switch
            await self._process_call_stream(
                websocket, 
                live_session, 
                call_sid,
                io_handler,
                existing_receive_task
            )

        except WebSocketDisconnect:
            self._logger.info("websocket_disconnected_in_session")
            raise
        except Exception as e:
            self._logger.error("session_error", error=str(e))
        finally:
            if live_session:
                live_session.stop()
                await live_session.__aexit__(None, None, None)
                self._active_sessions.pop(call_sid, None)

    async def _process_call_stream(
        self,
        websocket: WebSocket,
        live_session: GeminiLiveSession,
        call_sid: str,
        io_handler: TwilioMediaStreamHandler,
        existing_receive_task: asyncio.Task | None = None
    ) -> None:
        """Process bidirectional audio stream.
        
        Runs receive (from Twilio) and send (to Twilio) concurrently.
        Exits when:
        - WebSocket disconnects (call ended)
        - Agent switch is requested
        """
        if existing_receive_task and not existing_receive_task.done():
            existing_receive_task.cancel()

        # Create tasks for bidirectional streaming
        receive_task = asyncio.create_task(
            self._receive_from_twilio(websocket, live_session)
        )
        send_task = asyncio.create_task(
            self._send_to_twilio(websocket, live_session, io_handler)
        )
        
        try:
            # Wait for EITHER task to complete
            # - receive_task completes when Twilio disconnects
            # - send_task runs indefinitely unless agent switch or error
            done, pending = await asyncio.wait(
                [receive_task, send_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            # Check what completed
            for task in done:
                try:
                    task.result()  # Raise any exceptions
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self._logger.warning("task_error", error=str(e))

        finally:
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _receive_from_twilio(
        self,
        websocket: WebSocket,
        live_session: GeminiLiveSession,
    ) -> None:
        """Receive audio from Twilio and send to Gemini.
        
        Runs until WebSocket disconnects or agent switch requested.
        """
        try:
            while not self._switch_agent_requested:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=30.0  # Heartbeat timeout
                    )
                except asyncio.TimeoutError:
                    # No data for 30s, but connection might still be alive
                    continue
                    
                msg = json.loads(data)
                event = msg.get("event")

                if event == "media":
                    # Record user speech for TTFA
                    if self._metrics:
                        self._metrics.record_user_speech_end()
                        
                    # Convert and send audio to Gemini
                    audio = base64.b64decode(msg["media"]["payload"])
                    pcm_8k = audioop.ulaw2lin(audio, 2)
                    pcm_16k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, None)
                    await live_session.send_audio(pcm_16k)
                    
                elif event == "stop":
                    self._logger.info("twilio_stream_stop_received")
                    break
                    
        except WebSocketDisconnect:
            self._logger.info("twilio_websocket_disconnected")
            raise
        except Exception as e:
            self._logger.warning("receive_from_twilio_error", error=str(e))

    async def _send_to_twilio(
        self,
        websocket: WebSocket,
        live_session: GeminiLiveSession,
        io_handler: TwilioMediaStreamHandler,
    ) -> None:
        """Receive audio from Gemini and send to Twilio.
        
        Runs continuously, receiving responses from Gemini and
        forwarding audio to the caller.
        """
        resample_state = None
        first_audio_sent = False
        
        try:
            async for response in live_session.receive():
                if self._switch_agent_requested:
                    self._logger.info("agent_switch_requested_stopping_send")
                    break

                # Handle audio response
                if response.audio_data:
                    # Record first audio for TTFA
                    if not first_audio_sent and self._metrics:
                        self._metrics.record_first_audio_sent()
                        first_audio_sent = True
                    
                    # Convert 24kHz PCM to 8kHz mulaw for Twilio
                    pcm_in = response.audio_data
                    pcm_8k, resample_state = audioop.ratecv(
                        pcm_in, 2, 1, 24000, 8000, resample_state
                    )
                    mulaw = audioop.lin2ulaw(pcm_8k, 2)
                    payload = base64.b64encode(mulaw).decode("utf-8")
                    
                    stream_sid = self.orchestrator.context.session.metadata.get("stream_sid")
                    if stream_sid:
                        try:
                            await websocket.send_text(json.dumps({
                                "event": "media", 
                                "streamSid": stream_sid, 
                                "media": {"payload": payload}
                            }))
                        except Exception as e:
                            self._logger.warning("send_audio_error", error=str(e))
                            break
                        
                        # Reset silence tracker
                        if self._metrics:
                            self._metrics.reset_silence_tracker()
                        
                        # Mark greeting as completed after first response
                        if not self.orchestrator.context.session.greeting_completed:
                            self.orchestrator.context.session.mark_greeting_completed()

                # Handle tool calls
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        # Check for transfer_agent (routing)
                        if tool_call.tool_name == "transfer_agent":
                            await self._handle_transfer_agent(tool_call, io_handler, websocket)
                            return  # Exit to trigger agent switch
                            
                        # Execute regular tool
                        result = await self._execute_tool(tool_call, io_handler, websocket)
                        
                        # If tool caused agent switch, exit
                        if self._switch_agent_requested:
                            return
                            
                        # Send tool result back to Gemini
                        await live_session.send_tool_response(tool_call.call_id, result)
                            
        except asyncio.CancelledError:
            self._logger.debug("send_to_twilio_cancelled")
            raise
        except Exception as e:
            self._logger.warning("send_to_twilio_error", error=str(e))

    async def _handle_transfer_agent(
        self,
        tool_call: Any,
        io_handler: TwilioMediaStreamHandler,
        websocket: WebSocket,
    ) -> None:
        """Handle transfer_agent tool call for immediate routing.
        
        Sends bridge audio and prepares warm handoff data.
        """
        target = tool_call.arguments.get("target_agent_name", "task_manager")
        reason = tool_call.arguments.get("reason", "")
        
        self._logger.info(
            "transfer_agent_intercepted",
            target=target,
            reason=reason,
        )
        
        # Prepare warm handoff data
        user_intent = self.orchestrator.context.session.get_last_user_turn()
        self._handoff_data = self.orchestrator.context.session.prepare_handoff(
            target_agent=target,
            reason=reason,
            user_intent=user_intent,
        )
        
        # Update with user name if authenticated
        if self.orchestrator.context.is_authenticated():
            self._handoff_data.user_name = self.orchestrator.context.user.full_name
        
        # Signal agent switch
        self._switch_agent_requested = True
        self._next_agent_name = target

    async def _execute_tool(
        self,
        tool_call: Any,
        io_handler: TwilioMediaStreamHandler,
        websocket: WebSocket,
    ) -> Any:
        """Execute a tool and return the result."""
        agent = self.orchestrator._active_agent
        if not agent:
            return {"error": "No active agent"}
            
        tool = agent.get_tool(tool_call.tool_name)
        if not tool:
            return {"error": f"Tool '{tool_call.tool_name}' not found"}

        try:
            if hasattr(agent, "_current_context"):
                agent._current_context = self.orchestrator.context
            
            # Execute with timing
            start_time = time.perf_counter()
            self._logger.info("executing_tool", tool=tool_call.tool_name, args=tool_call.arguments)
            result = await tool.execute(**tool_call.arguments)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            self._logger.info("tool_executed", tool=tool_call.tool_name, duration_ms=round(duration_ms, 2))
            
            # Record metrics
            if self._metrics:
                self._metrics.record_tool_execution(tool_call.tool_name, duration_ms)
            
            # Handle user creation (special case for identity agent)
            if tool_call.tool_name == "create_user" and result.get("success"):
                self._logger.info("user_created_switching_to_task_manager", user=result.get("full_name"))
                
                self.orchestrator.context.set_user(UserContext(
                    user_id=result["user_id"],
                    phone_number=self.orchestrator.context.session.metadata["phone_number"],
                    full_name=result["full_name"],
                    is_authenticated=True
                ))
                
                # Prepare warm handoff to task_manager
                self._handoff_data = self.orchestrator.context.session.prepare_handoff(
                    target_agent="task_manager",
                    reason="User authenticated",
                )
                self._handoff_data.user_name = result["full_name"]
                
                self._switch_agent_requested = True
                self._next_agent_name = "task_manager"
                
                return {
                    "success": True,
                    "message": f"Account created for {result['full_name']}. Transferring to task manager."
                }
                
            return result
            
        except Exception as e:
            self._logger.error("tool_execution_error", tool=tool_call.tool_name, error=str(e))
            return {"error": str(e)}
