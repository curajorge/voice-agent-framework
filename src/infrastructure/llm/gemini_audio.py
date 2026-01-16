"""Gemini Native Audio Client (Session Persistence + Tool Response Fix)."""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
try:
    from websockets.exceptions import ConnectionClosed
except ImportError:
    class ConnectionClosed(Exception): pass

from src.infrastructure.llm.provider import LLMConfig, LLMMessage, LLMProvider, LLMResponse
from src.framework.core.signals import ToolCall

logger = structlog.get_logger(__name__)


class GeminiAudioClient(LLMProvider):
    """Gemini Native Audio Client."""

    DEFAULT_MODEL = "gemini-2.0-flash-exp"

    def __init__(
        self,
        api_key: str,
        config: LLMConfig | None = None,
    ) -> None:
        default_config = LLMConfig(
            model_name=self.DEFAULT_MODEL,
            temperature=0.7,
            voice_name="Kore",
            response_modality="AUDIO",
        )
        final_config = config if config else default_config
        super().__init__(final_config)

        self.api_key = api_key
        self._client: Any = None
        self._session: Any = None
        self._logger = structlog.get_logger(__name__)

    async def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
                from google.genai import types

                self._client = genai.Client(api_key=self.api_key)
                self._types = types
                self._logger.info("gemini_client_initialized")
            except ImportError:
                raise RuntimeError("google-genai package not installed")
        return self._client

    async def create_live_session(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> "GeminiLiveSession":
        client = await self._ensure_client()
        model_name = self.config.model_name

        config = self._types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=self._types.SpeechConfig(
                voice_config=self._types.VoiceConfig(
                    prebuilt_voice_config=self._types.PrebuiltVoiceConfig(
                        voice_name=self.config.voice_name,
                    )
                )
            ),
            system_instruction=self._types.Content(parts=[self._types.Part(text=system_prompt)]),
            tools=[self._types.Tool(function_declarations=tools)] if tools else None,
        )

        self._logger.info("connecting_live_session", model=model_name)
        session = client.aio.live.connect(
            model=model_name,
            config=config,
        )

        return GeminiLiveSession(session, self._types, self._logger)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._client = None
        self._logger.info("gemini_client_closed")

    async def generate(self, messages, system_prompt, tools=None): pass
    async def generate_stream(self, messages, system_prompt, tools=None): pass
    async def process_audio(self, audio_data, system_prompt, sample_rate=16000, tools=None): pass


class GeminiLiveSession:
    """Live bidirectional audio session with Gemini.
    
    Maintains a persistent connection for real-time audio streaming.
    The session stays open even when the model finishes speaking,
    allowing for continuous conversation.
    """

    def __init__(self, session: Any, types: Any, logger: Any) -> None:
        self._session_cm = session
        self._session: Any = None
        self._types = types
        self._logger = logger
        self._running = False
        self._turn_complete = asyncio.Event()
        # Track function names for tool responses
        self._pending_tool_calls: dict[str, str] = {}  # call_id -> function_name

    async def __aenter__(self) -> "GeminiLiveSession":
        self._session = await self._session_cm.__aenter__()
        self._running = True
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._running = False
        if self._session_cm:
            await self._session_cm.__aexit__(*args)

    async def send_audio(self, audio_data: bytes, mime_type: str = "audio/pcm;rate=16000") -> None:
        """Send audio data to the session.
        
        Non-blocking - errors are logged but don't stop the session.
        """
        if not self._session or not self._running: 
            return
        try:
            await self._session.send(input={"data": audio_data, "mime_type": mime_type})
        except Exception as e:
            # Don't stop the session for transient audio errors
            pass

    async def send_text(self, text: str) -> None:
        """Send text input to trigger a response."""
        if not self._session or not self._running: 
            return
        try:
            await self._session.send(input=text, end_of_turn=True)
        except Exception as e1:
            try:
                await self._session.send(input={"text": text}, end_of_turn=True)
            except Exception as e2:
                self._logger.error("send_text_failed", error=str(e2))

    async def receive(self) -> AsyncGenerator[LLMResponse, None]:
        """Receive responses from the session.
        
        This generator keeps running as long as the session is active,
        even when the model finishes speaking. It will yield responses
        whenever the model produces output.
        """
        if not self._session: 
            return

        while self._running:
            try:
                # Get responses for the current turn
                turn_had_content = False
                
                async for response in self._session.receive():
                    if not self._running: 
                        return

                    audio_data: bytes | None = None
                    text_content: str | None = None
                    tool_calls: list[ToolCall] = []

                    # Extract audio data
                    if hasattr(response, "data") and response.data:
                        audio_data = response.data
                        turn_had_content = True
                    
                    # Extract from server_content
                    if hasattr(response, "server_content") and response.server_content:
                        sc = response.server_content
                        if hasattr(sc, "model_turn") and sc.model_turn:
                            for part in sc.model_turn.parts:
                                if hasattr(part, "inline_data") and part.inline_data:
                                    audio_data = part.inline_data.data
                                    turn_had_content = True
                                if hasattr(part, "text") and part.text:
                                    text_content = part.text
                                    turn_had_content = True
                                if hasattr(part, "function_call") and part.function_call:
                                    fc = part.function_call
                                    call_id = getattr(fc, 'id', None) or fc.name
                                    # Store function name for later response
                                    self._pending_tool_calls[call_id] = fc.name
                                    tool_calls.append(ToolCall(
                                        tool_name=fc.name,
                                        arguments=dict(fc.args) if fc.args else {},
                                        call_id=call_id
                                    ))
                                    turn_had_content = True
                        
                        # Check for turn completion signal
                        if hasattr(sc, "turn_complete") and sc.turn_complete:
                            self._turn_complete.set()

                    # Extract tool calls from response.tool_call
                    if hasattr(response, "tool_call") and response.tool_call:
                        for fc in response.tool_call.function_calls:
                            call_id = getattr(fc, 'id', None) or fc.name
                            self._pending_tool_calls[call_id] = fc.name
                            tool_calls.append(ToolCall(
                                tool_name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                                call_id=call_id
                            ))
                            turn_had_content = True

                    # Only yield if we have content
                    if audio_data or text_content or tool_calls:
                        yield LLMResponse(
                            text=text_content,
                            audio_data=audio_data,
                            tool_calls=tool_calls if tool_calls else None,
                        )
                
                # Model finished its turn - this is NORMAL, not an error!
                # The session is still open and ready for more input.
                
                if turn_had_content:
                    self._logger.debug("model_turn_complete_waiting_for_next")
                
                # Small yield to prevent tight loop and allow other tasks to run
                await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                # Task was cancelled - exit cleanly
                self._logger.debug("receive_cancelled")
                return
                
            except Exception as e:
                error_str = str(e)
                # WebSocket closed normally
                if "1000" in error_str or "1001" in error_str:
                    self._logger.debug("websocket_closed_normally")
                    return
                    
                # Log other errors but keep trying
                self._logger.warning("gemini_receive_error", error=error_str)
                await asyncio.sleep(0.1)

    async def send_tool_response(self, call_id: str, result: Any) -> None:
        """Send a tool/function response back to the model.
        
        Uses the Google GenAI types to construct a LiveClientToolResponse,
        which is then passed to the 'input' argument of session.send().
        """
        if not self._session or not self._running: 
            return
            
        # Get the function name from our tracking dict
        function_name = self._pending_tool_calls.pop(call_id, call_id)
        
        try:
            # Construct the typed objects required by the library
            fn_response = self._types.FunctionResponse(
                name=function_name,
                id=call_id,
                response=result
            )
            
            tool_msg = self._types.LiveClientToolResponse(
                function_responses=[fn_response]
            )
            
            # Send via 'input' argument. This is the fix for the previous TypeError.
            await self._session.send(input=tool_msg)
            self._logger.debug("tool_response_sent", function=function_name)
                
        except Exception as e:
            self._logger.error("tool_response_error", function=function_name, error=str(e))

    def stop(self) -> None:
        """Signal the session to stop."""
        self._running = False
