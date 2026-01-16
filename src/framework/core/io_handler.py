"""IO Handler interfaces for input/output abstraction.

The IO system abstracts the source of input (CLI, WebSocket, Twilio)
from the agent processing logic.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

import structlog

from src.framework.core.signals import AudioSignal, Response, Signal, TextSignal

logger = structlog.get_logger(__name__)


class FillerType(str, Enum):
    """Types of filler audio for latency masking."""

    ROUTING = "routing"  # "One moment..."
    TOOL_EXECUTION = "tool_execution"  # "Let me check..."
    THINKING = "thinking"  # "Hmm, let me think..."
    CREATING = "creating"  # "Let me add that..."
    SEARCHING = "searching"  # "Looking that up..."


# Pre-defined filler phrases (text-to-speech will be used)
FILLER_PHRASES: dict[FillerType, list[str]] = {
    FillerType.ROUTING: [
        "One moment please.",
        "Just a moment.",
        "Let me connect you.",
    ],
    FillerType.TOOL_EXECUTION: [
        "Let me check on that.",
        "One second while I look that up.",
        "Checking now.",
    ],
    FillerType.THINKING: [
        "Let me think about that.",
        "Hmm, good question.",
    ],
    FillerType.CREATING: [
        "Let me add that for you.",
        "Creating that now.",
        "Adding that to your list.",
    ],
    FillerType.SEARCHING: [
        "Looking that up for you.",
        "Searching now.",
        "Let me find that.",
    ],
}


class IOHandler(ABC):
    """Abstract base class for input/output handlers.

    IOHandlers manage the communication channel between users and agents.
    They handle streaming input and output, regardless of the underlying
    transport (CLI, WebSocket, Twilio Media Streams).
    """

    def __init__(self, session_id: str) -> None:
        """Initialize the IO handler.

        Args:
            session_id: The session identifier for this handler.
        """
        self.session_id = session_id
        self._logger = structlog.get_logger(__name__).bind(
            session_id=session_id,
            handler_type=self.__class__.__name__,
        )
        self._filler_task: asyncio.Task | None = None
        self._filler_cancelled = asyncio.Event()

    @abstractmethod
    async def stream_input(self) -> AsyncGenerator[Signal, None]:
        """Stream input signals from the user.

        Yields audio chunks or text messages as they arrive.

        Yields:
            Signal objects (AudioSignal or TextSignal).
        """
        pass

    @abstractmethod
    async def stream_output(self, response: Response) -> None:
        """Send a response back to the user.

        Args:
            response: The response to send.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the IO handler and release resources."""
        pass

    async def send_text(self, content: str, agent_name: str = "system") -> None:
        """Convenience method to send text output.

        Args:
            content: The text content to send.
            agent_name: Name of the sending agent.
        """
        response = Response.text_response(
            session_id=self.session_id,
            agent_name=agent_name,
            content=content,
        )
        await self.stream_output(response)

    async def send_audio(
        self,
        audio_data: bytes,
        agent_name: str = "system",
    ) -> None:
        """Convenience method to send audio output.

        Args:
            audio_data: The audio bytes to send.
            agent_name: Name of the sending agent.
        """
        response = Response.audio_response(
            session_id=self.session_id,
            agent_name=agent_name,
            audio_data=audio_data,
        )
        await self.stream_output(response)

    async def send_filler(
        self,
        filler_type: FillerType,
        agent_name: str = "system",
    ) -> None:
        """Send filler audio/text to mask latency.
        
        This is non-blocking and interruptible. The filler will stop
        if cancel_filler() is called or if the user speaks.
        
        Args:
            filler_type: Type of filler to play.
            agent_name: Name of the sending agent.
        """
        import random
        
        phrases = FILLER_PHRASES.get(filler_type, FILLER_PHRASES[FillerType.THINKING])
        phrase = random.choice(phrases)
        
        self._logger.debug(
            "sending_filler",
            filler_type=filler_type.value,
            phrase=phrase,
        )
        
        # Cancel any existing filler
        await self.cancel_filler()
        
        # Reset cancellation flag
        self._filler_cancelled.clear()
        
        # Send the filler (subclasses can override for audio)
        await self._send_filler_impl(phrase, agent_name)

    async def _send_filler_impl(self, phrase: str, agent_name: str) -> None:
        """Implementation of filler sending.
        
        Subclasses can override this to send actual audio.
        
        Args:
            phrase: The filler phrase to send.
            agent_name: Name of the sending agent.
        """
        # Default: send as text
        await self.send_text(phrase, agent_name)

    async def cancel_filler(self) -> None:
        """Cancel any playing filler audio."""
        self._filler_cancelled.set()
        if self._filler_task and not self._filler_task.done():
            self._filler_task.cancel()
            try:
                await self._filler_task
            except asyncio.CancelledError:
                pass
        self._filler_task = None

    def is_filler_cancelled(self) -> bool:
        """Check if filler has been cancelled.
        
        Returns:
            True if filler was cancelled.
        """
        return self._filler_cancelled.is_set()


class CLIHandler(IOHandler):
    """Command-line interface handler for testing.

    Reads text input from stdin and prints responses to stdout.
    """

    def __init__(self, session_id: str) -> None:
        """Initialize the CLI handler.

        Args:
            session_id: The session identifier.
        """
        super().__init__(session_id)
        self._running = True

    async def stream_input(self) -> AsyncGenerator[Signal, None]:
        """Stream text input from the command line.

        Yields:
            TextSignal objects from user input.
        """
        import sys

        while self._running:
            try:
                # Use asyncio to read from stdin without blocking
                loop = asyncio.get_event_loop()
                line = await loop.run_in_executor(
                    None,
                    lambda: input("\n[You]: ").strip(),
                )

                if line.lower() in ("exit", "quit", "bye"):
                    self._running = False
                    break

                if line:
                    yield TextSignal(
                        session_id=self.session_id,
                        content=line,
                    )

            except EOFError:
                self._running = False
                break
            except KeyboardInterrupt:
                self._running = False
                break

    async def stream_output(self, response: Response) -> None:
        """Print response to stdout.

        Args:
            response: The response to display.
        """
        if response.text_content:
            print(f"\n[{response.agent_name}]: {response.text_content}")
        elif response.audio_data:
            print(f"\n[{response.agent_name}]: [Audio Response - {len(response.audio_data)} bytes]")
        elif response.routing_decision:
            self._logger.debug(
                "routing_decision",
                route_to=response.routing_decision.route_to,
            )

    async def close(self) -> None:
        """Close the CLI handler."""
        self._running = False
        print("\n[System]: Session ended.")


class WebSocketHandler(IOHandler):
    """WebSocket handler for real-time audio streaming.

    Handles bidirectional audio streaming over WebSocket connections.
    """

    def __init__(
        self,
        session_id: str,
        websocket: Any,  # WebSocket from FastAPI/Starlette
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the WebSocket handler.

        Args:
            session_id: The session identifier.
            websocket: The WebSocket connection object.
            sample_rate: Audio sample rate in Hz.
        """
        super().__init__(session_id)
        self.websocket = websocket
        self.sample_rate = sample_rate
        self._running = True

    async def stream_input(self) -> AsyncGenerator[Signal, None]:
        """Stream audio input from the WebSocket.

        Yields:
            AudioSignal objects from the WebSocket stream.
        """
        import json

        while self._running:
            try:
                data = await self.websocket.receive()

                if data.get("type") == "websocket.disconnect":
                    self._running = False
                    break

                if "bytes" in data:
                    # Raw audio bytes
                    yield AudioSignal(
                        session_id=self.session_id,
                        audio_data=data["bytes"],
                        sample_rate=self.sample_rate,
                    )
                elif "text" in data:
                    # JSON message or text
                    try:
                        msg = json.loads(data["text"])
                        if msg.get("type") == "audio":
                            import base64
                            audio_bytes = base64.b64decode(msg["data"])
                            yield AudioSignal(
                                session_id=self.session_id,
                                audio_data=audio_bytes,
                                sample_rate=msg.get("sample_rate", self.sample_rate),
                            )
                        elif msg.get("type") == "text":
                            yield TextSignal(
                                session_id=self.session_id,
                                content=msg["content"],
                            )
                    except json.JSONDecodeError:
                        # Plain text message
                        yield TextSignal(
                            session_id=self.session_id,
                            content=data["text"],
                        )

            except Exception as e:
                self._logger.error("websocket_receive_error", error=str(e))
                self._running = False
                break

    async def stream_output(self, response: Response) -> None:
        """Send response over WebSocket.

        Args:
            response: The response to send.
        """
        import base64

        try:
            if response.audio_data:
                # Send audio as base64-encoded JSON
                await self.websocket.send_json({
                    "type": "audio",
                    "data": base64.b64encode(response.audio_data).decode("utf-8"),
                    "agent": response.agent_name,
                })
            elif response.text_content:
                await self.websocket.send_json({
                    "type": "text",
                    "content": response.text_content,
                    "agent": response.agent_name,
                })

        except Exception as e:
            self._logger.error("websocket_send_error", error=str(e))

    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        try:
            await self.websocket.close()
        except Exception:
            pass


class TwilioMediaStreamHandler(IOHandler):
    """Handler for Twilio Media Streams (phone calls).

    Processes bidirectional audio streams from Twilio's WebSocket
    Media Streams API.
    """

    def __init__(
        self,
        session_id: str,
        websocket: Any,
        stream_sid: str | None = None,
    ) -> None:
        """Initialize the Twilio Media Stream handler.

        Args:
            session_id: The session identifier (call SID).
            websocket: The WebSocket connection.
            stream_sid: The Twilio stream SID.
        """
        super().__init__(session_id)
        self.websocket = websocket
        self.stream_sid = stream_sid
        self._running = True
        self._audio_buffer: list[bytes] = []

    async def stream_input(self) -> AsyncGenerator[Signal, None]:
        """Stream audio from Twilio Media Stream.

        Twilio sends mulaw-encoded audio at 8kHz. We accumulate chunks
        and yield them as AudioSignals.

        Yields:
            AudioSignal objects from the phone call.
        """
        import base64
        import json

        while self._running:
            try:
                data = await self.websocket.receive_text()
                msg = json.loads(data)

                event_type = msg.get("event")

                if event_type == "connected":
                    self._logger.info("twilio_stream_connected")

                elif event_type == "start":
                    self.stream_sid = msg["start"]["streamSid"]
                    self._logger.info(
                        "twilio_stream_started",
                        stream_sid=self.stream_sid,
                    )

                elif event_type == "media":
                    # Decode mulaw audio from Twilio
                    audio_payload = msg["media"]["payload"]
                    audio_bytes = base64.b64decode(audio_payload)

                    yield AudioSignal(
                        session_id=self.session_id,
                        audio_data=audio_bytes,
                        sample_rate=8000,  # Twilio uses 8kHz
                        encoding="MULAW",
                        metadata={
                            "stream_sid": self.stream_sid,
                            "timestamp": msg["media"].get("timestamp"),
                        },
                    )

                elif event_type == "stop":
                    self._logger.info("twilio_stream_stopped")
                    self._running = False
                    break

            except Exception as e:
                self._logger.error("twilio_receive_error", error=str(e))
                self._running = False
                break

    async def stream_output(self, response: Response) -> None:
        """Send audio back to Twilio.

        Args:
            response: The response containing audio to send.
        """
        import base64
        import json

        if response.audio_data and self.stream_sid:
            try:
                # Twilio expects base64-encoded mulaw audio
                payload = base64.b64encode(response.audio_data).decode("utf-8")

                await self.websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": payload,
                    },
                }))

            except Exception as e:
                self._logger.error("twilio_send_error", error=str(e))

    async def _send_filler_impl(self, phrase: str, agent_name: str) -> None:
        """Send filler as synthesized speech to Twilio.
        
        For Twilio, we need to send actual audio. This implementation
        sends a text marker; the actual TTS should be handled by the
        Twilio handler or a TTS service.
        
        Args:
            phrase: The filler phrase.
            agent_name: Name of the sending agent.
        """
        # Log the filler request - actual audio generation happens in TwilioVoiceHandler
        self._logger.info(
            "filler_requested",
            phrase=phrase,
            stream_sid=self.stream_sid,
        )
        # The TwilioVoiceHandler will intercept this and generate audio

    async def send_mark(self, name: str) -> None:
        """Send a mark event to Twilio for synchronization.

        Args:
            name: The mark name.
        """
        import json

        if self.stream_sid:
            await self.websocket.send_text(json.dumps({
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": name},
            }))

    async def clear_audio(self) -> None:
        """Send clear event to stop current audio playback."""
        import json

        if self.stream_sid:
            await self.websocket.send_text(json.dumps({
                "event": "clear",
                "streamSid": self.stream_sid,
            }))

    async def close(self) -> None:
        """Close the Twilio stream."""
        self._running = False
        try:
            await self.websocket.close()
        except Exception:
            pass
