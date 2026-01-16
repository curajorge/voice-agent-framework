"""FastAPI application factory and route definitions."""

import asyncio
import sys
import urllib.parse
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from src.server.config import Settings, get_settings
from src.server.twilio_handler import TwilioVoiceHandler
from src.framework.core.orchestrator import Orchestrator
from src.framework.core.context import GlobalContext, UserContext
from src.framework.core.io_handler import WebSocketHandler
from src.infrastructure.llm.gemini_audio import GeminiAudioClient
from src.infrastructure.llm.provider import LLMConfig
from src.infrastructure.database.service import DatabaseService
from src.client.agents.router import RouterAgent
from src.client.agents.task_agent import TaskManagerAgent
from src.client.agents.identity import IdentityAgent

logger = structlog.get_logger(__name__)


class AppState:
    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.database: DatabaseService | None = None
        self.gemini_client: GeminiAudioClient | None = None
        self.orchestrator: Orchestrator | None = None
        self.twilio_handler: TwilioVoiceHandler | None = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Load Settings
    try:
        settings = get_settings()
        app_state.settings = settings
    except Exception as e:
        logger.critical("configuration_error", error=str(e))
        sys.exit(1)

    logger.info(
        "starting_application",
        environment=settings.environment,
        host=settings.server_host,
        port=settings.server_port,
    )

    # Initialize Database
    try:
        logger.info("connecting_to_database", host=settings.db_host)
        app_state.database = DatabaseService(
            database_url=settings.get_database_url,
            echo=settings.debug,
        )
        await app_state.database.initialize()
    except Exception as e:
        logger.critical("database_connection_failed_fatal", error=str(e))
        print("\n" + "!"*60)
        print("FATAL ERROR: Could not connect to the database.")
        print(f"Target Host: {settings.db_host}")
        print("Reason: " + str(e))
        sys.exit(1)

    # Initialize Gemini
    try:
        app_state.gemini_client = GeminiAudioClient(
            api_key=settings.google_api_key,
            config=LLMConfig(
                model_name=settings.gemini_model,
                voice_name=settings.gemini_voice,
                response_modality="AUDIO",
            ),
        )
    except Exception as e:
        logger.critical("gemini_init_failed", error=str(e))
        sys.exit(1)

    # Initialize Orchestrator
    app_state.orchestrator = Orchestrator(
        context=GlobalContext(environment=settings.environment),
    )

    # Register Agents
    router_agent = RouterAgent(gemini_client=app_state.gemini_client)
    identity_agent = IdentityAgent(
        gemini_client=app_state.gemini_client,
        database_service=app_state.database,
    )
    task_agent = TaskManagerAgent(
        gemini_client=app_state.gemini_client,
        database_service=app_state.database,
    )

    app_state.orchestrator.register_agent(router_agent)
    app_state.orchestrator.register_agent(identity_agent)
    app_state.orchestrator.register_agent(task_agent)

    # Set Default Agent (Logic in Twilio Handler will override this based on caller ID)
    await app_state.orchestrator.set_active_agent("router")

    # Initialize Twilio
    app_state.twilio_handler = TwilioVoiceHandler(
        orchestrator=app_state.orchestrator,
        gemini_client=app_state.gemini_client,
        database_service=app_state.database,
    )

    logger.info("application_started")

    yield

    logger.info("shutting_down_application")
    if app_state.gemini_client:
        await app_state.gemini_client.close()
    if app_state.database:
        await app_state.database.close()
    logger.info("application_stopped")


def create_app() -> FastAPI:
    try:
        settings = get_settings()
        debug = settings.debug
    except:
        debug = False

    app = FastAPI(
        title="Kura-Next Voice Agent",
        description="Enterprise Voice Agent Framework with Gemini 2.5 Flash",
        version="1.0.0",
        lifespan=lifespan,
        debug=debug,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        return "<html><body><h1>Kura-Next</h1><p>Active</p></body></html>"

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        db_healthy = False
        if app_state.database:
            db_healthy = await app_state.database.health_check()
        return {"status": "healthy" if db_healthy else "degraded"}

    @app.post("/twilio/voice")
    async def twilio_voice_webhook(request: Request) -> Response:
        if not app_state.settings:
            raise HTTPException(status_code=500, detail="Server not initialized")
        
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "unknown")
        from_number = form_data.get("From", "unknown")
        
        host = request.headers.get("x-forwarded-host", request.headers.get("host", f"{app_state.settings.server_host}:{app_state.settings.server_port}"))
        forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        protocol = "wss" if forwarded_proto == "https" else "ws"
        
        # [FIX] URL Encode the phone number to preserve '+' characters
        encoded_from = urllib.parse.quote(from_number)
        ws_url = f"{protocol}://{host}/ws/twilio/{call_sid}?from_number={encoded_from}"

        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Connecting to Kura...</Say>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="caller" value="{from_number}"/>
        </Stream>
    </Connect>
</Response>"""
        return Response(content=twiml, media_type="application/xml")

    @app.websocket("/ws/twilio/{call_sid}")
    async def twilio_websocket(websocket: WebSocket, call_sid: str, from_number: str = "") -> None:
        if not app_state.twilio_handler:
            await websocket.close(code=1011)
            return
        
        if not from_number:
            # Try query param first
            from_number = websocket.query_params.get("from_number", "")
            
        # If still missing, check custom parameters passed by Twilio <Stream>
        # Note: These usually come in the first WebSocket message ('start' event), 
        # but for connection setup we might fallback to 'unknown' until the stream starts.
        if not from_number:
            from_number = "unknown"
            
        try:
            await app_state.twilio_handler.handle_call(websocket, call_sid, from_number)
        except Exception:
            try:
                await websocket.close(code=1011)
            except: pass

    @app.websocket("/ws/audio")
    async def audio_websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        if not app_state.orchestrator:
            await websocket.close(code=1011)
            return
        orchestrator = Orchestrator(
            context=GlobalContext(environment=app_state.settings.environment if app_state.settings else "development")
        )
        if app_state.orchestrator:
            for name, agent in app_state.orchestrator._agents.items():
                orchestrator.register_agent(agent)
            await orchestrator.set_active_agent("router")
        io_handler = WebSocketHandler(str(id(websocket)), websocket)
        try:
            await orchestrator.run(io_handler)
        finally:
            await io_handler.close()

app = create_app()
