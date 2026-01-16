"""Main entry point for the Kura-Next Voice Agent.

This module provides both CLI and server modes for running the voice agent.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import structlog
import uvicorn

from src.server.config import get_settings


def configure_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """Configure structured logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_format: Output format ('json' or 'console').
    """
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    import logging

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper()),
    )


async def run_cli_mode() -> None:
    """Run the agent in CLI mode for testing."""
    from src.framework.core.orchestrator import Orchestrator
    from src.framework.core.context import GlobalContext, Platform, SessionContext
    from src.framework.core.io_handler import CLIHandler
    from src.infrastructure.llm.gemini_audio import GeminiAudioClient
    from src.infrastructure.llm.provider import LLMConfig
    from src.infrastructure.database.service import DatabaseService
    from src.client.agents.router import RouterAgent
    from src.client.agents.task_agent import TaskManagerAgent
    from src.client.agents.identity import IdentityAgent

    settings = get_settings()
    logger = structlog.get_logger(__name__)

    print("\n" + "=" * 60)
    print("  Kura-Next Voice Agent - CLI Mode")
    print("=" * 60)
    print("\nInitializing...")

    # Initialize database
    database = DatabaseService(database_url=settings.database_url)
    await database.initialize()
    print("  [✓] Database initialized")

    # Initialize Gemini client
    gemini_client = GeminiAudioClient(
        api_key=settings.google_api_key,
        config=LLMConfig(
            model_name=settings.gemini_model,
            voice_name=settings.gemini_voice,
            response_modality="TEXT",  # Use text mode for CLI
        ),
    )
    print("  [✓] Gemini client initialized")

    # Create context
    context = GlobalContext(
        environment=settings.environment,
    )
    context.session = SessionContext(platform=Platform.CLI)

    # Initialize orchestrator
    orchestrator = Orchestrator(context=context)

    # Register agents
    router_agent = RouterAgent(gemini_client=gemini_client)
    identity_agent = IdentityAgent(
        gemini_client=gemini_client,
        database_service=database,
    )
    task_agent = TaskManagerAgent(
        gemini_client=gemini_client,
        database_service=database,
    )

    orchestrator.register_agent(router_agent)
    orchestrator.register_agent(identity_agent)
    orchestrator.register_agent(task_agent)
    print("  [✓] Agents registered")

    # Set default agent
    await orchestrator.set_active_agent("router")

    print("\n" + "-" * 60)
    print("Ready! Type your messages below. Type 'exit' to quit.")
    print("-" * 60 + "\n")

    # Create CLI handler
    io_handler = CLIHandler(session_id=context.session.session_id)

    try:
        await orchestrator.run(io_handler)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await gemini_client.close()
        await database.close()


def run_server(host: str, port: int, reload: bool = False) -> None:
    """Run the FastAPI server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        reload: Enable auto-reload for development.
    """
    uvicorn.run(
        "src.server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Kura-Next Voice Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in server mode (default)
  python -m src.main

  # Run in CLI mode for testing
  python -m src.main --cli

  # Run server on custom port
  python -m src.main --port 8000

  # Run with auto-reload for development
  python -m src.main --reload
        """,
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in CLI mode for testing",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from settings)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from settings)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default="console",
        choices=["json", "console"],
        help="Logging format",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging(args.log_level, args.log_format)

    # Get settings
    settings = get_settings()

    if args.cli:
        # Run CLI mode
        asyncio.run(run_cli_mode())
    else:
        # Run server mode
        host = args.host or settings.server_host
        port = args.port or settings.server_port

        print("\n" + "=" * 60)
        print("  Kura-Next Voice Agent - Server Mode")
        print("=" * 60)
        print(f"\n  Starting server on {host}:{port}")
        print(f"  Environment: {settings.environment}")
        print(f"  WebSocket: ws://{host}:{port}/ws/audio")
        print(f"  Twilio: POST http://{host}:{port}/twilio/voice")
        print("\n" + "=" * 60 + "\n")

        run_server(host, port, args.reload)


if __name__ == "__main__":
    main()
