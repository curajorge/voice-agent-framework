"""Database service for managing connections and sessions.

Provides a high-level interface for database operations with
connection pooling and session management.
"""

import ssl
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import structlog

from src.infrastructure.database.models import Base
from src.infrastructure.database.repository import TaskRepository, UserRepository

logger = structlog.get_logger(__name__)


class DatabaseService:
    """Database service managing connections and providing repositories."""

    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        """Initialize the database service."""
        self.database_url = database_url
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._echo = echo
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._logger = structlog.get_logger(__name__)

    async def initialize(self) -> None:
        """Initialize the database connection and create tables."""
        sanitized_url = self._sanitize_url()
        self._logger.info("initializing_database", url=sanitized_url)

        # Prepare connection arguments
        connect_args = {}
        if "postgresql" in self.database_url:
            # Create a relaxed SSL context to bypass verification errors
            # This fixes [SSL: CERTIFICATE_VERIFY_FAILED]
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            
            connect_args["ssl"] = ssl_ctx
            
            # Disable prepared statements for Supabase Pooler compatibility
            connect_args["statement_cache_size"] = 0

        self._engine = create_async_engine(
            self.database_url,
            echo=self._echo,
            pool_size=self._pool_size,
            max_overflow=self._max_overflow,
            connect_args=connect_args,
        )

        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Create tables
        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._logger.info("database_tables_created")
        except Exception as e:
            self._logger.error(
                "database_connection_failed",
                url=sanitized_url,
                error=str(e)
            )
            # Re-raise to ensure we don't start the app with a broken DB
            raise

    def _sanitize_url(self) -> str:
        """Sanitize the database URL for logging."""
        url = str(self.database_url)
        if "@" in url and ":" in url.split("@")[0]:
            parts = url.split("@")
            creds = parts[0].split(":")
            if len(creds) > 2:
                creds[-1] = "****"
            return ":".join(creds) + "@" + "@".join(parts[1:])
        return url

    async def close(self) -> None:
        """Close the database connection."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._logger.info("database_closed")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session context manager."""
        if not self._session_factory:
            raise RuntimeError("Database service not initialized")

        async with self._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    @asynccontextmanager
    async def repositories(
        self,
    ) -> AsyncGenerator[tuple[UserRepository, TaskRepository], None]:
        """Get repository instances within a session."""
        async with self.session() as session:
            yield UserRepository(session), TaskRepository(session)

    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        if not self._engine:
            return False

        try:
            async with self._engine.connect() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            self._logger.error("database_health_check_failed", error=str(e))
            return False
