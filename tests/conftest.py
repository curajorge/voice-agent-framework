"""Pytest configuration and fixtures."""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio

# Set test environment
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("GOOGLE_API_KEY", "test-api-key")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def database() -> AsyncGenerator:
    """Create a test database."""
    from src.infrastructure.database.service import DatabaseService

    db = DatabaseService(database_url="sqlite+aiosqlite:///:memory:")
    await db.initialize()
    yield db
    await db.close()


@pytest_asyncio.fixture
async def user_repository(database):
    """Create a user repository."""
    async with database.session() as session:
        from src.infrastructure.database.repository import UserRepository
        yield UserRepository(session)


@pytest_asyncio.fixture
async def task_repository(database):
    """Create a task repository."""
    async with database.session() as session:
        from src.infrastructure.database.repository import TaskRepository
        yield TaskRepository(session)


@pytest.fixture
def global_context():
    """Create a test global context."""
    from src.framework.core.context import GlobalContext, Platform, SessionContext

    context = GlobalContext(environment="test")
    context.session = SessionContext(platform=Platform.TEST)
    return context
