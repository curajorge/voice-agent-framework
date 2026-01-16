"""Tests for database operations."""

import pytest

from src.infrastructure.database.models import User, Task


@pytest.mark.asyncio
class TestUserRepository:
    """Tests for UserRepository."""

    async def test_create_user(self, database):
        """Test creating a user."""
        async with database.repositories() as (users, _):
            user = await users.create(
                phone_number="+1234567890",
                full_name="Test User",
            )

            assert user.user_id is not None
            assert user.phone_number == "+1234567890"
            assert user.full_name == "Test User"

    async def test_get_by_phone(self, database):
        """Test finding user by phone number."""
        async with database.repositories() as (users, _):
            await users.create(
                phone_number="+1111111111",
                full_name="Phone User",
            )

            found = await users.get_by_phone("+1111111111")
            assert found is not None
            assert found.full_name == "Phone User"

            not_found = await users.get_by_phone("+9999999999")
            assert not_found is None

    async def test_get_or_create(self, database):
        """Test get_or_create functionality."""
        async with database.repositories() as (users, _):
            # First call should create
            user1, created1 = await users.get_or_create(
                phone_number="+2222222222",
                full_name="New User",
            )
            assert created1 is True
            assert user1.full_name == "New User"

            # Second call should get existing
            user2, created2 = await users.get_or_create(
                phone_number="+2222222222",
            )
            assert created2 is False
            assert user2.user_id == user1.user_id


@pytest.mark.asyncio
class TestTaskRepository:
    """Tests for TaskRepository."""

    async def test_create_task(self, database):
        """Test creating a task."""
        async with database.repositories() as (users, tasks):
            user = await users.create(
                phone_number="+3333333333",
                full_name="Task User",
            )

            task = await tasks.create(
                user_id=user.user_id,
                description="Test task",
                priority=2,
            )

            assert task.task_id is not None
            assert task.description == "Test task"
            assert task.priority == 2
            assert task.status == "OPEN"

    async def test_get_by_user(self, database):
        """Test getting tasks by user."""
        async with database.repositories() as (users, tasks):
            user = await users.create(
                phone_number="+4444444444",
                full_name="Multi Task User",
            )

            await tasks.create(user_id=user.user_id, description="Task 1")
            await tasks.create(user_id=user.user_id, description="Task 2")
            await tasks.create(user_id=user.user_id, description="Task 3")

            user_tasks = await tasks.get_by_user(user.user_id)
            assert len(user_tasks) == 3

    async def test_update_status(self, database):
        """Test updating task status."""
        async with database.repositories() as (users, tasks):
            user = await users.create(
                phone_number="+5555555555",
                full_name="Status User",
            )

            task = await tasks.create(
                user_id=user.user_id,
                description="Status task",
            )

            updated = await tasks.update_status(task.task_id, "COMPLETED")
            assert updated is not None
            assert updated.status == "COMPLETED"

    async def test_search_tasks(self, database):
        """Test searching tasks."""
        async with database.repositories() as (users, tasks):
            user = await users.create(
                phone_number="+6666666666",
                full_name="Search User",
            )

            await tasks.create(
                user_id=user.user_id,
                description="Buy groceries",
            )
            await tasks.create(
                user_id=user.user_id,
                description="Call dentist",
            )
            await tasks.create(
                user_id=user.user_id,
                description="Buy new shoes",
            )

            results = await tasks.search(user.user_id, "Buy")
            assert len(results) == 2

    async def test_delete_task(self, database):
        """Test deleting a task."""
        async with database.repositories() as (users, tasks):
            user = await users.create(
                phone_number="+7777777777",
                full_name="Delete User",
            )

            task = await tasks.create(
                user_id=user.user_id,
                description="To be deleted",
            )

            deleted = await tasks.delete(task.task_id)
            assert deleted is True

            found = await tasks.get_by_id(task.task_id)
            assert found is None
