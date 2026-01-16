"""Repository pattern implementations for database access.

Repositories provide a clean abstraction over database operations,
isolating the data access logic from the business logic.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from src.infrastructure.database.models import Task, User

logger = structlog.get_logger(__name__)


class UserRepository:
    """Repository for User operations.

    Provides CRUD operations for User entities with async support.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            session: SQLAlchemy async session.
        """
        self.session = session
        self._logger = structlog.get_logger(__name__).bind(repository="user")

    async def create(
        self,
        phone_number: str,
        full_name: str | None = None,
        voice_preferences: dict[str, Any] | None = None,
    ) -> User:
        """Create a new user.

        Args:
            phone_number: User's phone number.
            full_name: Optional full name.
            voice_preferences: Optional voice preferences.

        Returns:
            The created User.
        """
        user = User(
            user_id=str(uuid4()),
            phone_number=phone_number,
            full_name=full_name,
            voice_preferences=voice_preferences or {},
        )
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)

        self._logger.info("user_created", user_id=user.user_id)
        return user

    async def get_by_id(self, user_id: str) -> User | None:
        """Get a user by ID.

        Args:
            user_id: The user's ID.

        Returns:
            The User if found, None otherwise.
        """
        result = await self.session.execute(
            select(User).where(User.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_phone(self, phone_number: str) -> User | None:
        """Get a user by phone number.

        Args:
            phone_number: The user's phone number.

        Returns:
            The User if found, None otherwise.
        """
        result = await self.session.execute(
            select(User).where(User.phone_number == phone_number)
        )
        return result.scalar_one_or_none()

    async def update(
        self,
        user_id: str,
        **kwargs: Any,
    ) -> User | None:
        """Update a user.

        Args:
            user_id: The user's ID.
            **kwargs: Fields to update.

        Returns:
            The updated User if found, None otherwise.
        """
        await self.session.execute(
            update(User).where(User.user_id == user_id).values(**kwargs)
        )
        await self.session.commit()

        user = await self.get_by_id(user_id)
        if user:
            self._logger.info("user_updated", user_id=user_id)
        return user

    async def delete(self, user_id: str) -> bool:
        """Delete a user.

        Args:
            user_id: The user's ID.

        Returns:
            True if deleted, False if not found.
        """
        result = await self.session.execute(
            delete(User).where(User.user_id == user_id)
        )
        await self.session.commit()
        deleted = result.rowcount > 0

        if deleted:
            self._logger.info("user_deleted", user_id=user_id)
        return deleted

    async def get_or_create(
        self,
        phone_number: str,
        full_name: str | None = None,
    ) -> tuple[User, bool]:
        """Get an existing user or create a new one.

        Args:
            phone_number: The user's phone number.
            full_name: Optional full name for new users.

        Returns:
            Tuple of (User, created) where created is True if new.
        """
        existing = await self.get_by_phone(phone_number)
        if existing:
            return existing, False

        user = await self.create(phone_number=phone_number, full_name=full_name)
        return user, True


class TaskRepository:
    """Repository for Task operations.

    Provides CRUD operations for Task entities with async support.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository.

        Args:
            session: SQLAlchemy async session.
        """
        self.session = session
        self._logger = structlog.get_logger(__name__).bind(repository="task")

    async def create(
        self,
        user_id: str,
        description: str,
        priority: int = 3,
        due_date: datetime | None = None,
    ) -> Task:
        """Create a new task.

        Args:
            user_id: The owner's user ID.
            description: Task description.
            priority: Priority level (1-5).
            due_date: Optional due date.

        Returns:
            The created Task.
        """
        task = Task(
            task_id=str(uuid4()),
            user_id=user_id,
            description=description,
            priority=max(1, min(5, priority)),  # Clamp to 1-5
            due_date=due_date,
            status="OPEN",
        )
        self.session.add(task)
        await self.session.commit()
        await self.session.refresh(task)

        self._logger.info("task_created", task_id=task.task_id, user_id=user_id)
        return task

    async def get_by_id(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task's ID.

        Returns:
            The Task if found, None otherwise.
        """
        result = await self.session.execute(
            select(Task).where(Task.task_id == task_id)
        )
        return result.scalar_one_or_none()

    async def get_by_user(
        self,
        user_id: str,
        status: str | None = None,
        priority: int | None = None,
        limit: int = 50,
    ) -> list[Task]:
        """Get tasks for a user with optional filters.

        Args:
            user_id: The user's ID.
            status: Optional status filter.
            priority: Optional priority filter.
            limit: Maximum number of tasks to return.

        Returns:
            List of matching Tasks.
        """
        query = select(Task).where(Task.user_id == user_id)

        if status:
            query = query.where(Task.status == status)
        if priority:
            query = query.where(Task.priority == priority)

        query = query.order_by(Task.priority.asc(), Task.due_date.asc()).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def search(
        self,
        user_id: str,
        query: str,
        status: str | None = None,
    ) -> list[Task]:
        """Search tasks by description.

        Args:
            user_id: The user's ID.
            query: Search query string.
            status: Optional status filter.

        Returns:
            List of matching Tasks.
        """
        stmt = select(Task).where(
            Task.user_id == user_id,
            Task.description.ilike(f"%{query}%"),
        )

        if status:
            stmt = stmt.where(Task.status == status)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def update_status(
        self,
        task_id: str,
        status: str,
    ) -> Task | None:
        """Update a task's status.

        Args:
            task_id: The task's ID.
            status: The new status.

        Returns:
            The updated Task if found, None otherwise.
        """
        valid_statuses = {"OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"}
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}")

        await self.session.execute(
            update(Task).where(Task.task_id == task_id).values(status=status)
        )
        await self.session.commit()

        task = await self.get_by_id(task_id)
        if task:
            self._logger.info("task_status_updated", task_id=task_id, status=status)
        return task

    async def update(
        self,
        task_id: str,
        **kwargs: Any,
    ) -> Task | None:
        """Update a task.

        Args:
            task_id: The task's ID.
            **kwargs: Fields to update.

        Returns:
            The updated Task if found, None otherwise.
        """
        # Validate priority if provided
        if "priority" in kwargs:
            kwargs["priority"] = max(1, min(5, kwargs["priority"]))

        # Validate status if provided
        if "status" in kwargs:
            valid_statuses = {"OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"}
            if kwargs["status"] not in valid_statuses:
                raise ValueError(f"Invalid status: {kwargs['status']}")

        await self.session.execute(
            update(Task).where(Task.task_id == task_id).values(**kwargs)
        )
        await self.session.commit()

        task = await self.get_by_id(task_id)
        if task:
            self._logger.info("task_updated", task_id=task_id)
        return task

    async def delete(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: The task's ID.

        Returns:
            True if deleted, False if not found.
        """
        result = await self.session.execute(
            delete(Task).where(Task.task_id == task_id)
        )
        await self.session.commit()
        deleted = result.rowcount > 0

        if deleted:
            self._logger.info("task_deleted", task_id=task_id)
        return deleted

    async def get_due_today(self, user_id: str) -> list[Task]:
        """Get tasks due today for a user.

        Args:
            user_id: The user's ID.

        Returns:
            List of tasks due today.
        """
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow = today.replace(hour=23, minute=59, second=59)

        result = await self.session.execute(
            select(Task).where(
                Task.user_id == user_id,
                Task.due_date >= today,
                Task.due_date <= tomorrow,
                Task.status.in_(["OPEN", "IN_PROGRESS"]),
            ).order_by(Task.priority.asc())
        )
        return list(result.scalars().all())

    async def get_high_priority(self, user_id: str, limit: int = 5) -> list[Task]:
        """Get high priority tasks for a user.

        Args:
            user_id: The user's ID.
            limit: Maximum number of tasks.

        Returns:
            List of high priority tasks.
        """
        result = await self.session.execute(
            select(Task).where(
                Task.user_id == user_id,
                Task.priority <= 2,
                Task.status.in_(["OPEN", "IN_PROGRESS"]),
            ).order_by(Task.priority.asc(), Task.due_date.asc()).limit(limit)
        )
        return list(result.scalars().all())

    async def get_open_count(self, user_id: str) -> int:
        """Count open and in-progress tasks for a user.

        Args:
            user_id: The user's ID.

        Returns:
            Count of active tasks.
        """
        stmt = select(func.count()).select_from(Task).where(
            Task.user_id == user_id,
            Task.status.in_(["OPEN", "IN_PROGRESS"])
        )
        result = await self.session.execute(stmt)
        return result.scalar_one()
