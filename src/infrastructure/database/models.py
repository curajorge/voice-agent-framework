"""SQLAlchemy models for the database.

Defines the User and Task models as per the architecture specification.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class User(Base):
    """User model representing authenticated users.

    Attributes:
        user_id: Primary key, unique identifier.
        phone_number: User's phone number (unique).
        full_name: User's full name.
        voice_preferences: JSON blob for voice settings.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
    """

    __tablename__ = "users"

    user_id: str = Column(String(36), primary_key=True)
    phone_number: str = Column(String(20), unique=True, nullable=False, index=True)
    full_name: str | None = Column(String(255), nullable=True)
    voice_preferences: dict[str, Any] = Column(JSON, default=dict)
    created_at: datetime = Column(DateTime, default=func.now(), nullable=False)
    updated_at: datetime = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationship to tasks
    tasks = relationship("Task", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        """String representation."""
        return f"<User(id={self.user_id}, phone={self.phone_number})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the user.
        """
        return {
            "user_id": self.user_id,
            "phone_number": self.phone_number,
            "full_name": self.full_name,
            "voice_preferences": self.voice_preferences,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Task(Base):
    """Task model representing user tasks.

    Attributes:
        task_id: Primary key, unique identifier.
        user_id: Foreign key to users table.
        description: Task description.
        priority: Priority level (1-5, 1 being highest).
        status: Task status (OPEN, IN_PROGRESS, COMPLETED, CANCELLED).
        due_date: Optional due date.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
    """

    __tablename__ = "tasks"

    task_id: str = Column(String(36), primary_key=True)
    user_id: str = Column(
        String(36), ForeignKey("users.user_id"), nullable=False, index=True
    )
    description: str = Column(Text, nullable=False)
    priority: int = Column(Integer, default=3, nullable=False)
    status: str = Column(String(20), default="OPEN", nullable=False, index=True)
    due_date: datetime | None = Column(DateTime, nullable=True)
    created_at: datetime = Column(DateTime, default=func.now(), nullable=False)
    updated_at: datetime = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationship to user
    user = relationship("User", back_populates="tasks")

    def __repr__(self) -> str:
        """String representation."""
        return f"<Task(id={self.task_id}, status={self.status})>"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the task.
        """
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
