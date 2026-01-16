"""Client implementation - The TaskMaster showcase application."""

from src.client.agents.router import RouterAgent
from src.client.agents.task_agent import TaskManagerAgent
from src.client.agents.identity import IdentityAgent

__all__ = [
    "RouterAgent",
    "TaskManagerAgent",
    "IdentityAgent",
]
