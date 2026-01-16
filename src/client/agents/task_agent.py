"""Task Manager Agent - The Scribe.

The Task Manager Agent handles all task-related operations including
creation, retrieval, updates, and deletion of tasks.

Tools that hit the database are marked as "slow" to trigger
latency masking (filler audio) in the Orchestrator.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.client.agents.base import BaseClientAgent
from src.framework.core.agent import ModelConfig, Tool
from src.framework.core.context import GlobalContext, HandoffData
from src.framework.core.signals import Response, Signal, TextSignal
from src.infrastructure.llm.gemini_audio import GeminiAudioClient
from src.infrastructure.database.service import DatabaseService

logger = structlog.get_logger(__name__)

# Default prompt path
DEFAULT_PROMPT_PATH = Path("resources/prompts/task_manager/v1_master.txt")


class TaskManagerAgent(BaseClientAgent):
    """The Scribe - Dedicated Task Management Specialist.

    The Task Manager Agent:
    - Creates new tasks with description, priority, and due dates
    - Retrieves and summarizes tasks
    - Updates task status and details
    - Provides intelligent task briefings
    
    Database-hitting tools are marked as slow to trigger filler audio.
    """

    def __init__(
        self,
        gemini_client: GeminiAudioClient,
        database_service: DatabaseService,
        system_prompt_path: Path | str | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        """Initialize the Task Manager Agent.

        Args:
            gemini_client: The Gemini audio client.
            database_service: The database service.
            system_prompt_path: Optional custom prompt path.
            model_config: Optional model configuration.
        """
        self.database = database_service

        # Create tools for task management
        # Database tools are marked as is_slow=True for latency masking
        tools = [
            Tool(
                name="create_task",
                description="Create a new task for the user",
                function=self._create_task,
                parameters={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "The task description",
                        },
                        "priority": {
                            "type": "integer",
                            "description": "Priority level from 1 (highest) to 5 (lowest)",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date in ISO8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
                        },
                    },
                    "required": ["description"],
                },
                is_slow=True,  # Triggers filler audio
            ),
            Tool(
                name="search_tasks",
                description="Search and retrieve tasks based on query and filters",
                function=self._search_tasks,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to match against task descriptions",
                        },
                        "status": {
                            "type": "string",
                            "description": "Filter by status: OPEN, IN_PROGRESS, COMPLETED, CANCELLED",
                            "enum": ["OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
                        },
                    },
                },
                is_slow=True,
            ),
            Tool(
                name="get_all_tasks",
                description="Get all tasks for the current user",
                function=self._get_all_tasks,
                parameters={
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "Optional status filter",
                            "enum": ["OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tasks to return",
                            "default": 10,
                        },
                    },
                },
                is_slow=True,
            ),
            Tool(
                name="update_task_status",
                description="Update the status of a specific task",
                function=self._update_task_status,
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The unique task identifier",
                        },
                        "status": {
                            "type": "string",
                            "description": "New status for the task",
                            "enum": ["OPEN", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
                        },
                    },
                    "required": ["task_id", "status"],
                },
                is_slow=True,
            ),
            Tool(
                name="get_todays_tasks",
                description="Get tasks that are due today",
                function=self._get_todays_tasks,
                parameters={
                    "type": "object",
                    "properties": {},
                },
                is_slow=True,
            ),
            Tool(
                name="get_high_priority_tasks",
                description="Get high priority tasks (priority 1-2)",
                function=self._get_high_priority_tasks,
                parameters={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of tasks",
                            "default": 5,
                        },
                    },
                },
                is_slow=True,
            ),
            Tool(
                name="delete_task",
                description="Delete a task permanently",
                function=self._delete_task,
                parameters={
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "The unique task identifier",
                        },
                    },
                    "required": ["task_id"],
                },
                is_slow=True,
            ),
        ]

        super().__init__(
            name="task_manager",
            system_prompt_path=system_prompt_path or DEFAULT_PROMPT_PATH,
            gemini_client=gemini_client,
            model_config=model_config or ModelConfig(
                temperature=0.7,
                response_modality="AUDIO",
                voice_name="Kore",
            ),
            tools=tools,
        )

        # Track context for multi-turn slot filling
        self._pending_task: dict[str, Any] = {}

    def _get_user_id(self, context: GlobalContext) -> str:
        """Get the current user ID from context.

        Args:
            context: The global context.

        Returns:
            The user ID.

        Raises:
            ValueError: If user is not authenticated.
        """
        if not context.is_authenticated():
            raise ValueError("User not authenticated")
        return context.user.user_id

    async def _create_task(
        self,
        description: str,
        priority: int = 3,
        due_date: str | None = None,
        _context: GlobalContext | None = None,
    ) -> dict[str, Any]:
        """Create a new task.

        Args:
            description: Task description.
            priority: Priority level (1-5).
            due_date: Optional due date string.
            _context: Injected context (handled by tool execution).

        Returns:
            Task creation result.
        """
        if not hasattr(self, "_current_context") or not self._current_context:
            return {"success": False, "error": "No context available"}

        user_id = self._get_user_id(self._current_context)

        # Parse due date if provided
        parsed_due_date: datetime | None = None
        if due_date:
            try:
                if "T" in due_date:
                    parsed_due_date = datetime.fromisoformat(due_date)
                else:
                    parsed_due_date = datetime.fromisoformat(f"{due_date}T23:59:59")
            except ValueError:
                self._logger.warning("invalid_due_date", due_date=due_date)

        async with self.database.repositories() as (_, tasks):
            task = await tasks.create(
                user_id=user_id,
                description=description,
                priority=priority,
                due_date=parsed_due_date,
            )

            self._logger.info(
                "task_created",
                task_id=task.task_id,
                user_id=user_id,
            )

            return {
                "success": True,
                "task_id": task.task_id,
                "description": task.description,
                "priority": task.priority,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "message": f"Task created: {description}",
            }

    async def _search_tasks(
        self,
        query: str = "",
        status: str | None = None,
    ) -> dict[str, Any]:
        """Search for tasks.

        Args:
            query: Search query.
            status: Optional status filter.

        Returns:
            Search results.
        """
        if not hasattr(self, "_current_context") or not self._current_context:
            return {"success": False, "error": "No context available"}

        user_id = self._get_user_id(self._current_context)

        async with self.database.repositories() as (_, tasks):
            if query:
                results = await tasks.search(user_id, query, status)
            else:
                results = await tasks.get_by_user(user_id, status=status)

            return {
                "success": True,
                "count": len(results),
                "tasks": [task.to_dict() for task in results],
            }

    async def _get_all_tasks(
        self,
        status: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Get all tasks for the user.

        Args:
            status: Optional status filter.
            limit: Maximum number of tasks.

        Returns:
            Task list.
        """
        if not hasattr(self, "_current_context") or not self._current_context:
            return {"success": False, "error": "No context available"}

        user_id = self._get_user_id(self._current_context)

        async with self.database.repositories() as (_, tasks):
            results = await tasks.get_by_user(user_id, status=status, limit=limit)

            return {
                "success": True,
                "count": len(results),
                "tasks": [task.to_dict() for task in results],
            }

    async def _update_task_status(
        self,
        task_id: str,
        status: str,
    ) -> dict[str, Any]:
        """Update a task's status.

        Args:
            task_id: The task ID.
            status: The new status.

        Returns:
            Update result.
        """
        async with self.database.repositories() as (_, tasks):
            task = await tasks.update_status(task_id, status)

            if task:
                return {
                    "success": True,
                    "task_id": task_id,
                    "new_status": status,
                    "message": f"Task updated to {status}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found",
                }

    async def _get_todays_tasks(self) -> dict[str, Any]:
        """Get tasks due today.

        Returns:
            Today's tasks.
        """
        if not hasattr(self, "_current_context") or not self._current_context:
            return {"success": False, "error": "No context available"}

        user_id = self._get_user_id(self._current_context)

        async with self.database.repositories() as (_, tasks):
            results = await tasks.get_due_today(user_id)

            return {
                "success": True,
                "count": len(results),
                "tasks": [task.to_dict() for task in results],
                "message": f"You have {len(results)} task(s) due today",
            }

    async def _get_high_priority_tasks(
        self,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Get high priority tasks.

        Args:
            limit: Maximum number of tasks.

        Returns:
            High priority tasks.
        """
        if not hasattr(self, "_current_context") or not self._current_context:
            return {"success": False, "error": "No context available"}

        user_id = self._get_user_id(self._current_context)

        async with self.database.repositories() as (_, tasks):
            results = await tasks.get_high_priority(user_id, limit)

            return {
                "success": True,
                "count": len(results),
                "tasks": [task.to_dict() for task in results],
                "message": f"You have {len(results)} high priority task(s)",
            }

    async def _delete_task(self, task_id: str) -> dict[str, Any]:
        """Delete a task.

        Args:
            task_id: The task ID.

        Returns:
            Deletion result.
        """
        async with self.database.repositories() as (_, tasks):
            deleted = await tasks.delete(task_id)

            if deleted:
                return {
                    "success": True,
                    "task_id": task_id,
                    "message": "Task deleted successfully",
                }
            else:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found",
                }

    async def process_signal(
        self,
        signal: Signal,
        context: GlobalContext,
    ) -> Response:
        """Process a task-related signal.

        Args:
            signal: The input signal.
            context: The global context.

        Returns:
            The agent's response.
        """
        # Store context for tool access
        self._current_context = context

        # Check for handover context from warm handoff
        handover = context.session.scratchpad.get("handover_context")
        if handover:
            self._logger.debug("received_handover", context=handover)
            context.session.scratchpad.set("handover_context", None)

        # Use base processing which handles LLM interaction
        response = await super().process_signal(signal, context)

        return response

    async def handle_tool_result(
        self,
        tool_name: str,
        result: Any,
        context: GlobalContext,
    ) -> Response | None:
        """Handle task tool results.

        Args:
            tool_name: The executed tool name.
            result: The tool result.
            context: The global context.

        Returns:
            Optional response to send to user.
        """
        self._logger.info(
            "tool_result",
            tool=tool_name,
            success=result.get("success", False),
        )

        # Generate a natural response based on tool result
        if result.get("success"):
            if tool_name == "create_task":
                return Response.text_response(
                    session_id=context.session.session_id,
                    agent_name=self.name,
                    content=f"I've created your task: {result.get('description')}. "
                    f"Priority is set to {result.get('priority')}.",
                )
            elif tool_name in ("get_all_tasks", "search_tasks"):
                tasks = result.get("tasks", [])
                if tasks:
                    summary = self._summarize_tasks(tasks)
                    return Response.text_response(
                        session_id=context.session.session_id,
                        agent_name=self.name,
                        content=summary,
                    )
                else:
                    return Response.text_response(
                        session_id=context.session.session_id,
                        agent_name=self.name,
                        content="You don't have any tasks matching that criteria.",
                    )

        return None

    def _summarize_tasks(self, tasks: list[dict[str, Any]]) -> str:
        """Create a natural summary of tasks.

        Args:
            tasks: List of task dictionaries.

        Returns:
            Natural language summary.
        """
        if not tasks:
            return "You have no tasks."

        count = len(tasks)

        # Group by priority
        high_priority = [t for t in tasks if t.get("priority", 3) <= 2]
        normal = [t for t in tasks if 2 < t.get("priority", 3) <= 4]
        low = [t for t in tasks if t.get("priority", 3) > 4]

        parts = [f"You have {count} task{'s' if count != 1 else ''}."]

        if high_priority:
            parts.append(f"\n{len(high_priority)} high priority:")
            for t in high_priority[:3]:
                parts.append(f"  - {t['description']}")

        if normal and len(parts) < 6:
            parts.append(f"\n{len(normal)} normal priority:")
            for t in normal[:2]:
                parts.append(f"  - {t['description']}")

        return "\n".join(parts)

    async def on_enter(
        self,
        context: GlobalContext,
        handoff_data: HandoffData | None = None,
    ) -> None:
        """Called when task manager becomes active.

        Args:
            context: The global context.
            handoff_data: Optional handoff data from previous agent.
        """
        await super().on_enter(context, handoff_data)
        self._current_context = context
        
        # Log warm handoff info
        if handoff_data:
            self._logger.info(
                "task_manager_activated_with_handoff",
                user=context.user.full_name,
                source=handoff_data.source_agent,
                intent=handoff_data.user_intent,
                greeting_done=handoff_data.greeting_completed,
            )
        else:
            self._logger.info(
                "task_manager_active",
                user=context.user.full_name,
            )
