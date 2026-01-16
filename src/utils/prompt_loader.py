"""Prompt loading and management utilities.

Handles loading, versioning, and rendering of system prompts
from the resources directory.
"""

from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class PromptLoader:
    """Utility class for loading and managing prompts.

    Supports versioned prompts and template variable substitution.
    """

    DEFAULT_RESOURCES_PATH = Path("resources/prompts")

    def __init__(self, base_path: Path | str | None = None) -> None:
        """Initialize the prompt loader.

        Args:
            base_path: Base path for prompt files.
        """
        self.base_path = Path(base_path) if base_path else self.DEFAULT_RESOURCES_PATH
        self._cache: dict[str, str] = {}
        self._logger = structlog.get_logger(__name__)

    def load(
        self,
        agent: str,
        version: str = "v1",
        variant: str = "system",
    ) -> str:
        """Load a prompt from file.

        Args:
            agent: Agent name (e.g., 'router', 'task_manager').
            version: Prompt version (e.g., 'v1', 'v2').
            variant: Prompt variant (e.g., 'system', 'master').

        Returns:
            The prompt content.

        Raises:
            FileNotFoundError: If prompt file doesn't exist.
        """
        cache_key = f"{agent}/{version}_{variant}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try different file patterns
        patterns = [
            self.base_path / agent / f"{version}_{variant}.txt",
            self.base_path / agent / f"{version}_{variant}.md",
            self.base_path / agent / f"{version}.txt",
        ]

        for path in patterns:
            if path.exists():
                content = path.read_text(encoding="utf-8")
                self._cache[cache_key] = content
                self._logger.debug("prompt_loaded", path=str(path))
                return content

        raise FileNotFoundError(
            f"Prompt not found for {agent}/{version}_{variant}"
        )

    def load_with_fallback(
        self,
        agent: str,
        version: str = "v1",
        variant: str = "system",
        default: str = "",
    ) -> str:
        """Load a prompt with fallback to default.

        Args:
            agent: Agent name.
            version: Prompt version.
            variant: Prompt variant.
            default: Default content if file not found.

        Returns:
            The prompt content or default.
        """
        try:
            return self.load(agent, version, variant)
        except FileNotFoundError:
            self._logger.warning(
                "prompt_not_found_using_default",
                agent=agent,
                version=version,
            )
            return default

    def render(
        self,
        content: str,
        variables: dict[str, Any],
    ) -> str:
        """Render a prompt with template variables.

        Supports {{variable}} syntax for substitution.

        Args:
            content: The prompt template.
            variables: Variables to substitute.

        Returns:
            The rendered prompt.
        """
        result = content
        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            result = result.replace(placeholder, str(value))
        return result

    def list_agents(self) -> list[str]:
        """List all available agent prompts.

        Returns:
            List of agent names.
        """
        if not self.base_path.exists():
            return []

        return [
            d.name for d in self.base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    def list_versions(self, agent: str) -> list[str]:
        """List available versions for an agent.

        Args:
            agent: Agent name.

        Returns:
            List of version strings.
        """
        agent_path = self.base_path / agent
        if not agent_path.exists():
            return []

        versions = set()
        for f in agent_path.iterdir():
            if f.is_file() and f.suffix in (".txt", ".md"):
                # Extract version from filename (e.g., "v1_system.txt" -> "v1")
                name = f.stem
                if "_" in name:
                    version = name.split("_")[0]
                    versions.add(version)

        return sorted(versions)

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()
        self._logger.debug("prompt_cache_cleared")
