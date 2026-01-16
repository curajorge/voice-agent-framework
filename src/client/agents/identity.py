"""Identity Agent - The Gatekeeper."""

from pathlib import Path
from typing import Any
import structlog
from src.client.agents.base import BaseClientAgent
from src.framework.core.agent import ModelConfig, Tool
from src.framework.core.context import GlobalContext, UserContext
from src.infrastructure.llm.gemini_audio import GeminiAudioClient
from src.infrastructure.database.service import DatabaseService

logger = structlog.get_logger(__name__)


class IdentityAgent(BaseClientAgent):
    """The Gatekeeper - Handles user authentication."""

    def __init__(
        self,
        gemini_client: GeminiAudioClient,
        database_service: DatabaseService,
        system_prompt_path: Path | str | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.database = database_service

        # Create tools
        tools = [
            Tool(
                name="create_user",
                description="Create account. Usage: create_user(phone_number='...', full_name='...')",
                function=self._create_user,
                parameters={
                    "type": "object",
                    "properties": {
                        "phone_number": {"type": "string"},
                        "full_name": {"type": "string"},
                    },
                    "required": ["phone_number", "full_name"],
                },
            ),
        ]

        super().__init__(
            name="identity",
            # Load from the file now that we have updated it with the optimized prompt
            system_prompt_path=system_prompt_path or Path("resources/prompts/identity/v1_system.txt"),
            gemini_client=gemini_client,
            model_config=model_config or ModelConfig(temperature=0.5, response_modality="AUDIO"),
            tools=tools,
        )

    def render_prompt(self, context: GlobalContext) -> str:
        """Render prompt with phone number."""
        # We use self.system_prompt (which loads from file) instead of the hardcoded constant
        phone = context.session.metadata.get("phone_number", "unknown")
        return self.system_prompt.replace("{{phone_number}}", phone)

    async def _create_user(self, phone_number: str, full_name: str) -> dict[str, Any]:
        """Create user in DB."""
        # Robustness: Remove spaces/formatting from phone
        clean_phone = phone_number.replace(" ", "").replace("-", "")
        
        async with self.database.repositories() as (users, _):
            try:
                user = await users.create(phone_number=clean_phone, full_name=full_name)
                return {
                    "success": True,
                    "user_id": user.user_id,
                    "full_name": user.full_name,
                    "message": "Account created."
                }
            except Exception as e:
                return {"success": False, "error": str(e)}

    async def process_signal(self, signal, context):
        # BaseClientAgent handles the processing
        return await super().process_signal(signal, context)
