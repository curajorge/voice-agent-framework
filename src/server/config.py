"""Application configuration management.

Handles environment variables and configuration settings using Pydantic.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Kura-Next"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8080
    websocket_path: str = "/ws/audio"

    # Google Gemini
    google_api_key: str = Field(..., description="Google API key for Gemini")
    # [FIX] Switched to stable experimental model for better Realtime API support
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_voice: str = "Kore"

    # Twilio
    twilio_account_sid: str = Field(default="", description="Twilio Account SID")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token")
    twilio_phone_number: str = Field(default="", description="Twilio Phone Number")

    # Database (Supabase/PostgreSQL)
    db_host: str = Field(default="", description="Database host (e.g. db.xyz.supabase.co)")
    db_user: str = Field(default="", description="Database user (usually 'postgres')")
    db_password: str = Field(default="", description="Database password")
    db_port: int = Field(default=5432, description="Database port (5432 or 6543)")
    db_name: str = Field(default="postgres", description="Database name")

    # Optional full URL override
    database_url: str = Field(default="", description="Full database URL override")

    @field_validator("db_host")
    @classmethod
    def validate_host_placeholder(cls, v: str) -> str:
        if "your-project-ref" in v:
            raise ValueError(
                "CRITICAL CONFIG ERROR: Your DB_HOST in .env is still set to the default placeholder! "
                "Please replace 'your-project-ref.supabase.co' with your actual Supabase host."
            )
        return v

    @property
    def get_database_url(self) -> str:
        # 1. Check for full URL override first
        if self.database_url:
            url = self.database_url
            if url.startswith("postgres://"):
                return url.replace("postgres://", "postgresql+asyncpg://", 1)
            if url.startswith("postgresql://") and "asyncpg" not in url:
                return url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url

        # 2. Construct from components (Supabase style)
        if not self.db_host or not self.db_user or not self.db_password:
            raise ValueError(
                "Missing Database Configuration! \n"
                "Please configure DB_HOST, DB_USER, and DB_PASSWORD in your .env file \n"
                "to connect to Supabase/PostgreSQL."
            )

        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def websocket_url(self) -> str:
        protocol = "wss" if self.is_production else "ws"
        return f"{protocol}://{self.server_host}:{self.server_port}{self.websocket_path}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
