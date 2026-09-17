"""MOSAIC Configuration Management Module."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment parameters."""
    
    PROJECT_NAME: str = "MOSAIC AI Orchestrator"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment & Debugging
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # Database
    DATABASE_URL: str = "sqlite:///./mosaic_dev.db"
    
    # LLM Settings (Default to mock for 100% offline execution)
    LLM_PROVIDER: str = "mock"
    LLM_MODEL: str = "mock-deterministic-v1"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_TIMEOUT: float = 60.0
    LLM_ENABLED: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
