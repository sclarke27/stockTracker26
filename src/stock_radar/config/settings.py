"""Pydantic settings models for application configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ApiKeys(BaseModel):
    """External API key configuration."""

    alpha_vantage: str = Field(description="Alpha Vantage API key")
    finnhub: str = Field(description="Finnhub API key")
    anthropic: str = Field(default="", description="Anthropic Claude API key")


class OllamaSettings(BaseModel):
    """Ollama LLM runtime configuration."""

    host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    default_model: str = Field(
        default="llama3.1:8b", description="Default model for routine analysis"
    )
    large_model: str = Field(default="llama3.1:70b", description="Large model for complex analysis")
    timeout_seconds: int = Field(default=30, description="Max inference time before timeout")


class CacheSettings(BaseModel):
    """Local cache and database configuration."""

    db_path: str = Field(default="data/stock_radar.db", description="SQLite database path")
    chroma_path: str = Field(default="data/chroma_data", description="ChromaDB storage directory")


class SecEdgarSettings(BaseModel):
    """SEC EDGAR API configuration."""

    user_agent_email: str = Field(description="Contact email for SEC User-Agent header")


class PredictionsSettings(BaseModel):
    """Predictions database configuration."""

    db_path: str = Field(
        default="data/predictions.db",
        description="SQLite database path for prediction storage",
    )


class AppSettings(BaseModel):
    """Top-level application settings."""

    api_keys: ApiKeys = Field(description="External API keys")
    ollama: OllamaSettings = Field(
        default_factory=OllamaSettings, description="Ollama configuration"
    )
    cache: CacheSettings = Field(
        default_factory=CacheSettings, description="Cache and storage settings"
    )
    sec_edgar: SecEdgarSettings = Field(description="SEC EDGAR configuration")
    predictions: PredictionsSettings = Field(
        default_factory=PredictionsSettings,
        description="Predictions database settings",
    )
