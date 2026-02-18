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


class EarningsLinguistSettings(BaseModel):
    """Configuration for the Earnings Linguist agent."""

    enabled: bool = Field(default=True, description="Whether the agent is active")
    default_horizon_days: int = Field(
        default=5, gt=0, description="Default prediction horizon in days"
    )
    escalation_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence below this triggers escalation to Claude API",
    )
    escalation_transcript_length: int = Field(
        default=24000,
        gt=0,
        description="Transcript character length above which to escalate",
    )
    ollama_model: str | None = Field(
        default=None,
        description="Override Ollama model (uses ollama.default_model if None)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model for escalation",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM sampling temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens to generate")


class NarrativeDivergenceSettings(BaseModel):
    """Configuration for the Narrative vs Price Divergence agent."""

    enabled: bool = Field(default=True, description="Whether the agent is active")
    default_horizon_days: int = Field(
        default=10, gt=0, description="Default prediction horizon in days"
    )
    escalation_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence below this triggers escalation to Claude API",
    )
    escalation_min_articles: int = Field(
        default=5,
        gt=0,
        description="Escalate to Claude if fewer articles than this (low data quality)",
    )
    ollama_model: str | None = Field(
        default=None,
        description="Override Ollama model (uses ollama.default_model if None)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model for escalation",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")


class SecFilingAnalyzerSettings(BaseModel):
    """Configuration for the SEC Filing Pattern Analyzer agent."""

    enabled: bool = Field(default=True, description="Whether the agent is active")
    default_horizon_days: int = Field(
        default=15, gt=0, description="Default prediction horizon in days"
    )
    escalation_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence below this triggers escalation to Claude API",
    )
    escalation_filing_count: int = Field(
        default=30,
        gt=0,
        description="Escalate to Claude if filing count exceeds this (complex synthesis)",
    )
    ollama_model: str | None = Field(
        default=None,
        description="Override Ollama model (uses ollama.default_model if None)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model for escalation",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")


class ContagionMapperSettings(BaseModel):
    """Configuration for the Cross-Sector Contagion Mapper agent."""

    enabled: bool = Field(default=True, description="Whether the agent is active")
    default_horizon_days: int = Field(
        default=5, gt=0, description="Default prediction horizon in days"
    )
    escalation_confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence below this triggers escalation to Claude API",
    )
    ollama_model: str | None = Field(
        default=None,
        description="Override Ollama model (uses ollama.default_model if None)",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model for escalation",
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM sampling temperature")
    max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")


class AgentsSettings(BaseModel):
    """Configuration for all analysis agents."""

    earnings_linguist: EarningsLinguistSettings = Field(
        default_factory=EarningsLinguistSettings,
        description="Earnings Linguist agent settings",
    )
    narrative_divergence: NarrativeDivergenceSettings = Field(
        default_factory=NarrativeDivergenceSettings,
        description="Narrative vs Price Divergence agent settings",
    )
    sec_filing_analyzer: SecFilingAnalyzerSettings = Field(
        default_factory=SecFilingAnalyzerSettings,
        description="SEC Filing Pattern Analyzer agent settings",
    )
    contagion_mapper: ContagionMapperSettings = Field(
        default_factory=ContagionMapperSettings,
        description="Cross-Sector Contagion Mapper agent settings",
    )


class ScoringSettings(BaseModel):
    """Prediction scoring runner configuration."""

    horizon_buffer_days: int = Field(
        default=1,
        ge=0,
        description="Days past horizon before a prediction is eligible for scoring",
    )
    lookback_days: int = Field(
        default=365,
        gt=0,
        description="Maximum calendar days back to consider predictions for scoring",
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
    agents: AgentsSettings = Field(
        default_factory=AgentsSettings, description="Agent configuration"
    )
    scoring: ScoringSettings = Field(
        default_factory=ScoringSettings, description="Scoring runner settings"
    )
