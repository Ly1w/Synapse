from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for a single LLM endpoint."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = "sk-placeholder"
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: float = 120.0


class FrameworkLLMConfig(BaseModel):
    """
    Multi-model configuration for the framework.

    Allows different models for planning (Master/Head) and execution (Node).
    """

    planner: LLMConfig = Field(default_factory=LLMConfig)
    executor: LLMConfig | None = None

    def get_executor(self) -> LLMConfig:
        return self.executor or self.planner
