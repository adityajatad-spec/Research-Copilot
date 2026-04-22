"""Configuration helpers for LLM-backed paper summarization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib import error, request

import openai


@dataclass(slots=True)
class Config:
    """Store runtime configuration for summary generation."""

    provider: str = "ollama"
    model: str = "llama3.2"
    api_key_env: str = "OPENAI_API_KEY"
    ollama_base_url: str = "http://localhost:11434/v1"
    max_tokens: int = 512
    temperature: float = 0.2


OLLAMA_NOT_RUNNING_MESSAGE = "Ollama is not running. Start Ollama and pull the required model."


def _get_openai_api_key(config: Config) -> str:
    """Read the configured OpenAI API key from the environment."""
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key. Set the {config.api_key_env} environment variable for provider 'openai'.")
    return api_key


def validate_provider_setup(config: Config) -> None:
    """Validate that the selected provider is configured and reachable."""
    if config.provider == "ollama":
        health_check_url = f"{config.ollama_base_url.rstrip('/')}/models"
        try:
            with request.urlopen(health_check_url, timeout=2):
                return
        except (error.URLError, TimeoutError) as exc:
            raise ValueError(OLLAMA_NOT_RUNNING_MESSAGE) from exc

    if config.provider == "openai":
        _get_openai_api_key(config)
        return

    raise ValueError(f"Unsupported provider: {config.provider}")


def get_client(config: Config) -> openai.OpenAI:
    """Build an OpenAI-compatible client from the active configuration."""
    if config.provider == "ollama":
        return openai.OpenAI(base_url=config.ollama_base_url, api_key="ollama")

    if config.provider == "openai":
        return openai.OpenAI(api_key=_get_openai_api_key(config))

    raise ValueError(f"Unsupported provider: {config.provider}")
