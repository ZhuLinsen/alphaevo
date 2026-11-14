"""LLM client — thin wrapper around litellm for strategy generation & reflection.

litellm is an optional dependency (pip install alphaevo[llm]).
All imports are lazy to avoid import errors when litellm is not installed.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from alphaevo.core.config import AppConfig, LLMConfig

logger = logging.getLogger(__name__)


class LLMNotAvailableError(RuntimeError):
    """Raised when litellm is not installed."""

    def __init__(self) -> None:
        super().__init__("LLM support requires litellm. Install with: pip install alphaevo[llm]")


class LLMClient:
    """Thin wrapper around litellm for structured LLM calls.

    Features:
    - Lazy import of litellm (only when first call is made)
    - Retry with tenacity
    - Structured JSON response parsing
    - Separate model for reflection (optional)
    """

    def __init__(self, config: LLMConfig, *, api_key: str | None = None) -> None:
        self.model = config.model
        self.reflect_model = config.reflect_model or config.model
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.api_key = api_key
        self._litellm: Any = None

    @classmethod
    def from_config(cls, config: AppConfig) -> LLMClient:
        from alphaevo.core.config import ConfigManager

        api_key = ConfigManager().get_llm_api_key()
        if not api_key:
            logger.warning(
                "No LLM API key configured. Set ALPHAEVO_API_KEY env var. "
                "LLM-based reflection will fall back to heuristics."
            )
        return cls(config.llm, api_key=api_key)

    def _ensure_litellm(self) -> Any:
        """Lazy import litellm."""
        if self._litellm is not None:
            return self._litellm
        try:
            import litellm

            litellm.suppress_debug_info = True
            litellm.drop_params = True
            self._litellm = litellm
            return litellm
        except ImportError as e:
            raise LLMNotAvailableError() from e

    @staticmethod
    def _effective_temperature(model_name: str, temperature: float) -> float:
        """Normalize temperature for Gemini 3/4 and reasoning-first models.

        Gemini 3/4 degrades reasoning below 1.0, while OpenAI GPT-5/6, o-series,
        and DeepSeek reasoning endpoints reject custom non-1.0 temperatures.
        """
        slug = model_name.strip().lower().rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if (
            slug.startswith(("gemini-3", "gemini-4", "gpt-5", "gpt-6", "qwq"))
            or slug in {"o1", "o3", "o4"}
            or slug.startswith(("o1-", "o3-", "o4-"))
            or "deepseek-r1" in slug
            or "deepseek-reasoner" in slug
            or "deepseek-v4-pro" in slug
        ):
            return 1.0
        return temperature

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> str:
        """Send a chat completion request and return the response text."""
        litellm = self._ensure_litellm()
        use_model = model or self.model

        kwargs: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "temperature": self._effective_temperature(use_model, temperature),
            "max_tokens": max_tokens,
            "timeout": timeout if timeout is not None else self.timeout,
            "num_retries": max_retries if max_retries is not None else self.max_retries,
        }
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key

        logger.debug("LLM call: model=%s, messages=%d", use_model, len(messages))
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content or ""
        logger.debug("LLM response: %d chars", len(content))
        return content

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Chat and parse the response as JSON.

        Extracts JSON from markdown code fences if present.
        """
        raw = self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        return self._extract_json_with_retry(
            raw,
            repair_call=lambda repair_messages: self.chat(
                repair_messages,
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            ),
            original_messages=messages,
        )

    def reflect(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.5,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> str:
        """Call using the reflection model (may differ from generation model)."""
        return self.chat(
            messages,
            model=self.reflect_model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )

    def reflect_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.4,
        timeout: int | None = None,
        max_retries: int | None = None,
    ) -> dict[str, Any]:
        """Reflect and parse as JSON."""
        raw = self.reflect(
            messages,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        return self._extract_json_with_retry(
            raw,
            repair_call=lambda repair_messages: self.reflect(
                repair_messages,
                temperature=0.0,
                timeout=timeout,
                max_retries=max_retries,
            ),
            original_messages=messages,
        )

    @classmethod
    def _extract_json_with_retry(
        cls,
        raw: str,
        *,
        repair_call: Any,
        original_messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Parse JSON, retrying once with a strict JSON-only repair prompt."""
        try:
            return cls._extract_json(raw)
        except ValueError:
            repair_messages = list(original_messages) + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Return the same answer as valid JSON only. "
                        "Do not include markdown fences or any explanation."
                    ),
                },
            ]
            repaired = repair_call(repair_messages)
            return cls._extract_json(repaired)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from text, handling markdown code fences."""
        text = text.strip()
        # Try to extract from code fences
        if "```" in text:
            parts = text.split("```")
            for part in parts[1::2]:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    return cast("dict[str, Any]", json.loads(candidate))
                except json.JSONDecodeError:
                    continue
        # Try direct parse
        try:
            return cast("dict[str, Any]", json.loads(text))
        except json.JSONDecodeError:
            # Try to find first { ... } block
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end > start:
                try:
                    return cast("dict[str, Any]", json.loads(text[start : end + 1]))
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}") from None
