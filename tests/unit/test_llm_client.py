"""Tests for LLMClient — JSON extraction, error handling, model selection."""

import json

import pytest

from alphaevo.core.config import LLMConfig
from alphaevo.core.llm import LLMClient, LLMNotAvailableError


@pytest.fixture
def llm_config():
    return LLMConfig(
        model="test-model",
        reflect_model="test-reflect-model",
    )


@pytest.fixture
def client(llm_config):
    return LLMClient(llm_config)


# ── JSON extraction ──────────────────────────────────────────────────


class TestExtractJson:
    def test_plain_json(self):
        text = '{"key": "value", "num": 42}'
        assert LLMClient._extract_json(text) == {"key": "value", "num": 42}

    def test_json_in_markdown_fence(self):
        text = 'Some text\n```json\n{"result": true}\n```\nMore text'
        assert LLMClient._extract_json(text) == {"result": True}

    def test_json_in_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert LLMClient._extract_json(text) == {"a": 1}

    def test_json_embedded_in_text(self):
        text = 'Here is the analysis: {"score": 0.8, "reason": "good"} end'
        assert LLMClient._extract_json(text) == {"score": 0.8, "reason": "good"}

    def test_nested_json(self):
        data = {"outer": {"inner": [1, 2, 3]}, "key": "val"}
        text = f"Response: {json.dumps(data)}"
        assert LLMClient._extract_json(text) == data

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            LLMClient._extract_json("not json at all")

    def test_whitespace_stripped(self):
        text = '  \n  {"ok": true}  \n  '
        assert LLMClient._extract_json(text) == {"ok": True}

    def test_multiple_fences_first_valid(self):
        text = '```\nnot json\n```\n```json\n{"valid": 1}\n```'
        assert LLMClient._extract_json(text) == {"valid": 1}


# ── Client construction ──────────────────────────────────────────────


class TestClientInit:
    def test_from_config(self):
        from alphaevo.core.config import AppConfig

        config = AppConfig()
        client = LLMClient.from_config(config)
        assert client.model == config.llm.model
        assert client.reflect_model == config.llm.reflect_model or config.llm.model

    def test_reflect_model_defaults_to_main(self):
        config = LLMConfig(model="main-model")
        client = LLMClient(config)
        assert client.reflect_model == "main-model"

    def test_separate_reflect_model(self):
        config = LLMConfig(model="main", reflect_model="reflect")
        client = LLMClient(config)
        assert client.model == "main"
        assert client.reflect_model == "reflect"


# ── Lazy import error ────────────────────────────────────────────────


class TestLazyImport:
    def test_litellm_not_available_raises(self, client, monkeypatch):
        """When litellm is not installed, chat() should raise LLMNotAvailableError."""
        # Force _litellm to None and mock import failure
        client._litellm = None

        def fake_ensure():
            raise LLMNotAvailableError()

        monkeypatch.setattr(client, "_ensure_litellm", fake_ensure)

        with pytest.raises(LLMNotAvailableError, match="litellm"):
            client.chat([{"role": "user", "content": "hello"}])


# ── Mock LLM calls ───────────────────────────────────────────────────


class TestMockedCalls:
    def _mock_litellm(self, client, response_text):
        """Set up a mock litellm module on the client."""

        class FakeMessage:
            content = response_text

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeLitellm:
            @staticmethod
            def completion(**kwargs):
                return FakeResponse()

        client._litellm = FakeLitellm()

    def test_chat_returns_text(self, client):
        self._mock_litellm(client, "Hello world")
        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == "Hello world"

    def test_chat_json_parses(self, client):
        self._mock_litellm(client, '{"answer": 42}')
        result = client.chat_json([{"role": "user", "content": "?"}])
        assert result == {"answer": 42}

    def test_chat_json_with_fence(self, client):
        self._mock_litellm(client, '```json\n{"x": 1}\n```')
        result = client.chat_json([{"role": "user", "content": "?"}])
        assert result == {"x": 1}

    def test_reflect_uses_reflect_model(self, client, monkeypatch):
        calls = []

        class FakeMessage:
            content = "reflected"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeLitellm:
            @staticmethod
            def completion(**kwargs):
                calls.append(kwargs)
                return FakeResponse()

        client._litellm = FakeLitellm()
        result = client.reflect([{"role": "user", "content": "analyze"}])
        assert result == "reflected"
        assert calls[0]["model"] == "test-reflect-model"

    def test_chat_override_timeout_and_retries(self, client):
        calls = []

        class FakeMessage:
            content = "ok"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeLitellm:
            @staticmethod
            def completion(**kwargs):
                calls.append(kwargs)
                return FakeResponse()

        client._litellm = FakeLitellm()
        result = client.chat(
            [{"role": "user", "content": "analyze"}],
            timeout=17,
            max_retries=0,
        )

        assert result == "ok"
        assert calls[0]["timeout"] == 17
        assert calls[0]["num_retries"] == 0

    def test_reflect_json(self, client):
        self._mock_litellm(client, '{"patterns": ["a", "b"]}')
        result = client.reflect_json([{"role": "user", "content": "?"}])
        assert result == {"patterns": ["a", "b"]}

    @pytest.mark.parametrize(
        ("model_name", "input_temp", "expected"),
        [
            ("gemini/gemini-3.8-flash", 0.4, 1.0),
            ("gemini/gemini-4-carbon", 0.0, 1.0),
            ("openai/gpt-6-astra", 0.4, 1.0),
            ("openai/gpt-5-mini", 0.2, 1.0),
            ("openai/o3-mini", 0.5, 1.0),
            ("deepseek/deepseek-v4-pro", 0.4, 1.0),
            ("anthropic/claude-sonnet-5.5", 0.4, 0.4),
            ("deepseek/deepseek-v4-flash", 0.3, 0.3),
            ("test-model", 0.0, 0.0),
        ],
    )
    def test_effective_temperature_normalizes_gemini3_and_reasoning_models(
        self, model_name, input_temp, expected
    ):
        assert LLMClient._effective_temperature(model_name, input_temp) == expected

    def test_chat_json_normalizes_temperature_for_gemini_3_8_flash(self):
        calls = []

        class FakeMessage:
            content = '{"ok": true}'

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeLitellm:
            @staticmethod
            def completion(**kwargs):
                calls.append(kwargs)
                return FakeResponse()

        client = LLMClient(LLMConfig(model="gemini/gemini-3.8-flash"))
        client._litellm = FakeLitellm()
        result = client.chat_json([{"role": "user", "content": "?"}], temperature=0.4)
        assert result == {"ok": True}
        assert calls[0]["temperature"] == 1.0

    def test_ensure_litellm_configures_drop_params_and_suppress_debug(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        fake_mod = SimpleNamespace(suppress_debug_info=False, drop_params=False)
        monkeypatch.setitem(sys.modules, "litellm", fake_mod)
        client = LLMClient(LLMConfig(model="gemini/gemini-3.8-flash"))
        loaded = client._ensure_litellm()
        assert loaded is fake_mod
        assert fake_mod.suppress_debug_info is True
        assert fake_mod.drop_params is True
