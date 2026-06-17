"""
Per-provider LLM connectivity tests and resolve_max_tokens unit tests.

Each provider test auto-skips if the corresponding API key is not set.
Run with: pytest tests/test_llm_providers.py -v
"""
import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent
load_dotenv(root_dir / ".env")
sys.path.insert(0, str(root_dir))

from core.llm_wrapper import call_llm, resolve_max_tokens

SIMPLE_PROMPT = "Reply with exactly one word: hello"
SYSTEM_PROMPT = "You are a test assistant. Be brief."


@pytest.mark.provider
def test_openai_connectivity():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    response = call_llm(
        prompt=SIMPLE_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        provider="openai",
        model="gpt-5.4-mini",
        temperature=0.0,
        max_retries=1,
        role="extraction"
    )
    assert response["success"], f"OpenAI call failed: {response.get('error')}"
    assert response["content"] is not None
    assert len(response["content"].strip()) > 0


@pytest.mark.provider
def test_deepseek_connectivity():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set")
    response = call_llm(
        prompt=SIMPLE_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        provider="deepseek",
        model="deepseek-v4-flash",
        temperature=0.0,
        max_retries=1,
        role="extraction"
    )
    assert response["success"], f"DeepSeek call failed: {response.get('error')}"
    assert response["content"] is not None
    assert len(response["content"].strip()) > 0


@pytest.mark.provider
def test_gemini_connectivity():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")
    response = call_llm(
        prompt=SIMPLE_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        provider="gemini",
        model="gemini-3.1-flash-lite",
        temperature=0.0,
        max_retries=1,
        role="extraction"
    )
    assert response["success"], f"Gemini call failed: {response.get('error')}"
    assert response["content"] is not None
    assert len(response["content"].strip()) > 0


@pytest.mark.provider
def test_anthropic_connectivity():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    response = call_llm(
        prompt=SIMPLE_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        provider="anthropic",
        model="claude-haiku-4-5",
        temperature=0.0,
        max_retries=1,
        role="extraction"
    )
    assert response["success"], f"Anthropic call failed: {response.get('error')}"
    assert response["content"] is not None
    assert len(response["content"].strip()) > 0


@pytest.mark.provider
def test_custom_openai_connectivity():
    if not os.environ.get("CUSTOM_OPENAI_API_BASE") or not os.environ.get("CUSTOM_OPENAI_API_KEY"):
        pytest.skip("CUSTOM_OPENAI_API_BASE or CUSTOM_OPENAI_API_KEY not set")
    response = call_llm(
        prompt=SIMPLE_PROMPT,
        system_prompt=SYSTEM_PROMPT,
        provider="custom_openai",
        model="gpt-5",
        temperature=1.0,
        max_retries=1,
        role="extraction"
    )
    assert response["success"], f"Custom OpenAI call failed: {response.get('error')}"
    assert response["content"] is not None


class TestResolveMaxTokens:
    def test_env_override_takes_priority(self, monkeypatch):
        monkeypatch.setenv("MAX_TOKENS_EXTRACTION", "32768")
        result = resolve_max_tokens("openai", "gpt-5.4-mini", role="extraction")
        assert result == 32768

    def test_litellm_detection_for_known_model(self):
        result = resolve_max_tokens("openai", "gpt-5.4-mini")
        assert result > 0

    def test_fallback_for_unknown_model(self, monkeypatch):
        monkeypatch.setenv("TOKEN_LIMIT_DEFAULT", "16384")
        monkeypatch.delenv("MAX_TOKENS_EXTRACTION", raising=False)
        result = resolve_max_tokens("unknown", "nonexistent-model-xyz", role="extraction")
        assert result == 16384

    def test_no_role_uses_litellm_or_fallback(self):
        result = resolve_max_tokens("anthropic", "claude-haiku-4-5")
        assert result > 0

    def test_default_fallback_value(self, monkeypatch):
        monkeypatch.delenv("TOKEN_LIMIT_DEFAULT", raising=False)
        result = resolve_max_tokens("unknown", "nonexistent-model-xyz")
        assert result == 16384
