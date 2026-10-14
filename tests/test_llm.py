from types import SimpleNamespace

import pytest

from agent.llm import AnthropicLLM, AzureOpenAILLM, CachedLLM, LLMUnavailable, OfflineLLM, OpenAILLM, available_providers, get_llm


def test_offline_llm_returns_scripted_replies_in_order():
    llm = OfflineLLM(["one", "two"])
    assert llm.complete("sys", [{"role": "user", "content": "a"}]).text == "one"
    assert llm.complete("sys", [{"role": "user", "content": "b"}]).text == "two"
    assert len(llm.calls) == 2
    with pytest.raises(LLMUnavailable):
        llm.complete("sys", [])


def test_offline_llm_accepts_a_callable_and_reports_tokens():
    llm = OfflineLLM(lambda system, messages: messages[-1]["content"].upper())
    r = llm.complete("sys", [{"role": "user", "content": "hello"}])
    assert r.text == "HELLO" and r.input_tokens >= 1 and r.output_tokens >= 1 and r.cost_usd == 0.0


def test_cache_avoids_second_call_and_marks_hit():
    inner = OfflineLLM(["first", "second"])
    llm = CachedLLM(inner)
    msgs = [{"role": "user", "content": "same question"}]
    a = llm.complete("sys", msgs)
    b = llm.complete("sys", msgs)
    assert a.text == b.text == "first"
    assert b.cached is True and a.cached is False
    assert len(inner.calls) == 1 and (llm.hits, llm.misses) == (1, 1)


def test_cache_key_changes_with_prompt():
    inner = OfflineLLM(["x", "y"])
    llm = CachedLLM(inner)
    llm.complete("sys", [{"role": "user", "content": "q1"}])
    llm.complete("sys", [{"role": "user", "content": "q2"}])
    assert len(inner.calls) == 2


def test_cache_persists_to_disk(tmp_path):
    msgs = [{"role": "user", "content": "persist me"}]
    first = CachedLLM(OfflineLLM(["stored"]), cache_dir=tmp_path)
    first.complete("sys", msgs)
    second_inner = OfflineLLM([])  # would raise if it were called
    second = CachedLLM(second_inner, cache_dir=tmp_path)
    r = second.complete("sys", msgs)
    assert r.text == "stored" and r.cached is True and second_inner.calls == []


class FakeAnthropicClient:
    def __init__(self):
        self.kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="the answer")],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )


def test_anthropic_backend_parses_response_and_computes_cost():
    client = FakeAnthropicClient()
    llm = AnthropicLLM(model="test-model", client=client, price_in=3.0, price_out=15.0)
    r = llm.complete("be brief", [{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.0)
    assert r.text == "the answer" and (r.input_tokens, r.output_tokens) == (1000, 200)
    assert r.cost_usd == pytest.approx((1000 * 3.0 + 200 * 15.0) / 1e6)
    assert client.kwargs["model"] == "test-model" and client.kwargs["system"] == "be brief"
    assert client.kwargs["max_tokens"] == 50 and client.kwargs["temperature"] == 0.0


def test_anthropic_backend_omits_temperature_when_none():
    client = FakeAnthropicClient()
    AnthropicLLM(model="m", client=client).complete("s", [], temperature=None)
    assert "temperature" not in client.kwargs


def test_cost_is_none_without_prices(monkeypatch):
    monkeypatch.delenv("AGENTGUARD_PRICE_IN", raising=False)
    monkeypatch.delenv("AGENTGUARD_PRICE_OUT", raising=False)
    r = AnthropicLLM(model="m", client=FakeAnthropicClient()).complete("s", [])
    assert r.cost_usd is None


ALL_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AGENTGUARD_PROVIDER"]


@pytest.fixture
def no_keys(monkeypatch):
    for k in ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_get_llm_without_any_key_gives_a_clear_error(no_keys):
    with pytest.raises(LLMUnavailable) as err:
        get_llm()
    assert "ANTHROPIC_API_KEY" in str(err.value) and "OPENAI_API_KEY" in str(err.value)


class FakeOpenAIClient:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="the answer"))],
                               usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=200))


def test_openai_backend_puts_the_system_prompt_first_and_parses_usage():
    client = FakeOpenAIClient()
    r = OpenAILLM(model="m", client=client, price_in=2.0, price_out=10.0).complete(
        "be brief", [{"role": "user", "content": "hi"}], max_tokens=50, temperature=0.0)
    assert r.text == "the answer" and (r.input_tokens, r.output_tokens) == (1000, 200)
    assert r.cost_usd == pytest.approx((1000 * 2.0 + 200 * 10.0) / 1e6)
    assert client.kwargs["messages"][0] == {"role": "system", "content": "be brief"}
    assert client.kwargs["messages"][1]["content"] == "hi" and client.kwargs["max_completion_tokens"] == 50


def test_openai_backend_omits_temperature_when_none():
    client = FakeOpenAIClient()
    OpenAILLM(model="m", client=client).complete("s", [], temperature=None)
    assert "temperature" not in client.kwargs


def test_azure_uses_the_deployment_name_as_the_model(no_keys):
    no_keys.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    client = FakeOpenAIClient()
    llm = AzureOpenAILLM(client=client)
    llm.complete("s", [])
    assert llm.model == "my-deployment" and client.kwargs["model"] == "my-deployment"


def test_azure_without_a_deployment_name_is_a_clear_error(no_keys):
    no_keys.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    with pytest.raises(LLMUnavailable) as err:
        AzureOpenAILLM()
    assert "AZURE_OPENAI_DEPLOYMENT" in str(err.value)


def test_available_providers_follow_the_environment(no_keys):
    assert available_providers() == []
    no_keys.setenv("OPENAI_API_KEY", "x")
    no_keys.setenv("AZURE_OPENAI_API_KEY", "x")
    assert available_providers() == ["openai"]  # azure also needs its endpoint
    no_keys.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    assert available_providers() == ["openai", "azure"]


def test_get_llm_picks_the_requested_provider_and_explains_a_missing_key(no_keys):
    no_keys.setenv("OPENAI_API_KEY", "test-key")
    llm = get_llm(provider="openai")
    assert isinstance(llm, OpenAILLM)
    with pytest.raises(LLMUnavailable) as err:
        get_llm(provider="anthropic")
    assert "ANTHROPIC_API_KEY" in str(err.value)
    with pytest.raises(LLMUnavailable):
        get_llm(provider="nonsense")


def test_the_provider_environment_variable_chooses_the_default(no_keys):
    no_keys.setenv("OPENAI_API_KEY", "k")
    no_keys.setenv("ANTHROPIC_API_KEY", "k")
    no_keys.setenv("AGENTGUARD_PROVIDER", "openai")
    assert isinstance(get_llm(), OpenAILLM)
