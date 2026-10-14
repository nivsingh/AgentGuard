"""A thin LLM adapter.

Everything else in the project talks to `LLM.complete()` and never to a vendor
SDK, so swapping models (or running offline in tests) touches only this file.

    OfflineLLM     scripted responses for tests, needs no key and no network
    AnthropicLLM   Anthropic, used when ANTHROPIC_API_KEY is set
    OpenAILLM      OpenAI, used when OPENAI_API_KEY is set
    AzureOpenAILLM Azure OpenAI, used when AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT are set
    CachedLLM      wraps any backend and caches by prompt hash (saves cost, makes reruns repeatable)

get_llm(provider) picks one. The same 100 AgentGuard cases run against any of them, so providers
can be compared on identical ground truth (`python -m guard compare`).

Cost is reported only when you provide prices (USD per million tokens) through
AGENTGUARD_PRICE_IN and AGENTGUARD_PRICE_OUT. Prices are not hardcoded because
they change; copy them from the provider's current pricing page.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None = None
    cached: bool = False


def _cost(input_tokens, output_tokens, price_in, price_out):
    if price_in is None or price_out is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _env_price(name):
    value = os.environ.get(name)
    return float(value) if value else None


class OfflineLLM:
    """Scripted backend. `responder` is a list of replies (used in order) or a callable
    (system, messages) -> str. Records every call in `.calls` for assertions."""

    def __init__(self, responder, model="offline"):
        self.model = model
        self.calls = []
        self._responder = responder
        self._i = 0

    def complete(self, system, messages, max_tokens=1024, temperature=0.0):
        start = time.monotonic()
        self.calls.append({"system": system, "messages": messages})
        if callable(self._responder):
            text = self._responder(system, messages)
        else:
            if self._i >= len(self._responder):
                raise LLMUnavailable("OfflineLLM script ran out of responses")
            text = self._responder[self._i]
            self._i += 1
        prompt = system + json.dumps(messages)
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            latency_ms=(time.monotonic() - start) * 1000,
            cost_usd=0.0,
        )


class AnthropicLLM:
    def __init__(self, model=None, client=None, price_in=None, price_out=None):
        self.model = model or os.environ.get("AGENTGUARD_MODEL", DEFAULT_MODEL)
        self.price_in = price_in if price_in is not None else _env_price("AGENTGUARD_PRICE_IN")
        self.price_out = price_out if price_out is not None else _env_price("AGENTGUARD_PRICE_OUT")
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise LLMUnavailable("The anthropic package is not installed: pip install anthropic") from e
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._client = client

    def complete(self, system, messages, max_tokens=1024, temperature=0.0):
        kwargs = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        start = time.monotonic()
        resp = self._client.messages.create(**kwargs)
        latency = (time.monotonic() - start) * 1000
        text = "".join(b.text for b in resp.content if getattr(b, "type", "text") == "text")
        i, o = resp.usage.input_tokens, resp.usage.output_tokens
        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=i,
            output_tokens=o,
            latency_ms=latency,
            cost_usd=_cost(i, o, self.price_in, self.price_out),
        )


def _openai_style_complete(client, model, system, messages, max_tokens, temperature, prices):
    kwargs = {"model": model, "max_completion_tokens": max_tokens,
              "messages": [{"role": "system", "content": system}, *messages]}
    if temperature is not None:
        kwargs["temperature"] = temperature
    start = time.monotonic()
    resp = client.chat.completions.create(**kwargs)
    latency = (time.monotonic() - start) * 1000
    i, o = resp.usage.prompt_tokens, resp.usage.completion_tokens
    return LLMResponse(text=resp.choices[0].message.content or "", model=model, input_tokens=i, output_tokens=o,
                       latency_ms=latency, cost_usd=_cost(i, o, *prices))


class OpenAILLM:
    DEFAULT_MODEL = "gpt-4o"  # set AGENTGUARD_MODEL to use another; model names change, so check the provider's docs

    def __init__(self, model=None, client=None, price_in=None, price_out=None):
        self.model = model or os.environ.get("AGENTGUARD_MODEL", self.DEFAULT_MODEL)
        self.prices = (price_in if price_in is not None else _env_price("AGENTGUARD_PRICE_IN"),
                       price_out if price_out is not None else _env_price("AGENTGUARD_PRICE_OUT"))
        if client is None:
            client = self._make_client()
        self._client = client

    @staticmethod
    def _make_client():
        try:
            import openai
        except ImportError as e:
            raise LLMUnavailable("The openai package is not installed: pip install openai") from e
        return openai.OpenAI()  # reads OPENAI_API_KEY

    def complete(self, system, messages, max_tokens=1024, temperature=0.0):
        return _openai_style_complete(self._client, self.model, system, messages, max_tokens, temperature, self.prices)


class AzureOpenAILLM(OpenAILLM):
    """Azure OpenAI. `model` is the deployment name (AZURE_OPENAI_DEPLOYMENT)."""

    def __init__(self, model=None, client=None, price_in=None, price_out=None):
        model = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or os.environ.get("AGENTGUARD_MODEL")
        if not model and client is None:
            raise LLMUnavailable("Set AZURE_OPENAI_DEPLOYMENT to the name of your Azure OpenAI deployment.")
        super().__init__(model=model, client=client, price_in=price_in, price_out=price_out)

    @staticmethod
    def _make_client():
        try:
            import openai
        except ImportError as e:
            raise LLMUnavailable("The openai package is not installed: pip install openai") from e
        return openai.AzureOpenAI(  # reads AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))


class CachedLLM:
    """Cache by hash of (model, system, messages, max_tokens, temperature)."""

    def __init__(self, inner, cache_dir=None):
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory = {}
        self.hits = 0
        self.misses = 0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, system, messages, max_tokens, temperature):
        blob = json.dumps([self.model, system, messages, max_tokens, temperature], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def complete(self, system, messages, max_tokens=1024, temperature=0.0):
        key = self._key(system, messages, max_tokens, temperature)
        hit = self._memory.get(key)
        if hit is None and self.cache_dir and (self.cache_dir / f"{key}.json").exists():
            hit = LLMResponse(**json.loads((self.cache_dir / f"{key}.json").read_text()))
        if hit is not None:
            self.hits += 1
            self._memory[key] = hit
            return replace(hit, cached=True, latency_ms=0.0, cost_usd=0.0 if hit.cost_usd is not None else None)
        self.misses += 1
        resp = self.inner.complete(system, messages, max_tokens=max_tokens, temperature=temperature)
        self._memory[key] = resp
        if self.cache_dir:
            data = {k: getattr(resp, k) for k in resp.__dataclass_fields__}
            (self.cache_dir / f"{key}.json").write_text(json.dumps(data))
        return resp


PROVIDER_KEYS = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
}
_BACKENDS = {"anthropic": AnthropicLLM, "openai": OpenAILLM, "azure": AzureOpenAILLM}


def available_providers():
    """Providers whose credentials are present in the environment, in a stable order."""
    return [name for name, keys in PROVIDER_KEYS.items() if all(os.environ.get(k) for k in keys)]


def get_llm(cache_dir=None, provider=None):
    """A real backend. `provider` is anthropic, openai or azure; defaults to AGENTGUARD_PROVIDER, then the
    first provider with credentials. Raises a clear error when none is configured.

    Tests build an OfflineLLM directly instead of calling this.
    """
    provider = provider or os.environ.get("AGENTGUARD_PROVIDER") or (available_providers() or [None])[0]
    if provider is None:
        raise LLMUnavailable(
            "No provider is configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or AZURE_OPENAI_API_KEY with "
            "AZURE_OPENAI_ENDPOINT, or pass an OfflineLLM in tests.")
    if provider not in _BACKENDS:
        raise LLMUnavailable(f"Unknown provider {provider!r}. Choose from {', '.join(_BACKENDS)}.")
    missing = [k for k in PROVIDER_KEYS[provider] if not os.environ.get(k)]
    if missing:
        raise LLMUnavailable(f"{', '.join(missing)} is not set, so the {provider} provider is unavailable.")
    llm = _BACKENDS[provider]()
    return CachedLLM(llm, cache_dir) if cache_dir else llm
