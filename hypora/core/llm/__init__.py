"""LLM provider adapter, model tiers, and token accounting (spec §17).

`LLMAdapter.complete(messages, tier=..., context=..., tools=..., output_schema=...)` is the only way
the platform calls a model. Provider SDKs (Gemini, OpenAI) sit behind `ProviderClient`s; every call
is priced, rate-limited, retried with bounded backoff, validated, and recorded as a `model_call`
event. See docs/notes/llm-adapter.md.
"""

from hypora.core.llm.adapter import LLMAdapter, build_client
from hypora.core.llm.credentials import (
    API_KEY_VARIABLES,
    ProviderCredentials,
    load_credentials,
)
from hypora.core.llm.errors import (
    DailyQuotaExhausted,
    LLMCallError,
    LLMError,
    MissingCredentialError,
    PriceUnknownError,
    ProviderError,
    ProviderErrorKind,
    StructuredOutputError,
)
from hypora.core.llm.pricing import ModelPrice, PriceTable, TokenPrices, load_price_table
from hypora.core.llm.providers.base import ProviderClient, ProviderRequest, ProviderResult
from hypora.core.llm.providers.fake import FakeProvider
from hypora.core.llm.rate_limit import ProviderAdmission
from hypora.core.llm.sinks import (
    EventSink,
    FileModelIOStore,
    InMemoryEventSink,
    InMemoryModelIOStore,
    ModelIOStore,
    redact,
)
from hypora.core.llm.types import (
    CallContext,
    FinishReason,
    LLMResponse,
    Message,
    MessageRole,
    ProviderReplay,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "API_KEY_VARIABLES",
    "CallContext",
    "DailyQuotaExhausted",
    "EventSink",
    "FakeProvider",
    "FileModelIOStore",
    "FinishReason",
    "InMemoryEventSink",
    "InMemoryModelIOStore",
    "LLMAdapter",
    "LLMCallError",
    "LLMError",
    "LLMResponse",
    "Message",
    "MessageRole",
    "MissingCredentialError",
    "ModelIOStore",
    "ModelPrice",
    "PriceTable",
    "PriceUnknownError",
    "ProviderAdmission",
    "ProviderClient",
    "ProviderCredentials",
    "ProviderError",
    "ProviderErrorKind",
    "ProviderReplay",
    "ProviderRequest",
    "ProviderResult",
    "StructuredOutputError",
    "TokenPrices",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "build_client",
    "load_credentials",
    "load_price_table",
    "redact",
]
