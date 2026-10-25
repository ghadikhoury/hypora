"""Errors raised by the LLM adapter. Provider SDK exceptions never escape a provider client: each is
classified into a `ProviderError` with a `ProviderErrorKind`, which decides retry and fallback."""

from enum import StrEnum

from hypora.core.config.models import FallbackTrigger
from hypora.core.llm.types import Usage


class ProviderErrorKind(StrEnum):
    """Why one attempt failed. Only the transient kinds are retried (spec §15.5 fault table)."""

    RATE_LIMIT = "rate_limit"  # 429 for a per-minute limit: back off and retry
    QUOTA = "quota"  # daily cap / no credit: never retried, the caller must stop or fall back
    SERVER_ERROR = "server_error"  # 5xx
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    INVALID_REQUEST = "invalid_request"  # 400/422: repair the request, don't repeat it
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"  # e.g. an unknown model ID
    INVALID_RESPONSE = "invalid_response"  # completed (billed) but undecodable: never retried
    UNKNOWN = "unknown"

    @property
    def retryable(self) -> bool:
        return self in _RETRYABLE

    @property
    def error_code(self) -> str:
        """UPPER_SNAKE code recorded in `model_call` events."""
        return _ERROR_CODES[self]

    @property
    def fallback_trigger(self) -> FallbackTrigger | None:
        """The PLAN.md §1 fallback trigger this failure counts as, if any."""
        return _FALLBACK_TRIGGERS.get(self)


_RETRYABLE = frozenset(
    {
        ProviderErrorKind.RATE_LIMIT,
        ProviderErrorKind.SERVER_ERROR,
        ProviderErrorKind.TIMEOUT,
        ProviderErrorKind.CONNECTION,
    }
)

_ERROR_CODES = {
    ProviderErrorKind.RATE_LIMIT: "RATE_LIMITED",
    ProviderErrorKind.QUOTA: "QUOTA_EXHAUSTED",
    ProviderErrorKind.SERVER_ERROR: "PROVIDER_SERVER_ERROR",
    ProviderErrorKind.TIMEOUT: "PROVIDER_TIMEOUT",
    ProviderErrorKind.CONNECTION: "PROVIDER_UNREACHABLE",
    ProviderErrorKind.INVALID_REQUEST: "INVALID_MODEL_REQUEST",
    ProviderErrorKind.AUTHENTICATION: "PROVIDER_AUTH_FAILED",
    ProviderErrorKind.PERMISSION: "PROVIDER_PERMISSION_DENIED",
    ProviderErrorKind.NOT_FOUND: "MODEL_NOT_FOUND",
    ProviderErrorKind.INVALID_RESPONSE: "PROVIDER_RESPONSE_INVALID",
    ProviderErrorKind.UNKNOWN: "PROVIDER_ERROR",
}

_FALLBACK_TRIGGERS = {
    ProviderErrorKind.RATE_LIMIT: FallbackTrigger.RATE_LIMIT,
    ProviderErrorKind.QUOTA: FallbackTrigger.QUOTA,
    ProviderErrorKind.SERVER_ERROR: FallbackTrigger.SERVER_ERROR,
}

OUTPUT_SCHEMA_INVALID = "OUTPUT_SCHEMA_INVALID"
"""Event error code: the model answered, but its output failed the requested Pydantic schema."""


class LLMError(Exception):
    """Base class for every error the adapter raises.

    When raised out of `LLMAdapter.complete`, `total_cost_usd` and `call_ids` cover every
    `model_call` event that call produced before failing, so billed attempts are never lost.
    """

    total_cost_usd: float = 0.0
    call_ids: tuple[str, ...] = ()


class ProviderError(LLMError):
    """One failed provider attempt, classified. The message never contains credentials.

    `usage` is set when the provider completed (and billed) the request but its response could
    not be used, so the adapter still records those tokens and their cost.
    """

    def __init__(
        self,
        kind: ProviderErrorKind,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        usage: Usage | None = None,
    ) -> None:
        super().__init__(f"{kind.value}: {message}")
        self.kind = kind
        self.detail = message
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.usage = usage


class DailyQuotaExhausted(ProviderError):
    """The configured requests-per-day cap was reached client-side; no request was sent."""

    def __init__(self, message: str, *, resets_in_s: float) -> None:
        super().__init__(ProviderErrorKind.QUOTA, message, retry_after_s=resets_in_s)


class LLMCallError(LLMError):
    """A logical call failed for good: retries (and any allowed fallback) are exhausted.

    Its `model_call` events are already in the log; `cause` is the last provider failure.
    """

    def __init__(self, message: str, *, cause: ProviderError, retries: int) -> None:
        super().__init__(message)
        self.cause = cause
        self.retries = retries

    @property
    def kind(self) -> ProviderErrorKind:
        return self.cause.kind


class StructuredOutputError(LLMError):
    """The output failed its schema twice: the first answer and the one retry with the error."""


class PriceUnknownError(LLMError):
    """No confirmed price for a model: the call is refused because its cost can't be budgeted."""


class MissingCredentialError(LLMError):
    """A provider's API key variable is not set. Names the variable, never a value."""
