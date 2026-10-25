"""`LLMAdapter`: the single entry point for every model call (spec §17, §9.4; PLAN.md §1, S0.5).

    admission = ProviderAdmission(models.rate_limits, pools.llm_concurrency)  # once per process
    adapter = LLMAdapter.from_run_config(run_config, events=event_log, admission=admission)
    response = await adapter.complete(messages, tier=ModelTier.LEAD, context=ctx,
                                      tools=tools, output_schema=StrategyReport)

One `complete` call, in order:
1. Resolve the tier to a model through the run's profile (`dev` / `eval`).
2. Refuse the call if the model's price is not confirmed (its cost could not be budgeted).
3. Take a provider concurrency permit, then a slot in the model's rate-limit windows (stop at once
   if the daily cap is reached), both from the process-wide `ProviderAdmission`, and send one
   attempt; retry transient failures with bounded exponential backoff and jitter.
4. If retries are exhausted and the profile allows it, fall back (dev only; never in eval).
5. Validate structured output against the Pydantic schema; on failure, ask once more with the
   validation error in context, then give up with `StructuredOutputError`.
6. Every attempt that reaches a final outcome (success, schema failure, or failure after
   retries) is recorded as one `model_call` event: provider, model, profile, tier, tokens,
   latency, admission wait, retries, cost, fallback flag/origin, redacted input/output refs. A
   response the provider completed but that could not be decoded still records its billed
   tokens and cost (`PROVIDER_RESPONSE_INVALID`).
"""

import asyncio
import json
import random
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, JsonValue, ValidationError

from hypora.core.config.models import (
    PLACEHOLDER_PREFIX,
    ModelsConfig,
    ProfileConfig,
    TierModel,
)
from hypora.core.config.run_config import RunConfig
from hypora.core.config.settings import ConfigError
from hypora.core.llm.credentials import ProviderCredentials, load_credentials
from hypora.core.llm.errors import (
    OUTPUT_SCHEMA_INVALID,
    LLMCallError,
    LLMError,
    ProviderError,
    StructuredOutputError,
)
from hypora.core.llm.pricing import PriceTable, load_price_table
from hypora.core.llm.providers.base import ProviderClient, ProviderRequest, ProviderResult
from hypora.core.llm.rate_limit import (
    Clock,
    ProviderAdmission,
    Reservation,
    Sleep,
    estimate_tokens,
)
from hypora.core.llm.retry import Backoff
from hypora.core.llm.sinks import EventSink, IOKind, ModelIOStore, redact
from hypora.core.llm.types import CallContext, LLMResponse, Message, MessageRole, ToolSpec
from hypora.core.schemas.events import (
    CallStatus,
    EventType,
    FallbackOrigin,
    ModelCallPayload,
    ModelProfile,
    ModelTier,
    Provider,
)

_MAX_VALIDATION_FEEDBACK = 2000

SCHEMA_RETRY_PROMPT = (
    "Your previous reply did not validate against the required JSON schema `{schema}`.\n"
    "Validation errors:\n{errors}\n"
    "Reply again with only a JSON value that matches the schema exactly."
)


@dataclass(frozen=True)
class _Outcome:
    response: LLMResponse
    validation_error: str | None


@dataclass
class _Ledger:
    """Every `model_call` event one `complete()` produced (attempts with usage, a schema retry,
    a fallback): their summed list-price cost and their call IDs, for budget accounting."""

    total_cost_usd: float = 0.0
    call_ids: list[str] = field(default_factory=list)

    def record(self, payload: ModelCallPayload) -> None:
        self.total_cost_usd += payload.cost_usd
        self.call_ids.append(payload.call_id)


class LLMAdapter:
    """Profile resolution, pricing, rate limits, retries, fallback, validation, and events."""

    def __init__(
        self,
        *,
        models: ModelsConfig,
        profile: ModelProfile,
        prices: PriceTable,
        clients: Mapping[Provider, ProviderClient],
        events: EventSink,
        admission: ProviderAdmission,
        io_store: ModelIOStore | None = None,
        secrets: Iterable[str] = (),
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        """`admission` is the process-wide `ProviderAdmission` shared by every adapter, so rate
        limits, daily caps, and concurrency caps hold across runs (it is required on purpose:
        a private one per adapter would let each new run reset the limits)."""
        if admission.rate_limits != models.rate_limits:
            raise ConfigError(
                "the shared ProviderAdmission was built with different rate_limits than this "
                "run's models config; one process must enforce one set of limits"
            )
        self._models = models
        self._profile = profile
        self._prices = prices
        self._clients = dict(clients)
        self._events = events
        self._admission = admission
        self._io_store = io_store
        self._secrets = tuple(secrets)
        self._clock = clock
        self._sleep = sleep
        self._backoff = Backoff(models.retry, (rng or random.Random()).random)
        self._check_ready()

    @classmethod
    def from_run_config(
        cls,
        run_config: RunConfig,
        *,
        events: EventSink,
        admission: ProviderAdmission,
        io_store: ModelIOStore | None = None,
        credentials: ProviderCredentials | None = None,
        prices: PriceTable | None = None,
    ) -> "LLMAdapter":
        """Build real SDK clients for exactly the providers this run's profile can use.

        Pass the process's single `ProviderAdmission` (built once, e.g. with
        `ProviderAdmission(run_config.models.rate_limits, run_config.pools.llm_concurrency)`)
        to every run's adapter.
        """
        credentials = credentials or load_credentials()
        timeout_s = run_config.models.retry.request_timeout_s
        targets = _targets(run_config.models, run_config.model_profile)
        clients = {
            provider: build_client(provider, credentials, timeout_s)
            for provider in {t.provider for t in targets}
        }
        return cls(
            models=run_config.models,
            profile=run_config.model_profile,
            prices=prices or load_price_table(),
            clients=clients,
            events=events,
            admission=admission,
            io_store=io_store,
            secrets=credentials.secret_values(),
        )

    @property
    def profile(self) -> ModelProfile:
        return self._profile

    def _check_ready(self) -> None:
        """Fail at construction, not mid-run: every reachable model is set, priced, and served."""
        active = self._models.profile(self._profile)
        unset = [t.value for t in ModelTier if _is_placeholder(active.tiers.for_tier(t))]
        if unset:
            raise ConfigError(
                f"profile {self._profile.value!r} has placeholder model IDs for tiers {unset}"
            )
        for target in _targets(self._models, self._profile):
            self._prices.require(target.provider, target.model_id)
            if target.provider not in self._clients:
                raise ConfigError(f"no {target.provider.value} client for {target.model_id}")

    # --- public API ---------------------------------------------------------------------------

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        tier: ModelTier,
        context: CallContext,
        tools: Sequence[ToolSpec] = (),
        output_schema: type[BaseModel] | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Run one logical model call (see the module docstring for the full sequence).

        With `output_schema`, `response.parsed` is a validated instance of it, unless the model
        answered with tool calls instead (then `parsed` is None and the caller runs the tools).
        Raises `LLMCallError` when the call fails for good, `StructuredOutputError` when the output
        fails its schema twice, `PriceUnknownError` for an unpriced model.
        """
        if not messages:
            raise ValueError("complete() needs at least one message")
        ledger = _Ledger()
        try:
            response = await self._complete(
                tuple(messages),
                tier,
                context,
                tuple(tools),
                output_schema,
                max_output_tokens,
                ledger,
            )
        except LLMError as error:
            error.total_cost_usd, error.call_ids = ledger.total_cost_usd, tuple(ledger.call_ids)
            raise
        return replace(
            response, total_cost_usd=ledger.total_cost_usd, call_ids=tuple(ledger.call_ids)
        )

    async def _complete(
        self,
        conversation: tuple[Message, ...],
        tier: ModelTier,
        context: CallContext,
        tools: tuple[ToolSpec, ...],
        output_schema: type[BaseModel] | None,
        max_output_tokens: int | None,
        ledger: "_Ledger",
    ) -> LLMResponse:
        args = (tier, context, tools, output_schema, max_output_tokens, ledger)
        outcome = await self._call_with_fallback(conversation, *args)
        if outcome.validation_error is None or output_schema is None:
            return outcome.response
        retry_conversation = (
            *conversation,
            outcome.response.message,
            Message.user(
                SCHEMA_RETRY_PROMPT.format(
                    schema=output_schema.__name__, errors=outcome.validation_error
                )
            ),
        )
        second = await self._call_with_fallback(retry_conversation, *args)
        if second.validation_error is not None:
            raise StructuredOutputError(
                f"{output_schema.__name__}: output failed validation twice "
                f"(calls {outcome.response.call_id}, {second.response.call_id}): "
                f"{second.validation_error}"
            )
        return replace(second.response, structured_output_attempts=2)

    # --- internals ----------------------------------------------------------------------------

    async def _call_with_fallback(
        self,
        messages: tuple[Message, ...],
        tier: ModelTier,
        context: CallContext,
        tools: tuple[ToolSpec, ...],
        output_schema: type[BaseModel] | None,
        max_output_tokens: int | None,
        ledger: "_Ledger",
    ) -> _Outcome:
        profile = self._models.profile(self._profile)
        primary = profile.tiers.for_tier(tier)
        args = (messages, tier, context, tools, output_schema, max_output_tokens, ledger)
        try:
            return await self._call_target(primary, *args, fallback_from=None)
        except LLMCallError as error:
            target = self._fallback_target(profile, tier, error)
            if target is None:
                raise
            origin = FallbackOrigin(
                provider=primary.provider,
                model_id=primary.model_id,
                reason=f"{error.cause.kind.error_code} after {error.retries} retries",
            )
            return await self._call_target(target, *args, fallback_from=origin)

    def _fallback_target(
        self, profile: ProfileConfig, tier: ModelTier, error: LLMCallError
    ) -> TierModel | None:
        """The fallback model for this failure, or None when fallback is not allowed.

        The eval profile never falls back, whatever its configuration says (PLAN.md §1): a
        recorded run must not mix models. Dev falls back only on a configured trigger, and never
        to a placeholder model.
        """
        if self._profile is ModelProfile.EVAL:
            return None
        policy = profile.fallback
        trigger = error.kind.fallback_trigger
        if not policy.enabled or policy.tiers_from is None:
            return None
        if trigger is None or trigger not in policy.triggers:
            return None
        target = self._models.profile(policy.tiers_from).tiers.for_tier(tier)
        if _is_placeholder(target) or target.provider not in self._clients:
            return None
        return target

    async def _call_target(
        self,
        target: TierModel,
        messages: tuple[Message, ...],
        tier: ModelTier,
        context: CallContext,
        tools: tuple[ToolSpec, ...],
        output_schema: type[BaseModel] | None,
        max_output_tokens: int | None,
        ledger: "_Ledger",
        *,
        fallback_from: FallbackOrigin | None,
    ) -> _Outcome:
        """One logical call to one model: admission, attempts with backoff, validation, events.

        Every attempt with provider-reported usage is billed, so it gets its own `model_call`
        event (and ledger entry) before any retry; attempts without usage (429, timeouts, ...)
        are summarized in the next event's `retries` and wait time.
        """
        price = self._prices.require(target.provider, target.model_id)
        client = self._clients[target.provider]
        call_id = f"mc_{uuid.uuid4().hex[:16]}"
        json_schema = output_schema.model_json_schema() if output_schema else None
        request = ProviderRequest(
            model_id=target.model_id,
            temperature=target.temperature,
            messages=messages,
            tools=tools,
            output_schema_name=output_schema.__name__ if output_schema else None,
            output_json_schema=json_schema,
            max_output_tokens=max_output_tokens,
        )
        request_json = _request_payload(request)
        input_ref = self._store(context.run_id, call_id, "input", request_json)
        estimated = estimate_tokens(len(json.dumps(request_json)))
        retries, waited_s, latency_s, events_emitted = 0, 0.0, 0.0, 0

        def event_id() -> str:
            # The first event of this call uses the call ID; any later one names its attempt.
            return call_id if events_emitted == 0 else f"{call_id}-r{retries}"

        def event_fields(**fields: Any) -> dict[str, Any]:
            return {
                "call_id": event_id(),
                "provider": target.provider,
                "model_id": target.model_id,
                "profile": self._profile,
                "tier": tier,
                "temperature": target.temperature,
                "prompt_ref": context.prompt_ref,
                "input_ref": input_ref,
                "output_schema": request.output_schema_name,
                "latency_ms": latency_s * 1000,
                "rate_limit_wait_ms": waited_s * 1000,
                "retries": retries,
                "fallback_used": fallback_from is not None,
                "fallback_from": fallback_from,
                **fields,
            }

        def failure_payload(error: ProviderError) -> ModelCallPayload:
            # A failure that reports usage (a completed-but-failed or undecodable response) was
            # billed: record its tokens and cost.
            billed = error.usage
            return ModelCallPayload(
                **event_fields(
                    input_tokens=billed.input_tokens if billed else 0,
                    output_tokens=billed.output_tokens if billed else 0,
                    cached_input_tokens=billed.cached_input_tokens if billed else 0,
                    cost_usd=price.cost_usd(billed) if billed else 0.0,
                    status=CallStatus.ERROR,
                    error_code=error.kind.error_code,
                )
            )

        while True:
            reservation: Reservation | None = None
            try:
                # Permit first, then the rate-limit slot, then the request: the windows are
                # charged at the moment of dispatch (rate_limit.ProviderAdmission.dispatch).
                admitted_from = self._clock()
                async with self._admission.dispatch(
                    target.provider, target.model_id, estimated
                ) as reservation:
                    started = self._clock()
                    waited_s += started - admitted_from
                    try:
                        result = await client.generate(request)
                    finally:
                        latency_s += self._clock() - started
                    reservation.settle(result.usage.input_tokens)
                break
            except ProviderError as error:
                if error.usage is not None and reservation is not None:
                    reservation.settle(error.usage.input_tokens)
                if self._backoff.should_retry(error, retries):
                    if error.usage is not None:
                        # Billed before the retry: its own event, never folded into the next.
                        await self._emit(context, failure_payload(error), ledger)
                        events_emitted += 1
                        latency_s = waited_s = 0.0  # the next event covers later attempts only
                    delay = self._backoff.delay_s(retries, error.retry_after_s)
                    await self._sleep(delay)
                    waited_s += delay
                    retries += 1
                    continue
                await self._emit(context, failure_payload(error), ledger)
                raise LLMCallError(
                    f"{target.provider.value}/{target.model_id} failed after {retries} "
                    f"retries: {error}",
                    cause=error,
                    retries=retries,
                ) from error

        cost = price.cost_usd(result.usage)
        parsed, validation_error = _validate(output_schema, result)
        final_id = event_id()
        message = Message(
            role=MessageRole.ASSISTANT,
            content=result.text,
            tool_calls=list(result.tool_calls),
            replay=result.replay,
        )
        output_ref = self._store(
            context.run_id,
            final_id,
            "output",
            {
                "text": result.text,
                "tool_calls": [c.model_dump(mode="json") for c in result.tool_calls],
                "finish_reason": result.finish_reason.value,
                "usage": result.usage.model_dump(mode="json"),
                "served_model_id": result.served_model_id,
                "validation_error": validation_error,
            },
        )
        await self._emit(
            context,
            ModelCallPayload(
                **event_fields(
                    output_ref=output_ref,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    cached_input_tokens=result.usage.cached_input_tokens,
                    cost_usd=cost,
                    status=CallStatus.OK if validation_error is None else CallStatus.ERROR,
                    error_code=None if validation_error is None else OUTPUT_SCHEMA_INVALID,
                )
            ),
            ledger,
        )
        response = LLMResponse(
            call_id=final_id,
            provider=target.provider,
            model_id=target.model_id,
            profile=self._profile,
            tier=tier,
            temperature=target.temperature,
            text=result.text,
            tool_calls=result.tool_calls,
            finish_reason=result.finish_reason,
            usage=result.usage,
            cost_usd=cost,
            latency_ms=latency_s * 1000,
            rate_limit_wait_ms=waited_s * 1000,
            retries=retries,
            message=message,
            parsed=parsed,
            fallback_from=fallback_from,
            served_model_id=result.served_model_id,
        )
        return _Outcome(response, validation_error)

    def _store(self, run_id: str, call_id: str, kind: IOKind, payload: JsonValue) -> str | None:
        if self._io_store is None:
            return None
        return self._io_store.put(run_id, call_id, kind, redact(payload, self._secrets))

    async def _emit(
        self, context: CallContext, payload: ModelCallPayload, ledger: "_Ledger"
    ) -> None:
        """Append the `model_call` event (in a thread: the event log is synchronous SQLite) and
        add it to the logical call's ledger."""
        ledger.record(payload)
        await asyncio.to_thread(
            self._events.append,
            context.run_id,
            EventType.MODEL_CALL,
            context.actor_role,
            payload,
            workspace_id=context.workspace_id,
            trial_id=context.trial_id,
            job_id=context.job_id,
            trace_id=context.trace_id,
            span_id=context.span_id,
            parent_span_id=context.parent_span_id,
        )


def build_client(
    provider: Provider, credentials: ProviderCredentials, timeout_s: float
) -> ProviderClient:
    """The real SDK client for `provider` (raises `MissingCredentialError` without a key)."""
    if provider is Provider.GEMINI:
        from hypora.core.llm.providers.gemini_client import GeminiClient

        return GeminiClient(credentials.api_key(provider), timeout_s=timeout_s)
    if provider is Provider.OPENAI:
        from hypora.core.llm.providers.openai_client import OpenAIClient

        return OpenAIClient(credentials.api_key(provider), timeout_s=timeout_s)
    raise ConfigError(f"no client is implemented for provider {provider.value!r}")


def _is_placeholder(target: TierModel) -> bool:
    return target.model_id.startswith(PLACEHOLDER_PREFIX)


def _targets(models: ModelsConfig, profile: ModelProfile) -> list[TierModel]:
    """Every model a run on `profile` can call: its tiers, plus usable fallback models (dev)."""
    active = models.profile(profile)
    targets = [active.tiers.for_tier(t) for t in ModelTier]
    policy = active.fallback
    if profile is not ModelProfile.EVAL and policy.enabled and policy.tiers_from is not None:
        fallback = models.profile(policy.tiers_from).tiers
        targets += [
            fallback.for_tier(t) for t in ModelTier if not _is_placeholder(fallback.for_tier(t))
        ]
    unique: dict[tuple[Provider, str], TierModel] = {}
    for target in targets:
        unique.setdefault((target.provider, target.model_id), target)
    return list(unique.values())


def _validate(
    output_schema: type[BaseModel] | None, result: ProviderResult
) -> tuple[BaseModel | None, str | None]:
    """Validate structured output; tool-call answers are not validated (no final answer yet)."""
    if output_schema is None or result.tool_calls:
        return None, None
    try:
        return output_schema.model_validate_json(result.text), None
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False, include_context=False)
        return None, json.dumps(errors, default=str)[:_MAX_VALIDATION_FEEDBACK]


def _request_payload(request: ProviderRequest) -> dict[str, JsonValue]:
    """The provider-neutral request as stored in the (redacted) input transcript."""
    return {
        "model_id": request.model_id,
        "temperature": request.temperature,
        "messages": [m.model_dump(mode="json", exclude={"replay"}) for m in request.messages],
        "tools": [t.model_dump(mode="json") for t in request.tools],
        "output_schema": request.output_schema_name,
        "output_json_schema": request.output_json_schema,
        "max_output_tokens": request.max_output_tokens,
    }
