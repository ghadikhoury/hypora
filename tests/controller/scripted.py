"""A scripted provider that plays Intake, the single agent, and the Reporter (PLAN.md S1.5).

Unlike `FakeProvider`'s fixed script, it reacts to what each request shows, so the deterministic
end-to-end test can use the real IDs the MCP server hands out: it reads the last tool result and
decides the next call the way a disciplined agent would. Every decision is a pure function of
the conversation, so a run is reproducible.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio

from hypora.core.llm.providers.base import ProviderRequest, ProviderResult
from hypora.core.llm.providers.fake import json_result, text_result, tool_call_result
from hypora.core.llm.types import MessageRole, ToolCall
from hypora.core.schemas import Provider

TEMPLATES = Path(__file__).resolve().parents[2] / "hypora/execution/templates/tabular"
LINEAR = (TEMPLATES / "linear" / "pipeline.py").read_text()
DESIGN = {
    "kind": "baseline",
    "operation": "fit",
    "hypothesis": "A regularized logistic regression is a strong baseline",
    "model_family": "logistic_regression",
    "expected_effect": "roc_auc above 0.95",
    "estimated_cost": {"compute_seconds": 30},
}
NARRATIVE: dict[str, object] = {
    "summary": "The logistic regression baseline reached {{test.roc_auc}} ROC-AUC on the "
    "held-out test split, close to its validation mean of {{validation.roc_auc}}.",
    "observations": ["Validation and test agree, so the estimate looks stable."],
    "limitations": ["Only a baseline was tried in this run."],
}


class ScriptedRun:
    """The provider client for one scripted run."""

    def __init__(
        self,
        run_id: str,
        *,
        poll_seconds: float = 2.0,
        job_limits: dict[str, Any] | None = None,
        on_intake: Callable[[], None] | None = None,
        on_tool_result: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.poll_seconds = poll_seconds
        self.job_limits = job_limits
        self.on_intake = on_intake
        self.on_tool_result = on_tool_result
        self.requests: list[ProviderRequest] = []
        self.tool_calls: list[ToolCall] = []
        self.trial_id: str | None = None
        self.job_id: str | None = None

    @property
    def provider(self) -> Provider:
        return Provider.GEMINI

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        schema = request.output_schema_name
        if schema == "IntakeDraft":
            if self.on_intake is not None:
                self.on_intake()
            return json_result(
                {
                    "objective": "Predict whether a breast tumour is malignant",
                    "target": "target",
                    "task_type": "binary_classification",
                    "primary_metric": "roc_auc",
                }
            )
        if schema == "FinalAnswer":
            return json_result({"final_trial_id": self.trial_id, "confidence": 0.9})
        if schema == "ReportNarrative":
            return json_result(NARRATIVE)
        return await self._agent_turn(request)

    def _call(self, name: str, **arguments: Any) -> ProviderResult:
        call = ToolCall(id=f"call_{len(self.tool_calls) + 1}", name=name, arguments=arguments)
        self.tool_calls.append(call)
        return tool_call_result(call)

    async def _agent_turn(self, request: ProviderRequest) -> ProviderResult:
        last = request.messages[-1]
        if last.role is not MessageRole.TOOL:
            return self._call("inspect_dataset")
        result = json.loads(last.content)
        if self.on_tool_result is not None:
            self.on_tool_result(str(last.tool_name), result)
        if last.is_error:
            return text_result(f"The tool failed: {result.get('code')}. Stopping.")
        if last.tool_name == "inspect_dataset":
            return self._call(
                "write_trial_files",
                idempotency_key=f"{self.run_id}:write_1",
                design=DESIGN,
                pipeline_py=LINEAR,
            )
        if last.tool_name == "write_trial_files":
            self.trial_id = result["trial_id"]
            return self._call(
                "submit_experiment",
                trial_id=self.trial_id,
                idempotency_key=f"{self.run_id}:submit_1",
                **({"limits": self.job_limits} if self.job_limits is not None else {}),
            )
        if last.tool_name == "submit_experiment":
            self.job_id = result["job_id"]
            return self._call("get_experiment_status", job_id=self.job_id)
        if last.tool_name == "get_experiment_status":
            if not result["finished"]:
                await anyio.sleep(self.poll_seconds)
                return self._call("get_experiment_status", job_id=self.job_id)
            return self._call("get_experiment_metrics", job_id=self.job_id)
        return text_result(f"Trial {self.trial_id} is the best (and only) trial.")
