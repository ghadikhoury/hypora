"""Validation of C's evidence-backed data methodology proposals."""

from collections.abc import Collection, Mapping

from hypora.core.schemas import DataContract, TaskSpecification


class MethodologyError(ValueError):
    """A proposed methodology is inconsistent with the task or its evidence."""


def validate_data_proposal(
    proposal: DataContract,
    task: TaskSpecification,
    columns: Collection[str],
    evidence: Mapping[str, tuple[str, str]],
) -> None:
    """Fail closed when a proposed contract changes scope or lacks exclusion evidence."""
    known = set(columns)
    if proposal.data_contract_version != 1:
        raise MethodologyError("initial data contract version must be one")
    if proposal.target != task.target or proposal.task_type != task.task_type:
        raise MethodologyError("target and task type must match the task specification")
    if proposal.target not in known:
        raise MethodologyError("target is not present in the dataset")
    allowed = set(proposal.allowed_features)
    excluded = {item.name for item in proposal.excluded_features}
    if len(excluded) != len(proposal.excluded_features):
        raise MethodologyError("excluded features must be unique")
    if allowed | excluded != known - {proposal.target}:
        raise MethodologyError("every non-target column must be accounted for")
    forbidden = set(task.forbidden_columns) & known
    if forbidden & allowed or not forbidden <= excluded:
        raise MethodologyError("the proposal must exclude the user's forbidden columns")
    for item in proposal.excluded_features:
        if item.name in forbidden:
            if item.evidence != f"task_spec:v{task.version}":
                raise MethodologyError("user exclusions must cite the task specification")
        elif item.evidence not in evidence or evidence[item.evidence][1] != item.name:
            raise MethodologyError("a proposed exclusion lacks matching diagnostic evidence")
    for key in (proposal.split.group_key, proposal.split.time_key):
        if key is not None and (key not in known or key == proposal.target):
            raise MethodologyError("split keys must name non-target dataset columns")
    for kind, key in (
        ("group_overlap", proposal.split.group_key),
        ("timestamp_order", proposal.split.time_key),
    ):
        if key is not None and (kind, key) not in evidence.values():
            raise MethodologyError("the proposed split key lacks matching diagnostic evidence")
    if proposal.open_questions:
        raise MethodologyError("unresolved methodology questions require escalation")
