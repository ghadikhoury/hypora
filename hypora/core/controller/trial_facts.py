"""Which contract-allowed features a trial's pipeline leaves out, by a static reading (D-058).

Live run `run_f8f3054c80e0` (S4.2j, D-058): the approved contract allowed every feature, yet C's
trials removed the strongest ones by name, C nominated a trial far worse on the primary metric
than an all-feature trial, and A's final review praised the exclusion. Nothing told either role
that the nominated pipeline dropped allowed features. The controller now computes it for every
validated trial and shows it, with the trials' validation ranking, to C before it nominates and
to A's final review. It never forces a choice.

This module reads `pipeline.py` with `ast` and never runs it. It reports only what the code
states by name:

- `dropped`: allowed features the code removes by name: a `drop(...)` call on columns
  (`columns=`, or `axis=1`/`axis="columns"`), `<columns>.difference(...)`, a set subtraction, a
  comprehension filter `if c not in [...]`, or a `"drop"` entry of a `ColumnTransformer` /
  `make_column_transformer`.
- `not_selected`: when a `ColumnTransformer` (whose default `remainder` drops every column it does
  not name) lists all its columns by name, or a comprehension keeps `if c in [...]`, the allowed
  features it never names.
- `unresolved`: places where the code removes or selects columns in a way a static reading cannot
  resolve (names built at run time). Column lists derived from the data itself (for example
  `select_dtypes`, as in the platform's templates) are not reported as removals.

Names are resolved through literal assignments (`DROP = ["a", "b"]`), simple concatenations, and
the trial's own `config` (`ctx.config["drop"]`, `ctx.config.get("drop", ...)`).
"""

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

_TRANSFORMERS: Final = frozenset({"ColumnTransformer", "make_column_transformer"})
_RESOLVE_PASSES: Final = 4
"""Rounds of resolving assignments that refer to other assigned names (a hard bound)."""


@dataclass(frozen=True)
class FeatureUse:
    """What a static reading of one pipeline says about the contract's allowed features."""

    parses: bool
    dropped: tuple[str, ...] = ()
    not_selected: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    def not_used(self, allowed: Sequence[str]) -> list[str]:
        """Allowed features the code removes or never selects by name, in contract order."""
        gone = set(self.dropped) | set(self.not_selected)
        return [name for name in allowed if name in gone]


def _func_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_config(node: ast.expr) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "config") or (
        isinstance(node, ast.Name) and node.id == "config"
    )


def _from_config(value: Any) -> set[str] | None:
    if isinstance(value, str):
        return {value}
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return set(value)
    return None


class _Reader:
    def __init__(self, tree: ast.AST, config: Mapping[str, Any]) -> None:
        self.tree = tree
        self.config = config
        self.names: dict[str, set[str] | None] = {}
        assignments: list[tuple[str, ast.expr]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assignments += [
                    (target.id, node.value)
                    for target in node.targets
                    if isinstance(target, ast.Name)
                ]
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.value is not None
            ):
                assignments.append((node.target.id, node.value))
        for _ in range(_RESOLVE_PASSES):
            resolved: dict[str, set[str] | None] = {}
            for name, value in assignments:
                found = self.resolve(value)
                if name in resolved:
                    held = resolved[name]
                    resolved[name] = None if held is None or found is None else held | found
                else:
                    resolved[name] = found
            if resolved == self.names:
                break
            self.names = resolved

    def resolve(self, node: ast.expr | None) -> set[str] | None:
        """The string names `node` stands for, or None when it is not resolvable statically."""
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            return {node.value} if isinstance(node.value, str) else None
        if isinstance(node, ast.List | ast.Tuple | ast.Set):
            found: set[str] = set()
            for item in node.elts:
                part = self.resolve(item.value if isinstance(item, ast.Starred) else item)
                if part is None:
                    return None
                found |= part
            return found
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.BitOr):
            left, right = self.resolve(node.left), self.resolve(node.right)
            return None if left is None or right is None else left | right
        if isinstance(node, ast.Subscript) and _is_config(node.value):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return _from_config(self.config.get(key.value))
            return None
        if isinstance(node, ast.Call):
            name = _func_name(node.func)
            if name in {"list", "set", "tuple", "sorted", "frozenset"} and len(node.args) == 1:
                return self.resolve(node.args[0])
            if (
                name == "get"
                and isinstance(node.func, ast.Attribute)
                and _is_config(node.func.value)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                if node.args[0].value in self.config:
                    return _from_config(self.config[node.args[0].value])
                return self.resolve(node.args[1]) if len(node.args) > 1 else set()
        return None


def _drop_columns(call: ast.Call) -> tuple[bool, ast.expr | None]:
    """Whether a `drop(...)` call drops columns, and the expression naming them."""
    keywords = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
    if "columns" in keywords:
        return True, keywords["columns"]
    axis = keywords.get("axis")
    by_columns = isinstance(axis, ast.Constant) and axis.value in (1, "columns")
    if not by_columns and len(call.args) >= 2:
        second = call.args[1]
        by_columns = isinstance(second, ast.Constant) and second.value in (1, "columns")
    if not by_columns:
        return False, None  # pandas' default drops rows by label
    return True, keywords.get("labels", call.args[0] if call.args else None)


def feature_use(
    source: str, allowed: Sequence[str], config: Mapping[str, Any] | None = None
) -> FeatureUse:
    """A static reading of `source` (module docstring); never executes it."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return FeatureUse(parses=False)
    reader = _Reader(tree, config or {})
    allowed_set = set(allowed)
    dropped: set[str] = set()
    selections: list[set[str]] = []
    unresolved: list[str] = []

    def remove(names: set[str] | None, what: str) -> None:
        if names is None:
            if what not in unresolved:
                unresolved.append(what)
        else:
            dropped.update(names & allowed_set)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _func_name(node.func)
            if name == "drop" and isinstance(node.func, ast.Attribute):
                columns, expr = _drop_columns(node)
                if columns:
                    remove(reader.resolve(expr), "a drop(...) call on columns not named in code")
            elif name == "difference" and isinstance(node.func, ast.Attribute) and node.args:
                remove(reader.resolve(node.args[0]), "a difference(...) of columns not named")
            elif name in _TRANSFORMERS:
                _column_transformer(node, name, reader, remove, selections, unresolved)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            right = reader.resolve(node.right)
            if right is not None and right & allowed_set:
                dropped.update(right & allowed_set)
        elif isinstance(node, ast.comprehension):
            for condition in node.ifs:
                for compare in ast.walk(condition):
                    if not isinstance(compare, ast.Compare):
                        continue
                    for op, comparator in zip(compare.ops, compare.comparators, strict=True):
                        if isinstance(comparator, ast.Constant):
                            continue  # a substring test ("id" in name), not a column list
                        if isinstance(op, ast.NotIn):
                            # A filter by a list derived from the data (the templates split
                            # numeric from the rest this way) removes nothing by name.
                            removed = reader.resolve(comparator)
                            if removed is not None:
                                dropped.update(removed & allowed_set)
                        elif isinstance(op, ast.In):
                            kept = reader.resolve(comparator)
                            if kept is not None and kept & allowed_set:
                                selections.append(kept)
    not_selected: set[str] = set()
    for kept in selections:
        not_selected |= allowed_set - kept - dropped
    return FeatureUse(
        parses=True,
        dropped=tuple(name for name in allowed if name in dropped),
        not_selected=tuple(name for name in allowed if name in not_selected),
        unresolved=tuple(unresolved),
    )


def _column_transformer(
    node: ast.Call,
    name: str,
    reader: _Reader,
    remove: Callable[[set[str] | None, str], None],
    selections: list[set[str]],
    unresolved: list[str],
) -> None:
    """Record a ColumnTransformer's by-name drops and, when it names every column, its selection."""
    keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
    if name == "ColumnTransformer":
        spec = keywords.get("transformers", node.args[0] if node.args else None)
        entries = spec.elts if isinstance(spec, ast.List | ast.Tuple) else None
        column_index = 2
    else:
        entries = list(node.args)
        column_index = 1
    if entries is None:
        unresolved.append("a ColumnTransformer whose transformers are not written in code")
        return
    remainder = keywords.get("remainder")
    drops_rest = remainder is None or (
        isinstance(remainder, ast.Constant) and remainder.value == "drop"
    )
    selected: set[str] = set()
    complete = True
    for entry in entries:
        if not isinstance(entry, ast.Tuple) or len(entry.elts) <= column_index:
            complete = False
            continue
        transformer = entry.elts[column_index - 1]
        columns = reader.resolve(entry.elts[column_index])
        if isinstance(transformer, ast.Constant) and transformer.value == "drop":
            remove(columns, "a ColumnTransformer drop entry not named in code")
        elif columns is None:
            complete = False  # derived from the data (e.g. select_dtypes): not a removal
        else:
            selected |= columns
    if drops_rest and complete:
        selections.append(selected)


__all__ = ["FeatureUse", "feature_use"]
