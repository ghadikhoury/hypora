"""Prompts are versioned files, loaded at runtime (`hypora/core/roles/prompts/*.md`).

Each file starts with a header naming the prompt and its version:

    ---
    prompt: single_agent
    version: 1
    ---
    <the prompt text>

`Prompt.ref` (`single_agent@v1`) is recorded on every model call that uses it (`prompt_ref` in the
`model_call` event), so a trace always says which prompt text produced which call. A prompt may
contain `$name` placeholders, filled by `render` (`string.Template`: a missing value is an
error, and `$$` is a literal dollar). Prompt text is never inlined in code.
"""

import re
import string
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

PROMPTS_DIR: Final = Path(__file__).resolve().parent / "prompts"
_HEADER = re.compile(r"\A---\nprompt: (?P<name>[a-z][a-z0-9_]*)\nversion: (?P<version>\d+)\n---\n")


class PromptError(ValueError):
    """A prompt file is missing or has no valid version header."""


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str

    @property
    def ref(self) -> str:
        """The `prompt_ref` recorded in model_call events, e.g. `single_agent@v1`."""
        return f"{self.name}@v{self.version}"

    def render(self, **values: object) -> str:
        """The text with every `$name` placeholder filled (each one must be given)."""
        try:
            return string.Template(self.text).substitute(
                {key: str(value) for key, value in values.items()}
            )
        except KeyError as missing:
            raise PromptError(f"prompt {self.ref} needs a value for {missing}") from None


@cache
def load_prompt(name: str, directory: Path = PROMPTS_DIR) -> Prompt:
    """Read `<directory>/<name>.md`; its header must name the same prompt."""
    path = directory / f"{name}.md"
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise PromptError(f"no prompt file {path}") from None
    header = _HEADER.match(content)
    if header is None:
        raise PromptError(f"{path} must start with a '---' header with prompt and version")
    if header["name"] != name:
        raise PromptError(f"{path} declares prompt {header['name']!r}, expected {name!r}")
    text = content[header.end() :].strip()
    if not text:
        raise PromptError(f"{path} has no text")
    return Prompt(name=name, version=int(header["version"]), text=text)
