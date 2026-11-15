"""Error message contract: what failed, which parameter caused it, what to do next.

Every failure raised by the prototype carries the literal next command so the
console can show an actionable panel instead of a traceback (DFX-O8).
"""
from __future__ import annotations

DEFAULT_NEXT = "python -m codes.run_pipeline"
SAMPLE_NEXT = "python -m codes.sample_data"


class PrototypeError(Exception):
    """Base class for every diagnosable prototype failure."""

    stage = "general"

    def render(self) -> str:
        return str(self)


class ConfigError(PrototypeError):
    stage = "config"

    def __init__(self, key: str, rule: str, observed=None, expected=None,
                 next_command: str = DEFAULT_NEXT):
        self.key = key
        self.rule = rule
        self.observed = observed
        self.expected = expected
        self.next_command = next_command
        super().__init__(self.render())

    def render(self) -> str:
        parts = [f"configuration '{self.key}' failed: {self.rule}"]
        if self.observed is not None:
            parts.append(f"observed={self.observed!r}")
        if self.expected is not None:
            parts.append(f"expected={self.expected!r}")
        parts.append(f"next: {self.next_command}")
        return " | ".join(parts)


class ContractError(PrototypeError):
    stage = "ingest"

    def __init__(self, file: str, rule: str, missing_fields=None,
                 expected=None, next_command: str = SAMPLE_NEXT):
        self.file = file
        self.rule = rule
        self.missing_fields = list(missing_fields or [])
        self.expected = expected
        self.next_command = next_command
        super().__init__(self.render())

    def render(self) -> str:
        parts = [f"input file '{self.file}' failed the field contract: {self.rule}"]
        if self.missing_fields:
            parts.append(f"missing/required: {self.missing_fields}")
        if self.expected:
            parts.append(f"expected: {self.expected}")
        parts.append(f"next: {self.next_command}")
        return " | ".join(parts)


class StageError(PrototypeError):
    stage = "stage"

    def __init__(self, stage: str, message: str, next_command: str = DEFAULT_NEXT):
        self.stage = stage
        self.message = message
        self.next_command = next_command
        super().__init__(self.render())

    def render(self) -> str:
        return f"stage '{self.stage}' failed: {self.message} | next: {self.next_command}"


class RegistryError(PrototypeError):
    stage = "registry"

    def __init__(self, name: str, available, next_command: str = DEFAULT_NEXT):
        self.name = name
        self.available = sorted(available)
        self.next_command = next_command
        super().__init__(self.render())

    def render(self) -> str:
        return (f"unknown model: {self.name}. available: {self.available} "
                f"| next: {self.next_command}")