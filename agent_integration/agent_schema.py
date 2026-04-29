from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class AgentToolEvent:
    """A lightweight, model-free representation of one agent tool call."""

    step: int
    tool_name: str
    display_name: str
    task_type: str
    resource_weight: float
    status: str
    summary: str
    output_type: str
    estimated_latency_level: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
