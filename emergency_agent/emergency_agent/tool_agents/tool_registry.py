from typing import Any, Callable, Dict, List

from tool_agents.image_analysis_tool import image_analysis
from tool_agents.report_making_tool import report_making
from tool_agents.segmentation_tool import RescueNetSegmentationTool, build_default_tool
from tool_agents.thinking_and_plan_tool import thinking_and_plan
from tool_agents.tool_ending_tool import tool_ending


class ToolRegistry:
    """Simple in-memory registry for available tools."""

    def __init__(self) -> None:
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string")
        if not callable(builder):
            raise ValueError("Tool builder must be callable")
        self._builders[name] = builder

    def create(self, name: str, **kwargs: Any) -> Any:
        if name not in self._builders:
            raise KeyError(f"Tool not registered: {name}")
        return self._builders[name](**kwargs)

    def list_tools(self) -> List[str]:
        return sorted(self._builders.keys())

    def is_registered(self, name: str) -> bool:
        return name in self._builders


registry = ToolRegistry()

# Register available tools here.
registry.register("rescuenet_segmentation", build_default_tool)
registry.register("thinking_and_plan", thinking_and_plan)
# Backward-compatible alias.
registry.register("tthinking_and_plan", thinking_and_plan)
registry.register("image_analysis", image_analysis)
registry.register("report_making", report_making)
registry.register("tool_ending", tool_ending)


def get_tool(name: str, **kwargs: Any) -> Any:
    """Factory helper to create a registered tool by name."""
    return registry.create(name, **kwargs)


def get_default_rescuenet_tool(**kwargs: Any) -> RescueNetSegmentationTool:
    """Typed convenience accessor for the RescueNet segmentation tool."""
    return registry.create("rescuenet_segmentation", **kwargs)
