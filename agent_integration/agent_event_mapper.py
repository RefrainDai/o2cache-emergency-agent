from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

import pandas as pd

from .agent_schema import AgentToolEvent


TOOL_TASK_MAPPING: Dict[str, Dict[str, Any]] = {
    "vqa": {"task_type": "快速问答业务", "resource_weight": 1.0, "display_name": "VQA快速问答"},
    "VQA快速问答": {"task_type": "快速问答业务", "resource_weight": 1.0, "display_name": "VQA快速问答"},
    "thinking_and_plan": {"task_type": "任务规划业务", "resource_weight": 0.5, "display_name": "任务规划"},
    "rescuenet_segmentation": {"task_type": "语义分割业务", "resource_weight": 3.0, "display_name": "灾害语义分割"},
    "image_analysis": {"task_type": "灾害图像深度分析业务", "resource_weight": 4.0, "display_name": "灾害图像深度分析"},
    "report_making": {"task_type": "结构化报告生成业务", "resource_weight": 2.0, "display_name": "结构化报告生成"},
    "tool_ending": {"task_type": "结果回传业务", "resource_weight": 0.5, "display_name": "结果整理与回传"},
}


def tool_history_to_events(tool_history: Iterable[Dict[str, Any]]) -> List[AgentToolEvent]:
    events: List[AgentToolEvent] = []
    for idx, item in enumerate(tool_history, start=1):
        tool_name = str(item.get("tool_name", "unknown"))
        mapping = TOOL_TASK_MAPPING.get(tool_name, {})
        event = AgentToolEvent(
            step=int(item.get("step", idx)),
            tool_name=tool_name,
            display_name=str(item.get("display_name") or mapping.get("display_name") or tool_name),
            task_type=str(item.get("task_type") or mapping.get("task_type") or "应急智能服务"),
            resource_weight=float(item.get("resource_weight", mapping.get("resource_weight", 1.0))),
            status=str(item.get("status", "已完成")),
            summary=str(item.get("summary", "")),
            output_type=str(item.get("output_type", "structured_payload")),
            estimated_latency_level=str(item.get("estimated_latency_level", "中")),
        )
        events.append(event)
    return events


def events_to_load_dataframe(
    tool_history: Iterable[Dict[str, Any]] | Iterable[AgentToolEvent],
    *,
    sequence_id: str = "智能体演示序列001",
    start_time: str = "2026-04-01 08:00:00",
    interval_minutes: int = 5,
    points_per_tool: int = 6,
) -> pd.DataFrame:
    """Convert agent tool events into a demo service-load DataFrame.

    The generated arrivals are demonstration loads derived from resource
    weights and tool-call counts. They are not real UAV telemetry.
    """

    raw_items = list(tool_history)
    if not raw_items:
        return pd.DataFrame(columns=["timestamp", "game_name", "sequence_id", "arrivals", "resource_queue", "waiting_queue"])
    if isinstance(raw_items[0], AgentToolEvent):
        events = raw_items  # type: ignore[assignment]
    else:
        events = tool_history_to_events(raw_items)  # type: ignore[arg-type]

    base = datetime.fromisoformat(start_time)
    rows: List[Dict[str, Any]] = []
    resource_queue = 12.0
    waiting_queue = 0.0
    call_counts: Dict[str, int] = {}
    row_index = 0
    for event_index, event in enumerate(events):
        call_counts[event.tool_name] = call_counts.get(event.tool_name, 0) + 1
        for local_step in range(max(int(points_per_tool), 3)):
            burst = 1.0 + 0.18 * local_step + 0.08 * event_index
            arrivals = max(1.0, round(float(event.resource_weight) * 2.0 * burst * call_counts[event.tool_name], 2))
            waiting_queue = max(0.0, round(waiting_queue + arrivals - resource_queue * 0.45, 2))
            baseline_action = max(0.0, round(waiting_queue + arrivals - resource_queue, 2))
            rows.append(
                {
                    "timestamp": base + timedelta(minutes=interval_minutes * row_index),
                    "game_name": event.task_type,
                    "sequence_id": sequence_id,
                    "arrivals": arrivals,
                    "resource_queue": round(resource_queue, 2),
                    "waiting_queue": waiting_queue,
                    "baseline_action": baseline_action,
                    "tool_name": event.tool_name,
                    "display_name": event.display_name,
                    "resource_weight": event.resource_weight,
                    "agent_step": event.step,
                }
            )
            resource_queue = max(4.0, round(resource_queue - arrivals * 0.2 + 1.5, 2))
            row_index += 1
    return pd.DataFrame(rows)


def chinese_event_frame(tool_history: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    rows = [event.to_dict() for event in tool_history_to_events(tool_history)]
    df = pd.DataFrame(rows)
    return df.rename(
        columns={
            "step": "步骤",
            "tool_name": "工具标识",
            "display_name": "工具名称",
            "task_type": "对应任务类型",
            "resource_weight": "资源需求权重",
            "status": "状态",
            "summary": "作用说明",
            "output_type": "输出类型",
            "estimated_latency_level": "估计时延等级",
        }
    )


def chinese_load_frame(load_df: pd.DataFrame) -> pd.DataFrame:
    return load_df.rename(
        columns={
            "timestamp": "时间步",
            "game_name": "任务类型",
            "sequence_id": "请求序列",
            "arrivals": "请求到达量",
            "resource_queue": "空闲算力资源",
            "waiting_queue": "等待服务队列",
            "baseline_action": "即时需求缺口",
            "tool_name": "工具标识",
            "display_name": "工具名称",
            "resource_weight": "资源需求权重",
            "agent_step": "智能体步骤",
        }
    )
