from __future__ import annotations

from typing import Any, Dict, List, Optional

from .agent_schema import AgentToolEvent


DEFAULT_USER_TASK = "请分析灾害图像并生成应急报告"


TOOL_EVENT_TEMPLATES = [
    {
        "tool_name": "thinking_and_plan",
        "display_name": "任务规划",
        "task_type": "任务规划业务",
        "resource_weight": 0.5,
        "summary": "解析用户意图，规划语义分割、图像分析、报告生成与结果回传流程。",
        "output_type": "plan",
        "estimated_latency_level": "低",
    },
    {
        "tool_name": "rescuenet_segmentation",
        "display_name": "灾害语义分割",
        "task_type": "语义分割业务",
        "resource_weight": 3.0,
        "summary": "对灾害图像进行语义分割，识别道路、水体、建筑受损等区域。",
        "output_type": "segmentation_mask",
        "estimated_latency_level": "高",
    },
    {
        "tool_name": "image_analysis",
        "display_name": "灾害图像深度分析",
        "task_type": "灾害图像深度分析业务",
        "resource_weight": 4.0,
        "summary": "结合图像线索和分割结果，分析受灾区域、道路通行状态与救援风险。",
        "output_type": "analysis_text",
        "estimated_latency_level": "高",
    },
    {
        "tool_name": "report_making",
        "display_name": "结构化报告生成",
        "task_type": "结构化报告生成业务",
        "resource_weight": 2.0,
        "summary": "汇总图像分析结论，形成应急处置建议和结构化灾情报告。",
        "output_type": "structured_report",
        "estimated_latency_level": "中",
    },
    {
        "tool_name": "tool_ending",
        "display_name": "结果整理与回传",
        "task_type": "结果回传业务",
        "resource_weight": 0.5,
        "summary": "整理工具调用结果，将灾害分析摘要回传给上层应急服务流程。",
        "output_type": "handoff_payload",
        "estimated_latency_level": "低",
    },
]


def run_demo_agent(user_task: str = DEFAULT_USER_TASK, image_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return a deterministic demo tool history without loading any model."""

    image_label = image_name or "演示灾害图像"
    history: List[Dict[str, Any]] = []
    for step, item in enumerate(TOOL_EVENT_TEMPLATES, start=1):
        event = AgentToolEvent(step=step, status="已完成", **item)
        payload = event.to_dict()
        payload["user_task"] = user_task or DEFAULT_USER_TASK
        payload["image_name"] = image_label
        payload["result"] = _mock_result_for_tool(payload)
        history.append(payload)
    return history


def _mock_result_for_tool(event: Dict[str, Any]) -> Dict[str, Any]:
    tool_name = event.get("tool_name")
    if tool_name == "thinking_and_plan":
        return {
            "plan": [
                "先理解用户任务与输入图像",
                "执行灾害场景语义分割",
                "进行灾害图像深度分析",
                "生成结构化应急报告",
                "整理结果并回传",
            ]
        }
    if tool_name == "rescuenet_segmentation":
        return {
            "detected_regions": ["疑似受损建筑", "局部阻断道路", "水体或积水区域"],
            "note": "演示模式仅给出结构化摘要，不执行真实分割推理。",
        }
    if tool_name == "image_analysis":
        return {
            "risk_summary": "图像中存在建筑受损、道路通行受阻和局部积水风险，建议优先核查交通通道与人员聚集区域。",
            "confidence": "演示级",
        }
    if tool_name == "report_making":
        return {
            "report_sections": ["场景理解", "受灾状态评估", "应急资源建议"],
            "summary": "形成面向应急中心的结构化灾情分析报告。",
        }
    if tool_name == "tool_ending":
        return {
            "handoff": "工具调用轨迹已整理，可转化为多类型智能服务请求负载。",
        }
    return {}
