import json
from typing import Any, Dict, List, Optional

from tool_agents.system_prompt import SYSTEM_PROMPT


def thinking_and_plan(
    user_query: str,
    available_tools: List[str],
    allow_tools: bool = True,
    tool_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    history_text = json.dumps(tool_history or [], ensure_ascii=False)
    planner_prompt = (
        "你是工具调度器。请严格输出一个 JSON 对象，不要输出 Markdown。\n"
        f"是否允许工具调用: {allow_tools}\n"
        f"可用工具: {available_tools}\n"
        "工具说明:\n"
        "1) rescuenet_segmentation: 对图像执行语义分割，输出原图、分割叠加图(overlay)和类别统计，不输出保存路径。\n"
        "2) image_analysis: 支持两种模式。模式A(有分割): 基于原图+overlay+类别统计分析；模式B(无分割): 仅基于原图分析。输出中需体现是否使用分割工具。\n"
        "3) report_making: 基于已有工具结果生成完整分析报告，必须纳入 image_analysis 的详细分析内容。仅当用户明确要求\"生成报告/写报告/总结\"时才调用。\n"
        "4) tool_ending: 结束工具阶段，把最终报告覆盖写入 /root/autodl-tmp/report.txt，并把工具输出打包给 LLM 生成最终答复。\n\n"
        "历史工具轨迹(JSON):\n"
        f"{history_text}\n\n"
        "可选动作:\n"
        "- 直接回答: {\"action\":\"respond\",\"response\":\"...\"}\n"
        "- 调用工具: {\"action\":\"call_tool\",\"tool_name\":\"rescuenet_segmentation|image_analysis|report_making|tool_ending\",\"tool_args\":{...},\"reason\":\"...\"}\n"
        "  说明: image_analysis 可传 tool_args.image_paths(字符串列表) 进行无分割分析。\n\n"
        "每次任务开始时，都必须先使用 thinking_and_plan 进行规划，再决定后续工具调用。\n"
        "当用户要求图像分割时，先调用 rescuenet_segmentation。\n"
        "当已经有分割结果但还没有图像分析时，下一步优先调用 image_analysis，不要重复调用 rescuenet_segmentation。\n"
        "当且仅当用户明确要求报告/总结时，且已有 image_analysis 结果，才调用 report_making。\n"
        "当任务完成时，调用 tool_ending。\n\n"
        f"用户请求: {user_query}"
    )

    return {
        "tool_name": "thinking_and_plan",
        "system_prompt": SYSTEM_PROMPT,
        "planner_prompt": planner_prompt,
    }
