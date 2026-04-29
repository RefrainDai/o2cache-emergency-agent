from typing import Any, Dict, List, Optional


BASE_ANALYSIS_PROMPT_WITH_SEG = (
    "你将看到与灾害场景相关的图像。"
    "请结合原图与分割叠加图（overlay）进行分析：原图用于识别真实视觉细节，叠加图用于辅助定位语义类别分布。"
    "请依据颜色图例、地物覆盖比例和视觉线索，对灾害现象进行详细、精确、审慎的分析，并输出完整中文细节。\n"
    "segmentation categories (for reference only):\n"
    "- Background: (0, 0, 0) (black)\n"
    "- Water: (61, 230, 250) (cyan)\n"
    "- Building-No-Damage: (180, 120, 120) (brown)\n"
    "- Building-Medium-Damage: (235, 255, 7) (bright yellow)\n"
    "- Building-Major-Damage: (255, 184, 6) (orange-yellow)\n"
    "- Building-Total-Destruction: (255, 0, 0) (red)\n"
    "- Vehicle: (255, 0, 245) (magenta)\n"
    "- Road-Clear: (140, 140, 140) (gray)\n"
    "- Road-Blocked: (160, 150, 20) (olive yellow)\n"
    "- Tree: (4, 250, 7) (bright green)\n"
    "- Pool: (255, 235, 0) (light yellow)\n"
    "输出要求：明确区分原图直接观察结论与分割图支持的判断，给出尽可能具体的灾害现象、受损区域、道路状态、植被/水体/建筑情况。"
)


BASE_ANALYSIS_PROMPT_NO_SEG = (
    "你将看到与灾害场景相关的原图。"
    "当前没有分割结果，仅可基于原图视觉线索进行分析。"
    "请详细描述可见的灾害现象、受损区域、道路状态、植被/水体/建筑情况，"
    "并在结论中明确说明哪些判断是高置信观察、哪些属于低置信推断。"
)


def image_analysis(
    user_query: str,
    tool_history: List[Dict[str, Any]],
    target_step: Optional[int] = None,
    image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    segmentation_steps = [
        item for item in tool_history if item.get("tool_name") == "rescuenet_segmentation"
    ]
    has_segmentation = bool(segmentation_steps)

    if not has_segmentation and not image_paths:
        raise ValueError(
            "image_analysis requires either previous rescuenet_segmentation result or tool_args.image_paths"
        )

    if not has_segmentation:
        items = [{"image_path": path} for path in image_paths or []]
        return {
            "tool_name": "image_analysis",
            "target_step": None,
            "used_segmentation_tool": False,
            "analysis_prompt": BASE_ANALYSIS_PROMPT_NO_SEG,
            "items": items,
        }

    target_item: Optional[Dict[str, Any]] = None
    if target_step is not None:
        for item in segmentation_steps:
            if item.get("step") == target_step:
                target_item = item
                break
    if target_item is None:
        target_item = segmentation_steps[-1]

    seg_result = target_item.get("result", {})
    image_items = seg_result.get("items", [])
    if not image_items:
        raise ValueError("No segmentation image payload found for image_analysis")

    return {
        "tool_name": "image_analysis",
        "target_step": target_item.get("step"),
        "used_segmentation_tool": True,
        "analysis_prompt": BASE_ANALYSIS_PROMPT_WITH_SEG,
        "items": image_items,
    }
