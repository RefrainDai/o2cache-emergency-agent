import json
from typing import Any, Dict, List


def _aggregate_class_stats(analysis_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    aggregate: Dict[str, Dict[str, float]] = {}
    for item in analysis_items:
        result = item.get("result", {})
        for analysis in result.get("analyses", []):
            stats = analysis.get("class_statistics", [])
            if not isinstance(stats, list):
                continue
            for stat in stats:
                label = stat.get("label")
                if not isinstance(label, str) or not label:
                    continue
                pixel_count = float(stat.get("pixel_count", 0) or 0)
                ratio = float(stat.get("ratio", 0) or 0)
                entry = aggregate.setdefault(label, {"pixel_count": 0.0, "ratio_sum": 0.0, "frames": 0.0})
                entry["pixel_count"] += pixel_count
                entry["ratio_sum"] += ratio
                entry["frames"] += 1.0
    return aggregate


def _avg_ratio(aggregate: Dict[str, Dict[str, float]], label: str) -> float:
    entry = aggregate.get(label)
    if not entry:
        return 0.0
    frames = entry.get("frames", 0.0)
    if frames <= 0:
        return 0.0
    return entry.get("ratio_sum", 0.0) / frames


def _severity_from_ratios(major_ratio: float, medium_ratio: float, total_ratio: float) -> str:
    if total_ratio >= 0.15 or major_ratio >= 0.08:
        return "Major"
    if total_ratio >= 0.05 or medium_ratio >= 0.04:
        return "Medium"
    return "Superficial"


def _building_density(no_damage_ratio: float, medium_ratio: float, major_ratio: float, total_ratio: float) -> str:
    building_ratio = no_damage_ratio + medium_ratio + major_ratio + total_ratio
    if building_ratio >= 0.30:
        return "高密度"
    if building_ratio >= 0.15:
        return "中密度"
    return "低密度"


def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in result.items():
        if key == "items" and isinstance(value, list):
            compact["items"] = [
                {
                    "image_path": item.get("image_path"),
                    "class_statistics": item.get("class_statistics", []),
                    "used_segmentation_tool": result.get("used_segmentation_tool"),
                }
                for item in value
            ]
            continue
        if isinstance(value, str) and len(value) > 500:
            compact[key] = f"<omitted long text, length={len(value)}>"
            continue
        compact[key] = value
    return compact


def report_making(
    user_query: str,
    tool_history: List[Dict[str, Any]],
    extra_notes: str = "",
) -> Dict[str, Any]:
    analysis_items = [
        item for item in tool_history if item.get("tool_name") == "image_analysis"
    ]

    lines = ["灾害分析报告", f"用户任务: {user_query}"]

    aggregate = _aggregate_class_stats(analysis_items)
    water_ratio = _avg_ratio(aggregate, "water")
    tree_ratio = _avg_ratio(aggregate, "tree")
    pool_ratio = _avg_ratio(aggregate, "pool")
    vehicle_ratio = _avg_ratio(aggregate, "vehicle")
    road_clear_ratio = _avg_ratio(aggregate, "road-clear")
    road_blocked_ratio = _avg_ratio(aggregate, "road-blocked")
    b_no_damage = _avg_ratio(aggregate, "building-no-damage")
    b_medium = _avg_ratio(aggregate, "building-medium-damage")
    b_major = _avg_ratio(aggregate, "building-major-damage")
    b_total = _avg_ratio(aggregate, "building-total-destruction")

    damage_overall = _severity_from_ratios(b_major, b_medium, b_total)
    density = _building_density(b_no_damage, b_medium, b_major, b_total)

    # 统计类“数量”采用像素统计值，并在文本中明确其含义。
    water_px = int(aggregate.get("water", {}).get("pixel_count", 0))
    tree_px = int(aggregate.get("tree", {}).get("pixel_count", 0))
    pool_px = int(aggregate.get("pool", {}).get("pixel_count", 0))
    vehicle_px = int(aggregate.get("vehicle", {}).get("pixel_count", 0))
    road_px = int(aggregate.get("road-clear", {}).get("pixel_count", 0) + aggregate.get("road-blocked", {}).get("pixel_count", 0))
    building_px = int(
        aggregate.get("building-no-damage", {}).get("pixel_count", 0)
        + aggregate.get("building-medium-damage", {}).get("pixel_count", 0)
        + aggregate.get("building-major-damage", {}).get("pixel_count", 0)
        + aggregate.get("building-total-destruction", {}).get("pixel_count", 0)
    )

    lines.append("1.场景理解：")
    if not analysis_items:
        lines.append("未检测到可用的 image_analysis 输出，暂无法形成场景理解。")
    else:
        lines.append(
            "图片包含水/树/泳池/汽车/建筑物/道路等地物；以下数量基于语义分割像素统计（非实例计数）："
        )
        lines.append(
            f"水体={water_px}, 树木={tree_px}, 泳池={pool_px}, 汽车={vehicle_px}, 建筑物={building_px}, 道路={road_px}。"
        )
        lines.append(
            f"建筑物密度判断为{density}（建筑区域平均占比约{(b_no_damage + b_medium + b_major + b_total):.2%}）。"
        )

    lines.append("2.受灾状况评估：")
    lines.append(f"a.整体受损等级：{damage_overall}")
    lines.append("b.详细受损情况：")
    lines.append(
        "▪建筑物受损情况："
        f"未受损占比约{b_no_damage:.2%}；"
        f"轻微/中度受损占比约{b_medium:.2%}；"
        f"严重受损占比约{b_major:.2%}；"
        f"全部毁坏占比约{b_total:.2%}。"
    )
    max_damage = max(
        [("未受损", b_no_damage), ("轻微/中度受损", b_medium), ("严重受损", b_major), ("全部毁坏", b_total)],
        key=lambda x: x[1],
    )
    min_damage = min(
        [("未受损", b_no_damage), ("轻微/中度受损", b_medium), ("严重受损", b_major), ("全部毁坏", b_total)],
        key=lambda x: x[1],
    )
    lines.append(f"最大建筑物状态: {max_damage[0]}；最小建筑物状态: {min_damage[0]}。")
    road_state = "道路整体偏通畅" if road_clear_ratio >= road_blocked_ratio else "道路受阻较明显"
    debris_hint = "Covered in Debris 风险偏高" if road_blocked_ratio >= 0.03 else "Covered in Debris 风险暂不突出"
    flood_hint = "Flooded 风险偏高" if water_ratio >= 0.05 else "Flooded 风险暂不突出"
    lines.append(
        "▪道路受损情况："
        f"road-clear={road_clear_ratio:.2%}, road-blocked={road_blocked_ratio:.2%}，{road_state}；{debris_hint}；{flood_hint}。"
    )

    lines.append("3.评估风险并制定应急方案：")
    need_urgent = damage_overall in {"Medium", "Major"} or road_blocked_ratio >= 0.03
    lines.append(f"a.是否需要进行紧急恢复：{'是' if need_urgent else '否（建议持续监测）'}")
    lines.append("b.资源与路线应急方案：")
    lines.append("- 先开展主干道路障清理与通行恢复，优先打通 road-blocked 高占比区域。")
    lines.append("- 对严重受损与全部毁坏建筑集中区进行搜救与结构安全排查。")
    lines.append("- 在可能积水区域布置排涝与警戒带，必要时调整救援路线绕行 flooded 风险段。")
    lines.append("- 资源配置建议：道路抢通组、建筑搜救组、医疗转运组、无人机复勘组并行协作。")

    lines.append("附：工具执行摘要")
    if not tool_history:
        lines.append("- 暂无工具执行记录。")
    else:
        for idx, item in enumerate(tool_history, start=1):
            name = item.get("tool_name", "unknown")
            result = item.get("result", {})
            lines.append(f"- 步骤{idx}: 工具 `{name}`")
            lines.append(f"  结果: {json.dumps(_compact_result(result), ensure_ascii=False)}")

    if extra_notes:
        lines.append(f"- 补充说明: {extra_notes}")

    return {
        "tool_name": "report_making",
        "report": "\n".join(lines),
    }
