import os
import json
from typing import Any, Dict, List


DEFAULT_REPORT_PATH = "/root/autodl-tmp/report.txt"


def _compact_value(value: Any) -> Any:
    if isinstance(value, dict):
        compact: Dict[str, Any] = {}
        for key, item in value.items():
            if key.endswith("_base64"):
                compact[key] = f"<omitted base64, length={len(item)}>"
            else:
                compact[key] = _compact_value(item)
        return compact
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    if isinstance(value, str) and len(value) > 1000:
        return f"<omitted long text, length={len(value)}>"
    return value


def tool_ending(
    user_query: str,
    tool_history: List[Dict[str, Any]],
    report: str = "",
) -> Dict[str, Any]:
    final_report = report
    if not final_report:
        for item in reversed(tool_history):
            if item.get("tool_name") == "report_making":
                final_report = item.get("result", {}).get("report", "")
                if final_report:
                    break

    report_path = DEFAULT_REPORT_PATH
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(final_report)

    payload = {
        "user_query": user_query,
        "tool_history": _compact_value(tool_history),
        "report": final_report,
        "report_path": report_path,
        "status": "completed",
    }
    return {
        "tool_name": "tool_ending",
        "finished": True,
        "report_path": report_path,
        "report_written": True,
        "llm_handoff": json.dumps(payload, ensure_ascii=False),
    }
