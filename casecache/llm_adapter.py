from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import os
import re
import time

import requests

from .common import clamp, safe_float
from .schema import infer_regime, summarize_window


def parse_policy_response(text: str, *, baseline_action: float, residual_scale: float, max_allocation: float) -> Dict[str, Any]:
    candidate: Dict[str, Any] = {}
    try:
        candidate = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", str(text), flags=re.S)
        if match:
            try:
                candidate = json.loads(match.group(0))
            except Exception:
                candidate = {}
    raw_action = clamp(safe_float(candidate.get("raw_action", 0.0)), -1.0, 1.0)
    allocation = clamp(baseline_action + raw_action * residual_scale, 0.0, max_allocation)
    confidence = clamp(safe_float(candidate.get("confidence", 0.0)), 0.0, 1.0)
    reason_tags = candidate.get("reason_tags", [])
    used_case_ids = candidate.get("used_case_ids", [])
    if not isinstance(reason_tags, list):
        reason_tags = [str(reason_tags)]
    if not isinstance(used_case_ids, list):
        used_case_ids = [str(used_case_ids)]
    return {
        "raw_action": raw_action,
        "allocation": allocation,
        "confidence": confidence,
        "regime": str(candidate.get("regime", "unknown")),
        "reason_tags": [str(x) for x in reason_tags][:4],
        "used_case_ids": [str(x) for x in used_case_ids][:8],
    }


def _normalize_url(base_url: str) -> str:
    url = str(base_url).strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return url


@dataclass
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model: str
    timeout: int = 60
    force_json_mode: bool = False
    max_retries: int = 1
    retry_backoff: float = 1.0

    def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.1, max_tokens: int = 256) -> Tuple[str, Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("API key is empty. Please provide a runtime API key.")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if self.force_json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}" if not self.api_key.lower().startswith("bearer ") else self.api_key, "Content-Type": "application/json"}
        last_exc: Optional[Exception] = None
        for attempt in range(int(self.max_retries) + 1):
            try:
                response = requests.post(_normalize_url(self.base_url), headers=headers, json=payload, timeout=int(self.timeout))
                response.raise_for_status()
                body = response.json()
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    content = "".join(str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in content)
                return str(content), body
            except Exception as exc:
                last_exc = exc
                if attempt < int(self.max_retries):
                    time.sleep(float(self.retry_backoff) * (2**attempt))
        raise RuntimeError(f"LLM request failed: {last_exc}")


SYSTEM_PROMPT = """你是一个面向应急智能服务资源缓存预测的策略助手。你需要基于最近轨迹、当前状态和历史相似案例，为当前 step 输出合法的缓存控制动作。只能输出 JSON，不要输出解释、前缀或 Markdown。"""


def _format_window(window: Sequence[Dict[str, Any]], max_rows: int = 6) -> str:
    if not window:
        return "无最近轨迹。"
    rows = []
    for row in list(window)[-max_rows:]:
        s = row.get("state_summary", {})
        rows.append(
            f"t={row.get('t', 0)} arrivals={safe_float(s.get('current_arrivals',0)):.2f} "
            f"waiting={safe_float(s.get('waiting_queue',0)):.2f} resource={safe_float(s.get('resource_queue',0)):.2f} "
            f"allocation={safe_float(row.get('allocation',0)):.2f} reward={safe_float(row.get('reward',0)):.4f}"
        )
    return "\n".join(rows)


def _format_cases(cases: Sequence[Dict[str, Any]], max_cases: int = 5) -> str:
    if not cases:
        return "无历史案例。"
    rows = []
    for c in list(cases)[:max_cases]:
        rows.append(
            f"case_id={c.get('case_id')} score={safe_float(c.get('score',0)):.4f} regime={c.get('regime_tag')} "
            f"game={c.get('game_name')} quality={safe_float(c.get('quality_score',0)):.4f} "
            f"teacher_raw={safe_float(c.get('teacher_raw_action',0)):.4f} allocation={safe_float(c.get('teacher_allocation',0)):.2f} "
            f"future_reward_mean={safe_float(c.get('future_outcome',{}).get('reward_mean',0)):.4f}"
        )
    return "\n".join(rows)


def make_policy_prompt(
    *,
    game_name: str,
    seen_status: str,
    recent_window: Sequence[Dict[str, Any]],
    current_state_summary: Dict[str, Any],
    retrieved_cases: Sequence[Dict[str, Any]],
    residual_scale: float,
    max_allocation: float,
) -> List[Dict[str, str]]:
    trend = summarize_window(recent_window)
    regime = infer_regime(current_state_summary, trend)
    s = current_state_summary
    user = f"""
任务：预测当前 step 的资源缓存残差动作 raw_action。
当前业务：game_name={game_name}, seen_status={seen_status}
动作约束：raw_action ∈ [-1,1]；allocation = clip(baseline_action + raw_action * residual_scale, 0, max_allocation)。
residual_scale={float(residual_scale):.2f}, max_allocation={float(max_allocation):.2f}

当前状态：
resource_queue={safe_float(s.get('resource_queue',0)):.2f}
waiting_queue={safe_float(s.get('waiting_queue',0)):.2f}
current_arrivals={safe_float(s.get('current_arrivals',0)):.2f}
backlog_pressure={safe_float(s.get('backlog_pressure',0)):.4f}
baseline_action={safe_float(s.get('baseline_action',0)):.2f}
inferred_regime={regime}

最近轨迹：
{_format_window(recent_window)}

Top-K历史案例：
{_format_cases(retrieved_cases)}

请输出如下 JSON：
{{"raw_action": 0.0, "allocation": 0.0, "confidence": 0.0, "regime": "stable", "reason_tags": ["..."], "used_case_ids": ["..."]}}
""".strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
