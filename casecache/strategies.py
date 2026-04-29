from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import time

import numpy as np
import pandas as pd

from .cases import CaseLibrary
from .common import clamp, safe_float
from .data import prepare_request_dataframe
from .llm_adapter import OpenAICompatibleClient, make_policy_prompt, parse_policy_response
from .rewards import action_to_allocation, compute_allocation_outcome, myopic_base_allocation
from .schema import infer_regime, normalize_record, summarize_window


@dataclass
class StrategyConfig:
    strategy: str = "RAG-only"
    window: int = 5
    k: int = 5
    residual_scale: float = 200.0
    max_allocation: float = 800.0
    initial_resource: float = 10.0
    initial_waiting: float = 0.0
    waiting_penalty: float = 3.0
    idle_penalty: float = 1.0
    reward_scale: float = 100.0
    min_confidence: float = 0.15
    llm_every: int = 1


def heuristic_raw_action(current_state: Dict[str, Any], recent_window: Sequence[Dict[str, Any]]) -> float:
    trend = summarize_window(recent_window)
    regime = infer_regime(current_state, trend)
    backlog = safe_float(current_state.get("backlog_pressure", 0.0))
    waiting = safe_float(current_state.get("waiting_queue", 0.0))
    arrivals = safe_float(current_state.get("current_arrivals", 0.0))
    raw = 0.0
    if regime == "surge":
        raw = 0.35 + 0.15 * min(backlog, 2.0)
    elif regime == "backlog_recovery":
        raw = 0.25
    elif regime == "oversupply_risk":
        raw = -0.35
    elif regime == "volatile":
        raw = 0.10 if waiting + arrivals > 0 else -0.05
    else:
        raw = 0.0
    return clamp(raw, -1.0, 1.0)


def _state_from_current(resource: float, waiting: float, arrivals: float, progress: float, game: str, seq: str, seen_status: str, max_allocation: float) -> Dict[str, Any]:
    baseline = myopic_base_allocation(resource, waiting, arrivals, max_allocation=max_allocation)
    return {
        "game_name": str(game),
        "access_key_id": str(seq),
        "seen_status": str(seen_status),
        "resource_queue": float(resource),
        "waiting_queue": float(waiting),
        "current_arrivals": float(arrivals),
        "progress": float(progress),
        "backlog_pressure": float(waiting / (resource + arrivals + 1.0)),
        "baseline_action": float(baseline),
    }


def run_strategy_on_requests(
    df: pd.DataFrame,
    *,
    config: StrategyConfig,
    case_library: Optional[CaseLibrary] = None,
    llm_client: Optional[OpenAICompatibleClient] = None,
    max_steps_per_sequence: Optional[int] = None,
) -> Dict[str, Any]:
    data = prepare_request_dataframe(df)
    rows: List[Dict[str, Any]] = []
    returns: List[float] = []
    use_llm = str(config.strategy).lower() in {"rag+llm", "llm"}
    for (game, seq), g in data.groupby(["game_name", "access_key_id"], sort=False):
        g = g.reset_index(drop=True)
        if max_steps_per_sequence is not None:
            g = g.iloc[: int(max_steps_per_sequence)].copy()
        n = len(g)
        resource = float(config.initial_resource)
        waiting = float(config.initial_waiting)
        recent_rows: List[Dict[str, Any]] = []
        ep_return = 0.0
        last_llm_raw = 0.0
        seen_status = "seen" if case_library and str(game) in case_library.games_in_library else "unseen"
        for t, rec in g.iterrows():
            arrivals = float(rec["num"])
            progress = 0.0 if n <= 1 else float(t) / float(n - 1)
            current = _state_from_current(resource, waiting, arrivals, progress, str(game), str(seq), seen_status, config.max_allocation)
            retrieved_cases: List[Dict[str, Any]] = []
            fallback_raw = 0.0
            default_source = "baseline"
            if case_library is not None and case_library.cases:
                retrieved_cases = case_library.query(game_name=str(game), recent_window=recent_rows[-config.window :], current_state_summary=current, k=config.k)
                fallback_raw = case_library.fallback_raw_action(game_name=str(game), recent_window=recent_rows[-config.window :], current_state_summary=current, k=config.k)
                default_source = "retriever"
            strategy_lower = str(config.strategy).lower()
            llm_result = None
            if strategy_lower == "baseline":
                raw = 0.0
                source = "baseline"
            elif strategy_lower in {"heuristic", "rule-heuristic"}:
                raw = heuristic_raw_action(current, recent_rows[-config.window :])
                source = "heuristic"
            elif strategy_lower in {"rag-only", "retriever"}:
                raw = fallback_raw
                source = default_source
            elif strategy_lower in {"rag+llm", "llm"}:
                raw = fallback_raw
                source = f"llm_fallback:{default_source}"
                if llm_client is not None and (int(t) % max(int(config.llm_every), 1) == 0):
                    try:
                        messages = make_policy_prompt(
                            game_name=str(game),
                            seen_status=seen_status,
                            recent_window=recent_rows[-config.window :],
                            current_state_summary=current,
                            retrieved_cases=retrieved_cases,
                            residual_scale=config.residual_scale,
                            max_allocation=config.max_allocation,
                        )
                        content, raw_body = llm_client.chat(messages)
                        llm_result = parse_policy_response(content, baseline_action=current["baseline_action"], residual_scale=config.residual_scale, max_allocation=config.max_allocation)
                        llm_result["raw_response"] = content
                        llm_result["usage"] = raw_body.get("usage", {}) if isinstance(raw_body, dict) else {}
                        if safe_float(llm_result.get("confidence", 0.0)) >= float(config.min_confidence):
                            raw = safe_float(llm_result.get("raw_action", fallback_raw))
                            source = "llm"
                            last_llm_raw = raw
                        else:
                            source = f"llm_low_conf_fallback:{default_source}"
                    except Exception as exc:
                        llm_result = {"error_type": type(exc).__name__, "error_message": str(exc)}
                        source = f"llm_exception_fallback:{default_source}"
                        raw = fallback_raw
                    last_llm_raw = raw
                elif llm_client is not None and config.llm_every > 1:
                    raw = last_llm_raw
                    source = "llm_hold"
                else:
                    raw = fallback_raw
                    source = f"llm_disabled_fallback:{default_source}"
            else:
                raw = 0.0
                source = "baseline"
            raw = clamp(raw, -1.0, 1.0)
            allocation = action_to_allocation(raw, current["baseline_action"], residual_scale=config.residual_scale, max_allocation=config.max_allocation)
            metrics = compute_allocation_outcome(resource, waiting, arrivals, allocation, waiting_penalty=config.waiting_penalty, idle_penalty=config.idle_penalty, reward_scale=config.reward_scale)
            next_arrivals = float(g.loc[t + 1, "num"]) if int(t) + 1 < n else 0.0
            next_state = _state_from_current(metrics["next_resource_queue"], metrics["next_waiting_queue"], next_arrivals, min(progress + 1.0 / max(n - 1, 1), 1.0), str(game), str(seq), seen_status, config.max_allocation)
            row: Dict[str, Any] = {
                "t": int(t),
                "task": {"goal": str(game), "access_key_id": str(seq), "seen_status": seen_status},
                "game_name": str(game),
                "access_key_id": str(seq),
                "seen_status": seen_status,
                "state_summary": current,
                "next_state_summary": next_state,
                "raw_action": raw,
                "allocation": allocation,
                "reward": metrics["reward"],
                "cache_available": metrics["cache_available"],
                "hit_rate": metrics["hit_rate"],
                "waste_rate": metrics["waste_rate"],
                "waiting_cost": metrics["waiting_cost"],
                "idle_cost": metrics["idle_cost"],
                "action_source": source,
                "retrieved_cases": [{k: v for k, v in c.items() if k not in {"feature", "recent_window"}} for c in retrieved_cases],
                "retrieved_case_ids": [str(c.get("case_id")) for c in retrieved_cases],
            }
            if llm_result is not None:
                row["llm_result"] = llm_result
            rows.append(row)
            recent_rows.append(row)
            ep_return += metrics["reward"]
            resource = metrics["next_resource_queue"]
            waiting = metrics["next_waiting_queue"]
        returns.append(ep_return)
    summary = {
        "strategy": config.strategy,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "num_tasks": int(len(returns)),
        "num_rows": int(len(rows)),
        "use_llm": bool(use_llm),
        "window": int(config.window),
        "k": int(config.k),
        "waiting_penalty": float(config.waiting_penalty),
        "idle_penalty": float(config.idle_penalty),
        "reward_scale": float(config.reward_scale),
        "mean_reward": float(np.mean([safe_float(r.get("reward", 0.0)) for r in rows])) if rows else 0.0,
        "mean_hit_rate": float(np.mean([safe_float(r.get("hit_rate", 0.0)) for r in rows])) if rows else 0.0,
        "mean_waste_rate": float(np.mean([safe_float(r.get("waste_rate", 0.0)) for r in rows])) if rows else 0.0,
        "mean_allocation": float(np.mean([safe_float(r.get("allocation", 0.0)) for r in rows])) if rows else 0.0,
    }
    return {"summary": summary, "trace": rows, "returns": returns}


def trace_to_frame(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for r in rows:
        s = r.get("state_summary", {})
        ns = r.get("next_state_summary", {})
        out.append({
            "t": r.get("t"),
            "game_name": r.get("game_name"),
            "access_key_id": r.get("access_key_id"),
            "current_arrivals": safe_float(s.get("current_arrivals", 0.0)),
            "resource_queue": safe_float(s.get("resource_queue", 0.0)),
            "waiting_queue": safe_float(s.get("waiting_queue", 0.0)),
            "baseline_action": safe_float(s.get("baseline_action", 0.0)),
            "raw_action": safe_float(r.get("raw_action", 0.0)),
            "allocation": safe_float(r.get("allocation", 0.0)),
            "reward": safe_float(r.get("reward", 0.0)),
            "hit_rate": safe_float(r.get("hit_rate", 0.0)),
            "waste_rate": safe_float(r.get("waste_rate", 0.0)),
            "next_resource_queue": safe_float(ns.get("resource_queue", 0.0)),
            "next_waiting_queue": safe_float(ns.get("waiting_queue", 0.0)),
            "action_source": r.get("action_source", ""),
            "retrieved_case_ids": ",".join(r.get("retrieved_case_ids", [])) if isinstance(r.get("retrieved_case_ids"), list) else "",
        })
    return pd.DataFrame(out)
