from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np

from .common import clamp, mean, safe_float, safe_int, slope, std
from .rewards import myopic_base_allocation, raw_action_from_allocation


def _legacy_state_summary(record: Dict[str, Any]) -> Dict[str, float]:
    state = record.get("state", [0.0, 0.0])
    resource_queue = safe_float(state[0] if len(state) > 0 else 0.0)
    waiting_queue = safe_float(state[1] if len(state) > 1 else 0.0)
    arrivals = safe_float(record.get("arrivals", record.get("current_arrivals", 0.0)))
    baseline_action = myopic_base_allocation(resource_queue, waiting_queue, arrivals)
    backlog_pressure = waiting_queue / (resource_queue + arrivals + 1.0)
    return {
        "resource_queue": resource_queue,
        "waiting_queue": waiting_queue,
        "current_arrivals": arrivals,
        "progress": safe_float(record.get("progress", 0.0)),
        "backlog_pressure": backlog_pressure,
        "baseline_action": baseline_action,
    }


def compact_state_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    s = dict(summary or {})
    resource_queue = safe_float(s.get("resource_queue", 0.0))
    waiting_queue = safe_float(s.get("waiting_queue", 0.0))
    arrivals = safe_float(s.get("current_arrivals", s.get("arrivals", 0.0)))
    baseline = safe_float(s.get("baseline_action", myopic_base_allocation(resource_queue, waiting_queue, arrivals)))
    backlog_pressure = safe_float(s.get("backlog_pressure", waiting_queue / (resource_queue + arrivals + 1.0)))
    return {
        "game_name": str(s.get("game_name", "")),
        "access_key_id": str(s.get("access_key_id", "")),
        "seen_status": str(s.get("seen_status", "unknown")),
        "resource_queue": resource_queue,
        "waiting_queue": waiting_queue,
        "current_arrivals": arrivals,
        "progress": safe_float(s.get("progress", 0.0)),
        "backlog_pressure": backlog_pressure,
        "baseline_action": baseline,
    }


def normalize_record(record: Dict[str, Any], residual_scale: float = 200.0) -> Dict[str, Any]:
    record = dict(record)
    task = dict(record.get("task", {})) if isinstance(record.get("task"), dict) else {}
    game_name = str(record.get("game_name", task.get("goal", task.get("game_name", "unknown"))))
    access_key_id = str(record.get("access_key_id", task.get("access_key_id", "")))
    seen_status = str(record.get("seen_status", task.get("seen_status", "unknown")))

    state_summary = record.get("state_summary")
    if not isinstance(state_summary, dict):
        state_summary = _legacy_state_summary(record)
    state_summary = compact_state_summary({**state_summary, "game_name": game_name, "access_key_id": access_key_id, "seen_status": seen_status})

    next_state_summary = record.get("next_state_summary")
    if not isinstance(next_state_summary, dict):
        next_state = record.get("next_state", [0.0, 0.0])
        next_state_summary = {
            "resource_queue": safe_float(next_state[0] if len(next_state) > 0 else 0.0),
            "waiting_queue": safe_float(next_state[1] if len(next_state) > 1 else 0.0),
            "current_arrivals": safe_float(record.get("next_arrivals", 0.0)),
            "progress": clamp(state_summary.get("progress", 0.0) + 0.01, 0.0, 1.0),
            "backlog_pressure": safe_float(record.get("next_backlog_pressure", 0.0)),
        }
    next_state_summary = compact_state_summary({**next_state_summary, "game_name": game_name, "access_key_id": access_key_id, "seen_status": seen_status})

    baseline = safe_float(state_summary.get("baseline_action", 0.0))
    allocation = safe_float(record.get("allocation", record.get("action", baseline)))
    raw_action = record.get("raw_action")
    if raw_action is None:
        raw_action = raw_action_from_allocation(allocation, baseline, residual_scale=residual_scale)
    raw_action = clamp(safe_float(raw_action), -1.0, 1.0)
    return {
        "phase": str(record.get("phase", "unknown")),
        "batch": safe_int(record.get("batch", 0)),
        "traj_id": safe_int(record.get("traj_id", 0)),
        "t": safe_int(record.get("t", record.get("step", 0))),
        "task": {"goal": game_name, "access_key_id": access_key_id, "seen_status": seen_status, **task},
        "game_name": game_name,
        "access_key_id": access_key_id,
        "seen_status": seen_status,
        "state_summary": state_summary,
        "next_state_summary": next_state_summary,
        "descriptor": list(record.get("descriptor", [])) if isinstance(record.get("descriptor", []), (list, tuple)) else [],
        "raw_action": raw_action,
        "allocation": safe_float(allocation),
        "reward": safe_float(record.get("reward", 0.0)),
        "cache_available": safe_float(record.get("cache_available", state_summary["resource_queue"] + allocation)),
        "hit_rate": clamp(safe_float(record.get("hit_rate", 0.0)), 0.0, 1.0),
        "waste_rate": clamp(safe_float(record.get("waste_rate", 0.0)), 0.0, 1.0),
        "action_source": str(record.get("action_source", "history")),
        "llm_result": record.get("llm_result"),
        "prompt_style": record.get("prompt_style"),
        "ts": safe_int(record.get("ts", 0)),
    }


def trajectory_key(rec: Dict[str, Any]) -> str:
    return f"{rec.get('phase','unknown')}|{rec.get('batch',0)}|{rec.get('traj_id',0)}|{rec.get('game_name','unknown')}|{rec.get('access_key_id','')}"


def summarize_window(window: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not window:
        return {
            "arrivals_mean": 0.0,
            "arrivals_slope": 0.0,
            "waiting_mean": 0.0,
            "waiting_slope": 0.0,
            "resource_mean": 0.0,
            "resource_slope": 0.0,
            "reward_mean": 0.0,
            "reward_slope": 0.0,
            "allocation_mean": 0.0,
            "raw_action_mean": 0.0,
            "hit_rate_mean": 0.0,
            "waste_rate_mean": 0.0,
            "volatility": 0.0,
        }
    arrivals = [safe_float(r["state_summary"].get("current_arrivals", 0.0)) for r in window]
    waiting = [safe_float(r["state_summary"].get("waiting_queue", 0.0)) for r in window]
    resource = [safe_float(r["state_summary"].get("resource_queue", 0.0)) for r in window]
    rewards = [safe_float(r.get("reward", 0.0)) for r in window]
    allocations = [safe_float(r.get("allocation", 0.0)) for r in window]
    raws = [safe_float(r.get("raw_action", 0.0)) for r in window]
    hits = [safe_float(r.get("hit_rate", 0.0)) for r in window]
    wastes = [safe_float(r.get("waste_rate", 0.0)) for r in window]
    return {
        "arrivals_mean": mean(arrivals),
        "arrivals_slope": slope(arrivals),
        "waiting_mean": mean(waiting),
        "waiting_slope": slope(waiting),
        "resource_mean": mean(resource),
        "resource_slope": slope(resource),
        "reward_mean": mean(rewards),
        "reward_slope": slope(rewards),
        "allocation_mean": mean(allocations),
        "raw_action_mean": mean(raws),
        "hit_rate_mean": mean(hits),
        "waste_rate_mean": mean(wastes),
        "volatility": std(arrivals),
    }


def infer_regime(current_state: Dict[str, Any], trend: Dict[str, float]) -> str:
    waiting = safe_float(current_state.get("waiting_queue", 0.0))
    resource = safe_float(current_state.get("resource_queue", 0.0))
    arrivals = safe_float(current_state.get("current_arrivals", 0.0))
    baseline = safe_float(current_state.get("baseline_action", 0.0))
    backlog = safe_float(current_state.get("backlog_pressure", waiting / (resource + arrivals + 1.0)))
    if waiting > max(10.0, arrivals * 0.6):
        return "backlog_recovery" if trend.get("waiting_slope", 0.0) <= 0.0 else "surge"
    if trend.get("waste_rate_mean", 0.0) > 0.18 and resource > arrivals and baseline <= arrivals:
        return "oversupply_risk"
    if abs(trend.get("arrivals_slope", 0.0)) < 1.0 and abs(trend.get("waiting_slope", 0.0)) < 1.0 and backlog < 0.3:
        return "stable"
    return "volatile"


def state_vector(current_state: Dict[str, Any]) -> np.ndarray:
    return np.asarray([
        safe_float(current_state.get("resource_queue", 0.0)),
        safe_float(current_state.get("waiting_queue", 0.0)),
        safe_float(current_state.get("current_arrivals", 0.0)),
        safe_float(current_state.get("baseline_action", 0.0)),
        safe_float(current_state.get("backlog_pressure", 0.0)),
    ], dtype=np.float32)


def trend_vector(trend: Dict[str, float]) -> np.ndarray:
    return np.asarray([
        safe_float(trend.get("arrivals_mean", 0.0)),
        safe_float(trend.get("arrivals_slope", 0.0)),
        safe_float(trend.get("waiting_slope", 0.0)),
        safe_float(trend.get("resource_slope", 0.0)),
        safe_float(trend.get("reward_mean", 0.0)),
        safe_float(trend.get("reward_slope", 0.0)),
        safe_float(trend.get("waste_rate_mean", 0.0)),
        safe_float(trend.get("volatility", 0.0)),
    ], dtype=np.float32)


def summarize_future(decision_record: Dict[str, Any], future_window: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not future_window:
        future_window = [decision_record]
    rewards = [safe_float(r.get("reward", 0.0)) for r in future_window]
    hits = [safe_float(r.get("hit_rate", 0.0)) for r in future_window]
    wastes = [safe_float(r.get("waste_rate", 0.0)) for r in future_window]
    first_waiting = safe_float(decision_record["state_summary"].get("waiting_queue", 0.0))
    last_waiting = safe_float(future_window[-1].get("next_state_summary", {}).get("waiting_queue", 0.0))
    return {
        "reward_sum": float(np.sum(np.asarray(rewards, dtype=np.float32))),
        "reward_mean": mean(rewards),
        "waiting_delta": float(last_waiting - first_waiting),
        "hit_rate_mean": mean(hits),
        "waste_rate_mean": mean(wastes),
        "stability_score": float(mean(hits) - mean(wastes)),
    }


def quality_score_from_future(future: Dict[str, float]) -> float:
    x = (
        1.5 * safe_float(future.get("reward_mean", 0.0))
        + 0.5 * safe_float(future.get("hit_rate_mean", 0.0))
        - 0.5 * safe_float(future.get("waste_rate_mean", 0.0))
        - 0.01 * max(safe_float(future.get("waiting_delta", 0.0)), 0.0)
    )
    return float(np.tanh(x))


def make_takeaway(regime: str, decision: Dict[str, Any], trend: Dict[str, float], future: Dict[str, float]) -> str:
    state = decision.get("state_summary", {})
    raw = safe_float(decision.get("raw_action", 0.0))
    alloc = safe_float(decision.get("allocation", 0.0))
    return (
        f"regime={regime}; waiting={safe_float(state.get('waiting_queue',0.0)):.2f}; "
        f"arrivals={safe_float(state.get('current_arrivals',0.0)):.2f}; "
        f"trend_arrivals_slope={safe_float(trend.get('arrivals_slope',0.0)):.2f}; "
        f"teacher_raw={raw:.3f}; allocation={alloc:.2f}; "
        f"future_reward_mean={safe_float(future.get('reward_mean',0.0)):.3f}."
    )
