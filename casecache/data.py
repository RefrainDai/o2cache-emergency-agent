from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

from .common import safe_float

CANONICAL_FIELDS = ["log_time", "num", "access_key_id", "game_name"]
FIELD_ALIASES = {
    "timestamp": "log_time",
    "time": "log_time",
    "ts": "log_time",
    "arrivals": "num",
    "arrival": "num",
    "current_arrivals": "num",
    "request_count": "num",
    "requests": "num",
    "service_type": "game_name",
    "business_type": "game_name",
    "task_type": "game_name",
    "sequence_id": "access_key_id",
    "seq_id": "access_key_id",
    "user_id": "access_key_id",
}


@dataclass
class ValidationReport:
    ok: bool
    missing_fields: List[str]
    mapped_columns: Dict[str, str]
    warnings: List[str]
    summary: Dict[str, Any]


def _rename_aliases(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    lower_to_original = {str(c).lower(): str(c) for c in df.columns}
    rename: Dict[str, str] = {}
    for canonical in CANONICAL_FIELDS:
        if canonical in df.columns:
            mapping[canonical] = canonical
            continue
        for alias, target in FIELD_ALIASES.items():
            if target == canonical and alias.lower() in lower_to_original:
                original = lower_to_original[alias.lower()]
                rename[original] = canonical
                mapping[canonical] = original
                break
    out = df.rename(columns=rename).copy()
    for canonical in CANONICAL_FIELDS:
        if canonical in out.columns and canonical not in mapping:
            mapping[canonical] = canonical
    return out, mapping


def load_dataset(path_or_file: Any, *, file_type: Optional[str] = None) -> pd.DataFrame:
    name = getattr(path_or_file, "name", str(path_or_file))
    suffix = Path(name).suffix.lower()
    ftype = (file_type or suffix).lower().replace(".", "")
    if ftype in {"csv", ""}:
        df = pd.read_csv(path_or_file)
    elif ftype in {"jsonl", "ndjson"}:
        rows = []
        for line in path_or_file:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    elif ftype == "json":
        df = pd.read_json(path_or_file)
    else:
        raise ValueError(f"Unsupported data type: {ftype}")
    df, _ = _rename_aliases(df)
    if "log_time" in df.columns:
        df["log_time"] = pd.to_datetime(df["log_time"], errors="coerce")
    if "num" in df.columns:
        df["num"] = pd.to_numeric(df["num"], errors="coerce")
    if "access_key_id" in df.columns:
        df["access_key_id"] = df["access_key_id"].astype(str)
    if "game_name" in df.columns:
        df["game_name"] = df["game_name"].astype(str)
    return df


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    df2, mapped = _rename_aliases(df)
    missing = [c for c in CANONICAL_FIELDS if c not in df2.columns]
    warnings: List[str] = []
    if "log_time" in df2.columns:
        null_ts = int(pd.to_datetime(df2["log_time"], errors="coerce").isna().sum())
        if null_ts > 0:
            warnings.append(f"log_time 中有 {null_ts} 行无法解析为时间。")
    if "num" in df2.columns:
        nums = pd.to_numeric(df2["num"], errors="coerce")
        null_num = int(nums.isna().sum())
        if null_num > 0:
            warnings.append(f"num 中有 {null_num} 行无法转为数值。")
        if (nums.dropna() < 0).any():
            warnings.append("num 中存在负数请求量，平台会将其视为异常数据。")
    summary: Dict[str, Any] = {
        "num_rows": int(len(df2)),
        "num_columns": int(len(df2.columns)),
    }
    if "game_name" in df2.columns:
        summary["num_game_types"] = int(df2["game_name"].nunique(dropna=True))
    if "access_key_id" in df2.columns:
        summary["num_sequences"] = int(df2["access_key_id"].nunique(dropna=True))
    if "num" in df2.columns:
        nums = pd.to_numeric(df2["num"], errors="coerce")
        summary.update({
            "arrival_mean": float(nums.mean()) if nums.notna().any() else 0.0,
            "arrival_max": float(nums.max()) if nums.notna().any() else 0.0,
            "arrival_std": float(nums.std()) if nums.notna().sum() > 1 else 0.0,
        })
    return ValidationReport(ok=len(missing) == 0 and not any("无法" in w for w in warnings), missing_fields=missing, mapped_columns=mapped, warnings=warnings, summary=summary)


def prepare_request_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df, _ = _rename_aliases(df)
    required = set(CANONICAL_FIELDS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    out = df[CANONICAL_FIELDS].copy()
    out["log_time"] = pd.to_datetime(out["log_time"], errors="coerce")
    out["num"] = pd.to_numeric(out["num"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["access_key_id"] = out["access_key_id"].astype(str)
    out["game_name"] = out["game_name"].astype(str)
    out = out.dropna(subset=["log_time", "access_key_id", "game_name"])
    out = out.sort_values(["game_name", "access_key_id", "log_time"], kind="mergesort").reset_index(drop=True)
    return out


def dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    df = prepare_request_dataframe(df)
    seq_lengths = df.groupby(["game_name", "access_key_id"], sort=False).size()
    stats = {
        "rows": int(len(df)),
        "game_types": int(df["game_name"].nunique()),
        "sequences": int(seq_lengths.shape[0]),
        "mean_arrivals": float(df["num"].mean()),
        "max_arrivals": float(df["num"].max()),
        "nonzero_ratio": float((df["num"] > 0).mean()),
        "seq_len_min": int(seq_lengths.min()) if not seq_lengths.empty else 0,
        "seq_len_mean": float(seq_lengths.mean()) if not seq_lengths.empty else 0.0,
        "seq_len_max": int(seq_lengths.max()) if not seq_lengths.empty else 0,
        "time_start": str(df["log_time"].min()) if len(df) else "",
        "time_end": str(df["log_time"].max()) if len(df) else "",
    }
    return stats


def request_groups(df: pd.DataFrame):
    df = prepare_request_dataframe(df)
    return df.groupby(["game_name", "access_key_id"], sort=False)


def build_naive_state_rows(
    df: pd.DataFrame,
    *,
    initial_resource: float = 10.0,
    initial_waiting: float = 0.0,
    residual_scale: float = 200.0,
    max_allocation: float = 800.0,
) -> List[Dict[str, Any]]:
    """Create state rows from raw request data using baseline allocation.

    This is for cold-start visualization. It does not claim to be an expert
    policy. It produces a trace with baseline actions to allow users to inspect
    data dynamics before training or loading a real history file.
    """
    from .rewards import compute_allocation_outcome, myopic_base_allocation, action_to_allocation

    rows: List[Dict[str, Any]] = []
    for (game, seq), g in request_groups(df):
        resource = float(initial_resource)
        waiting = float(initial_waiting)
        g = g.reset_index(drop=True)
        n = len(g)
        for t, rec in g.iterrows():
            arrivals = float(rec["num"])
            progress = 0.0 if n <= 1 else float(t) / float(n - 1)
            baseline = myopic_base_allocation(resource, waiting, arrivals, max_allocation=max_allocation)
            allocation = baseline
            raw_action = 0.0
            metrics = compute_allocation_outcome(resource, waiting, arrivals, allocation)
            state_summary = {
                "game_name": str(game),
                "access_key_id": str(seq),
                "seen_status": "cold_start",
                "resource_queue": resource,
                "waiting_queue": waiting,
                "current_arrivals": arrivals,
                "progress": progress,
                "backlog_pressure": waiting / (resource + arrivals + 1.0),
                "baseline_action": baseline,
            }
            next_summary = {
                "game_name": str(game),
                "access_key_id": str(seq),
                "seen_status": "cold_start",
                "resource_queue": metrics["next_resource_queue"],
                "waiting_queue": metrics["next_waiting_queue"],
                "current_arrivals": float(g.loc[t + 1, "num"]) if t + 1 < n else 0.0,
                "progress": min(progress + 1.0 / max(n - 1, 1), 1.0),
                "backlog_pressure": metrics["next_waiting_queue"] / (metrics["next_resource_queue"] + (float(g.loc[t + 1, "num"]) if t + 1 < n else 0.0) + 1.0),
                "baseline_action": myopic_base_allocation(metrics["next_resource_queue"], metrics["next_waiting_queue"], float(g.loc[t + 1, "num"]) if t + 1 < n else 0.0, max_allocation=max_allocation),
            }
            rows.append({
                "phase": "cold_start_baseline",
                "batch": 0,
                "traj_id": hash((str(game), str(seq))) % 10_000_000,
                "t": int(t),
                "task": {"goal": str(game), "access_key_id": str(seq), "seen_status": "cold_start"},
                "game_name": str(game),
                "access_key_id": str(seq),
                "seen_status": "cold_start",
                "state_summary": state_summary,
                "next_state_summary": next_summary,
                "raw_action": raw_action,
                "allocation": allocation,
                "reward": metrics["reward"],
                "cache_available": metrics["cache_available"],
                "hit_rate": metrics["hit_rate"],
                "waste_rate": metrics["waste_rate"],
                "action_source": "baseline",
            })
            resource = metrics["next_resource_queue"]
            waiting = metrics["next_waiting_queue"]
    return rows
