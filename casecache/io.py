from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json

import numpy as np
import pandas as pd

from .common import iter_jsonl, read_json, write_json, write_jsonl
from .schema import normalize_record
from .strategies import trace_to_frame


def load_history_records(path: str | Path, *, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    return [normalize_record(r) for r in iter_jsonl(path, max_lines=max_lines)]


def load_trace(path: str | Path, *, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    return [normalize_record(r) if "state_summary" in r else r for r in iter_jsonl(path, max_lines=max_lines)]


def save_run_outputs(output_dir: str | Path, *, summary: Dict[str, Any], trace: Sequence[Dict[str, Any]]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "output.summary.json", summary)
    write_jsonl(out / "output.trace.jsonl", trace)
    df = trace_to_frame(trace)
    df.to_csv(out / "output.trace.csv", index=False, encoding="utf-8-sig")
    # Use npz for compatibility with the existing project convention.
    np.savez(
        out / "output.npz",
        rewards=df["reward"].to_numpy(dtype=np.float32) if "reward" in df else np.asarray([], dtype=np.float32),
        allocations=df["allocation"].to_numpy(dtype=np.float32) if "allocation" in df else np.asarray([], dtype=np.float32),
        hit_rates=df["hit_rate"].to_numpy(dtype=np.float32) if "hit_rate" in df else np.asarray([], dtype=np.float32),
        waste_rates=df["waste_rate"].to_numpy(dtype=np.float32) if "waste_rate" in df else np.asarray([], dtype=np.float32),
        action_source=df["action_source"].astype(str).to_numpy() if "action_source" in df else np.asarray([], dtype=object),
        mean_return=np.asarray([float(summary.get("mean_return", 0.0))], dtype=np.float32),
    )


def summarize_npz(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    summary: Dict[str, Any] = {"file": str(p), "fields": list(data.files)}
    for name in data.files:
        arr = data[name]
        try:
            flat = np.asarray(arr, dtype=np.float64).reshape(-1)
            if flat.size:
                summary[f"{name}_mean"] = float(flat.mean())
                summary[f"{name}_min"] = float(flat.min())
                summary[f"{name}_max"] = float(flat.max())
                summary[f"{name}_shape"] = tuple(arr.shape)
        except Exception:
            summary[f"{name}_shape"] = tuple(arr.shape)
    return summary


def discover_outputs(root: str | Path) -> List[Dict[str, Any]]:
    root = Path(root)
    rows: List[Dict[str, Any]] = []
    if not root.exists():
        return rows
    for p in root.rglob("*"):
        if p.is_file() and (p.name.endswith("summary.json") or p.name.endswith("trace.jsonl") or p.suffix == ".npz"):
            rows.append({"path": str(p), "name": p.name, "parent": str(p.parent), "kind": p.suffix or p.name})
    return rows
