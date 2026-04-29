from __future__ import annotations

from typing import Any, Dict, List, Sequence
import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

from .cases import pca_coordinates
from .common import safe_float
from .strategies import trace_to_frame


def line_df_for_window(label: str, window: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(window):
        s = r.get("state_summary", {})
        rows.append({
            "label": label,
            "index": int(i),
            "arrivals": safe_float(s.get("current_arrivals", 0.0)),
            "waiting_queue": safe_float(s.get("waiting_queue", 0.0)),
            "resource_queue": safe_float(s.get("resource_queue", 0.0)),
            "reward": safe_float(r.get("reward", 0.0)),
        })
    return pd.DataFrame(rows)


def case_map_frame(case_library) -> pd.DataFrame:
    df = case_library.to_frame()
    if df.empty or case_library.case_features is None or case_library.case_features.shape[0] == 0:
        df["pc1"] = []
        df["pc2"] = []
        return df
    coords = pca_coordinates(case_library.case_features, n_components=2)
    df = df.copy()
    df["pc1"] = coords[:, 0]
    df["pc2"] = coords[:, 1]
    return df


def strategy_summary_frame(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for res in results:
        summary = dict(res.get("summary", {}))
        rows.append(summary)
    return pd.DataFrame(rows)
