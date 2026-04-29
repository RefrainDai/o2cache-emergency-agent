from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import pickle

import numpy as np
import pandas as pd

from .common import iter_jsonl, safe_float, stable_case_id, write_jsonl
from .schema import (
    infer_regime,
    make_takeaway,
    normalize_record,
    quality_score_from_future,
    state_vector,
    summarize_future,
    summarize_window,
    trajectory_key,
    trend_vector,
)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.size == 0:
        return np.zeros_like(matrix)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / norm


def _encode(values: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    mapping: Dict[str, int] = {}
    codes = np.zeros((len(values),), dtype=np.int32)
    for i, v in enumerate(values):
        key = str(v)
        if key not in mapping:
            mapping[key] = len(mapping)
        codes[i] = mapping[key]
    return codes, mapping


@dataclass
class CaseLibrary:
    window: int = 5
    future_horizon: int = 5
    max_cases: Optional[int] = None
    same_game_bonus: float = 0.05
    cases: List[Dict[str, Any]] = field(default_factory=list)
    case_features: Optional[np.ndarray] = None
    case_state_unit: Optional[np.ndarray] = None
    case_trend_unit: Optional[np.ndarray] = None
    case_quality_scores: Optional[np.ndarray] = None
    case_game_codes: Optional[np.ndarray] = None
    case_regime_codes: Optional[np.ndarray] = None
    case_trajectory_codes: Optional[np.ndarray] = None
    game_code_map: Dict[str, int] = field(default_factory=dict)
    regime_code_map: Dict[str, int] = field(default_factory=dict)
    games_in_library: set[str] = field(default_factory=set)

    @classmethod
    def from_history_path(
        cls,
        path: str | Path,
        *,
        window: int = 5,
        future_horizon: int = 5,
        max_cases: Optional[int] = None,
        same_game_bonus: float = 0.05,
        max_lines: Optional[int] = None,
    ) -> "CaseLibrary":
        records = [normalize_record(r) for r in iter_jsonl(path, max_lines=max_lines)]
        lib = cls(window=window, future_horizon=future_horizon, max_cases=max_cases, same_game_bonus=same_game_bonus)
        lib.build_from_records(records)
        return lib

    def build_from_records(self, records: Sequence[Dict[str, Any]]) -> None:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            rec = normalize_record(r)
            grouped.setdefault(trajectory_key(rec), []).append(rec)
            self.games_in_library.add(str(rec.get("game_name", "unknown")))
        cases: List[Dict[str, Any]] = []
        for key, traj in grouped.items():
            traj = sorted(traj, key=lambda x: int(x.get("t", 0)))
            if len(traj) <= self.window:
                continue
            for idx in range(self.window, len(traj)):
                recent_window = traj[idx - self.window : idx]
                decision = traj[idx]
                future_window = traj[idx : idx + self.future_horizon]
                trend = summarize_window(recent_window)
                regime = infer_regime(decision["state_summary"], trend)
                future = summarize_future(decision, future_window)
                quality = quality_score_from_future(future)
                case_id = stable_case_id(key, f"t{decision['t']}")
                case = {
                    "case_id": case_id,
                    "trajectory_key": key,
                    "game_name": decision.get("game_name", "unknown"),
                    "access_key_id": decision.get("access_key_id", ""),
                    "seen_status": decision.get("seen_status", "unknown"),
                    "decision_step": int(decision.get("t", 0)),
                    "recent_window": recent_window,
                    "current_state_summary": decision["state_summary"],
                    "baseline_action": safe_float(decision["state_summary"].get("baseline_action", 0.0)),
                    "teacher_raw_action": safe_float(decision.get("raw_action", 0.0)),
                    "teacher_allocation": safe_float(decision.get("allocation", 0.0)),
                    "reward": safe_float(decision.get("reward", 0.0)),
                    "hit_rate": safe_float(decision.get("hit_rate", 0.0)),
                    "waste_rate": safe_float(decision.get("waste_rate", 0.0)),
                    "trend_summary": trend,
                    "regime_tag": regime,
                    "future_outcome": future,
                    "quality_score": quality,
                    "takeaway": make_takeaway(regime, decision, trend, future),
                }
                case["feature"] = np.concatenate([state_vector(case["current_state_summary"]), trend_vector(case["trend_summary"])], axis=0).astype(np.float32)
                cases.append(case)
        if self.max_cases is not None and len(cases) > int(self.max_cases):
            cases = sorted(cases, key=lambda c: (safe_float(c.get("quality_score", 0.0)), -int(c.get("decision_step", 0))), reverse=True)[: int(self.max_cases)]
        self.cases = cases
        self._finalize()

    def _finalize(self) -> None:
        if not self.cases:
            self.case_features = np.zeros((0, 13), dtype=np.float32)
            self.case_state_unit = np.zeros((0, 5), dtype=np.float32)
            self.case_trend_unit = np.zeros((0, 8), dtype=np.float32)
            self.case_quality_scores = np.zeros((0,), dtype=np.float32)
            self.case_game_codes = np.zeros((0,), dtype=np.int32)
            self.case_regime_codes = np.zeros((0,), dtype=np.int32)
            self.case_trajectory_codes = np.zeros((0,), dtype=np.int32)
            return
        self.case_features = np.stack([np.asarray(c["feature"], dtype=np.float32) for c in self.cases], axis=0)
        self.case_state_unit = _normalize_rows(self.case_features[:, :5])
        self.case_trend_unit = _normalize_rows(self.case_features[:, 5:])
        self.case_quality_scores = np.asarray([safe_float(c.get("quality_score", 0.0)) for c in self.cases], dtype=np.float32)
        self.case_game_codes, self.game_code_map = _encode([str(c.get("game_name", "")) for c in self.cases])
        self.case_regime_codes, self.regime_code_map = _encode([str(c.get("regime_tag", "")) for c in self.cases])
        self.case_trajectory_codes, _ = _encode([str(c.get("trajectory_key", "")) for c in self.cases])
        self.games_in_library = set(str(c.get("game_name", "")) for c in self.cases)

    def stats(self) -> Dict[str, Any]:
        if not self.cases:
            return {"num_cases": 0}
        df = self.to_frame(include_feature=False)
        return {
            "num_cases": int(len(df)),
            "num_games": int(df["game_name"].nunique()),
            "num_trajectories": int(df["trajectory_key"].nunique()),
            "quality_mean": float(df["quality_score"].mean()),
            "quality_max": float(df["quality_score"].max()),
            "reward_mean": float(df["reward"].mean()),
            "hit_rate_mean": float(df["hit_rate"].mean()),
            "waste_rate_mean": float(df["waste_rate"].mean()),
        }

    def to_frame(self, include_feature: bool = False) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for c in self.cases:
            row = {
                "case_id": c.get("case_id"),
                "trajectory_key": c.get("trajectory_key"),
                "game_name": c.get("game_name"),
                "access_key_id": c.get("access_key_id"),
                "seen_status": c.get("seen_status"),
                "decision_step": c.get("decision_step"),
                "regime_tag": c.get("regime_tag"),
                "teacher_raw_action": c.get("teacher_raw_action"),
                "teacher_allocation": c.get("teacher_allocation"),
                "reward": c.get("reward"),
                "hit_rate": c.get("hit_rate"),
                "waste_rate": c.get("waste_rate"),
                "quality_score": c.get("quality_score"),
                "takeaway": c.get("takeaway"),
                "arrivals_mean": c.get("trend_summary", {}).get("arrivals_mean"),
                "arrivals_slope": c.get("trend_summary", {}).get("arrivals_slope"),
                "waiting_slope": c.get("trend_summary", {}).get("waiting_slope"),
                "resource_slope": c.get("trend_summary", {}).get("resource_slope"),
            }
            if include_feature:
                row["feature"] = c.get("feature")
            rows.append(row)
        return pd.DataFrame(rows)

    def query(self, *, game_name: str, recent_window: Sequence[Dict[str, Any]], current_state_summary: Dict[str, Any], k: int = 5, exclude_case_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        if self.case_features is None:
            self._finalize()
        if self.case_features is None or self.case_features.shape[0] == 0:
            return []
        trend = summarize_window([normalize_record(r) for r in recent_window] if recent_window else [])
        qregime = infer_regime(current_state_summary, trend)
        state_query = _normalize_vector(state_vector(current_state_summary))
        trend_query = _normalize_vector(trend_vector(trend))
        state_scores = self.case_state_unit @ state_query
        trend_scores = self.case_trend_unit @ trend_query
        quality_scores = self.case_quality_scores if self.case_quality_scores is not None else np.zeros_like(state_scores)
        scores = (0.45 * state_scores + 0.30 * trend_scores + 0.05 * quality_scores).astype(np.float32)
        regime_code = self.regime_code_map.get(str(qregime))
        if regime_code is not None and self.case_regime_codes is not None:
            regime_match = (self.case_regime_codes == regime_code).astype(np.float32)
            scores = scores + 0.15 * regime_match
        else:
            regime_match = np.zeros_like(scores)
        game_code = self.game_code_map.get(str(game_name))
        if game_code is not None and self.case_game_codes is not None:
            same_game = (self.case_game_codes == game_code).astype(np.float32)
            scores = scores + self.same_game_bonus * same_game
        else:
            same_game = np.zeros_like(scores)
        if exclude_case_ids:
            exclude = set(str(x) for x in exclude_case_ids)
            scores = scores.copy()
            for i, c in enumerate(self.cases):
                if str(c.get("case_id")) in exclude:
                    scores[i] = -np.inf
        total = int(scores.shape[0])
        idxs = np.argsort(scores)[::-1]
        selected: List[Dict[str, Any]] = []
        seen_traj: set[str] = set()
        for idx in idxs:
            if len(selected) >= int(k):
                break
            if not np.isfinite(scores[idx]):
                continue
            c = dict(self.cases[int(idx)])
            c["score"] = float(scores[idx])
            c["state_similarity"] = float(state_scores[idx])
            c["trend_similarity"] = float(trend_scores[idx])
            c["query_regime"] = str(qregime)
            c["regime_match"] = bool(regime_match[idx] > 0.5)
            c["same_game_bonus"] = float(self.same_game_bonus if same_game[idx] > 0.5 else 0.0)
            # Encourage diversity in the first pass, but do not block if there are not enough trajectories.
            traj = str(c.get("trajectory_key", ""))
            if len(selected) < int(k) and traj in seen_traj and len(seen_traj) < int(k):
                continue
            seen_traj.add(traj)
            selected.append(c)
        if len(selected) < int(k):
            existing = {str(c.get("case_id")) for c in selected}
            for idx in idxs:
                if len(selected) >= int(k):
                    break
                c0 = self.cases[int(idx)]
                if str(c0.get("case_id")) in existing or not np.isfinite(scores[idx]):
                    continue
                c = dict(c0)
                c["score"] = float(scores[idx])
                c["state_similarity"] = float(state_scores[idx])
                c["trend_similarity"] = float(trend_scores[idx])
                c["query_regime"] = str(qregime)
                c["regime_match"] = bool(regime_match[idx] > 0.5)
                c["same_game_bonus"] = float(self.same_game_bonus if same_game[idx] > 0.5 else 0.0)
                selected.append(c)
        return selected

    def fallback_raw_action(self, *, game_name: str, recent_window: Sequence[Dict[str, Any]], current_state_summary: Dict[str, Any], k: int = 5) -> float:
        cases = self.query(game_name=game_name, recent_window=recent_window, current_state_summary=current_state_summary, k=k)
        if not cases:
            return 0.0
        scores = np.asarray([max(safe_float(c.get("score", 0.0)), 0.0) for c in cases], dtype=np.float32)
        raws = np.asarray([safe_float(c.get("teacher_raw_action", 0.0)) for c in cases], dtype=np.float32)
        if float(scores.sum()) <= 1e-8:
            return float(np.median(raws))
        weights = scores / scores.sum()
        return float(np.clip(np.dot(weights, raws), -1.0, 1.0))

    def save_cache(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_cache(path: str | Path) -> "CaseLibrary":
        with Path(path).open("rb") as f:
            return pickle.load(f)

    def export_jsonl(self, path: str | Path) -> None:
        rows = []
        for c in self.cases:
            d = dict(c)
            if "feature" in d:
                d["feature"] = np.asarray(d["feature"]).tolist()
            rows.append(d)
        write_jsonl(path, rows)


def pca_coordinates(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        return np.zeros((0, n_components), dtype=np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    # Stable and dependency-free PCA by SVD.
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :n_components] * s[:n_components]
    if coords.shape[1] < n_components:
        pad = np.zeros((coords.shape[0], n_components - coords.shape[1]), dtype=np.float32)
        coords = np.concatenate([coords, pad], axis=1)
    return coords.astype(np.float32)
