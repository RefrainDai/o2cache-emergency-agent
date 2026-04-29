from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:  # pragma: no cover
    px = None
    go = None

from casecache.cases import CaseLibrary
from casecache.common import safe_float, write_json, write_jsonl
from casecache.data import (
    build_naive_state_rows,
    dataset_statistics,
    load_dataset,
    prepare_request_dataframe,
    validate_dataset,
)
from casecache.io import discover_outputs, load_trace, save_run_outputs, summarize_npz
from casecache.llm_adapter import OpenAICompatibleClient
from casecache.rewards import compute_allocation_outcome, evaluate_raw_action, myopic_base_allocation
from casecache.schema import infer_regime, normalize_record, summarize_window
from casecache.strategies import StrategyConfig, run_strategy_on_requests, trace_to_frame
from casecache.viz import case_map_frame, line_df_for_window, strategy_summary_frame
from agent_integration.agent_event_mapper import chinese_event_frame, chinese_load_frame, events_to_load_dataframe
from agent_integration.mock_agent_runner import DEFAULT_USER_TASK, run_demo_agent

ROOT = Path(__file__).resolve().parent
DEMO_DATA = ROOT / ("demo" + "_data")
DEMO_OUTPUTS = ROOT / ("demo" + "_outputs")
ASSETS = ROOT / "assets"
OVERVIEW_IMAGE = ASSETS / "project_overview.png"

SOURCE_BUILTIN = "builtin"
SOURCE_AGENT = "agent"
SOURCE_UPLOAD = "upload"
SOURCE_LOCAL = "local"
BUILTIN_SOURCE_LABEL = "内置演示任务负载"
AGENT_SOURCE_LABEL = "多模态智能体工具调用演示负载"
UPLOAD_SOURCE_LABEL = "用户上传任务负载"
LOCAL_SOURCE_LABEL = "本机任务负载文件"

# 页面展示名称与内部策略标识的映射。内部标识仅用于保持既有运行逻辑稳定。
METHOD_OPTIONS = {
    "即时规则分配基线": "Baseline",
    "启发式状态调节基线": "Rule-Heuristic",
    "相似案例缓存预测（消融基线）": "retriever",
    "ICL少样本缓存决策（主方法）": "llm",
}
METHOD_DESCRIPTIONS = {
    "即时规则分配基线": "仅依据当前空闲资源、等待队列和请求到达量进行短视分配。",
    "启发式状态调节基线": "根据请求趋势和排队压力进行规则修正。",
    "相似案例缓存预测（消融基线）": "检索 Top-K 相似历史案例，并融合历史决策动作形成缓存建议。",
    "ICL少样本缓存决策（主方法）": "将相似历史案例与当前状态共同组织为上下文输入，由 ICL 决策器生成下一时隙资源缓存量。",
}

st.set_page_config(
    page_title="应急智枢 ICL资源缓存预测平台",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    --brand-blue: #0b3a74;
    --brand-cyan: #0ea5e9;
    --brand-ink: #0f172a;
    --brand-muted: #64748b;
    --brand-line: #dbe7f3;
    --brand-soft: #f6fbff;
    --brand-accent: #f59e0b;
}
.stApp {background: #f8fbff;}
.block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1280px;}
.small-note {color: var(--brand-muted); font-size: 0.95rem;}
.section-note {color: #475569; font-size: 0.98rem; line-height: 1.75; margin: 0.25rem 0 1rem 0;}
.hero-band {
    border: 1px solid #0f5ea8;
    border-radius: 8px;
    padding: 1.35rem 1.45rem;
    background:
        linear-gradient(135deg, rgba(7, 39, 78, 0.96), rgba(12, 90, 153, 0.92)),
        radial-gradient(circle at 88% 18%, rgba(14, 165, 233, 0.38), transparent 28%);
    color: white;
    box-shadow: 0 12px 28px rgba(15, 61, 114, 0.18);
    margin-bottom: 1rem;
}
.hero-eyebrow {font-size: 0.92rem; letter-spacing: 0; color: #bde9ff; font-weight: 700; margin-bottom: 0.35rem;}
.hero-title {font-size: 2.05rem; line-height: 1.24; font-weight: 800; margin-bottom: 0.5rem;}
.hero-subtitle {font-size: 1.02rem; color: #e0f2fe; margin-bottom: 0.65rem;}
.hero-copy {font-size: 0.98rem; line-height: 1.72; color: #f8fafc; max-width: 980px;}
.flow-wrap {display: flex; flex-wrap: wrap; gap: 0.55rem; align-items: stretch; margin: 0.55rem 0 1.2rem 0;}
.flow-step {
    flex: 1 1 148px;
    min-width: 140px;
    border: 1px solid var(--brand-line);
    border-radius: 8px;
    background: #ffffff;
    padding: 0.78rem 0.82rem;
    box-shadow: 0 4px 14px rgba(15, 61, 114, 0.07);
    color: var(--brand-ink);
}
.flow-num {display:inline-block; color:white; background: var(--brand-blue); border-radius:999px; padding:0.08rem 0.46rem; font-weight:700; font-size:0.78rem; margin-bottom:0.36rem;}
.flow-title {font-weight: 700; font-size: 0.96rem; line-height: 1.45;}
.cap-card, .metric-card {
    border: 1px solid var(--brand-line);
    border-radius: 8px;
    padding: 0.9rem 0.95rem;
    background: #ffffff;
    box-shadow: 0 5px 16px rgba(15, 61, 114, 0.07);
    min-height: 118px;
}
.cap-card strong {display:block; color: var(--brand-blue); font-size: 1rem; margin-bottom: 0.35rem;}
.cap-card span {display:block; color:#475569; font-size:0.93rem; line-height:1.62;}
.metric-card {min-height: 104px; border-top: 3px solid var(--brand-cyan);}
.metric-card.emphasis {border-top-color: var(--brand-accent);}
.metric-name {color:#475569; font-size:0.86rem; margin-bottom:0.35rem;}
.metric-value {color:var(--brand-ink); font-size:1.55rem; font-weight:800;}
.metric-caption {color:#64748b;font-size:0.82rem;margin-top:0.25rem;line-height:1.45;}
.soft-panel {
    border: 1px solid var(--brand-line);
    border-radius: 8px;
    background: #ffffff;
    padding: 0.95rem 1rem;
    box-shadow: 0 4px 14px rgba(15, 61, 114, 0.06);
    margin: 0.35rem 0 1rem 0;
}
.decision-chain {
    border-left: 4px solid var(--brand-accent);
    background: #fffaf0;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: #334155;
    line-height: 1.7;
    margin-bottom: 1rem;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--brand-line);
    border-radius: 8px;
    padding: 0.72rem 0.82rem;
    box-shadow: 0 4px 14px rgba(15, 61, 114, 0.06);
}
div[data-testid="stMetricLabel"] p {color:#475569; font-size:0.86rem;}
div[data-testid="stMetricValue"] {color: var(--brand-ink);}
section[data-testid="stSidebar"] {background: #092345;}
section[data-testid="stSidebar"] * {color: #f8fafc;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {background: rgba(255,255,255,0.04); border-radius: 6px; padding: 0.18rem 0.25rem;}
</style>
""",
    unsafe_allow_html=True,
)


def _init_state():
    defaults = {
        "request_df": None,
        "history_records": None,
        "case_library": None,
        "last_result": None,
        "last_run_dir": None,
        "agent_tool_history": None,
        "agent_load_df": None,
        "request_source_kind": SOURCE_BUILTIN,
        "request_source_label": BUILTIN_SOURCE_LABEL,
        "request_source_note": "平台内置灾害任务负载样例。",
        "request_source_metadata": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def page_title(title: str, subtitle: str = ""):
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='small-note'>{subtitle}</div>", unsafe_allow_html=True)
        st.divider()


def section_note(text: str):
    st.markdown(f"<div class='section-note'>{text}</div>", unsafe_allow_html=True)


def set_request_source(
    df: pd.DataFrame,
    *,
    kind: str,
    label: str,
    note: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    clear_outputs: bool = True,
) -> None:
    previous_kind = st.session_state.get("request_source_kind")
    st.session_state.request_df = df.copy()
    st.session_state.request_source_kind = kind
    st.session_state.request_source_label = label
    st.session_state.request_source_note = note
    st.session_state.request_source_metadata = metadata or {}
    if clear_outputs:
        st.session_state.last_result = None
        st.session_state["comparison_results"] = []
        if previous_kind != kind:
            st.session_state.case_library = None
            st.session_state.history_records = None


def restore_builtin_request_source() -> None:
    set_request_source(
        load_demo_requests(),
        kind=SOURCE_BUILTIN,
        label=BUILTIN_SOURCE_LABEL,
        note="平台内置灾害任务负载样例。",
    )


def get_current_request_df() -> pd.DataFrame:
    df = st.session_state.get("request_df")
    if df is None:
        restore_builtin_request_source()
        df = st.session_state.request_df
    return df


def current_source_label() -> str:
    return str(st.session_state.get("request_source_label") or BUILTIN_SOURCE_LABEL)


def current_source_kind() -> str:
    return str(st.session_state.get("request_source_kind") or SOURCE_BUILTIN)


def render_data_source_status(context_key: str, *, show_restore: bool = True) -> None:
    note = st.session_state.get("request_source_note", "")
    st.markdown(
        f"<div class='soft-panel'><strong>当前数据来源：</strong>{current_source_label()}"
        + (f"<br><span class='small-note'>{note}</span>" if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    if show_restore and current_source_kind() != SOURCE_BUILTIN:
        if st.button("恢复使用内置演示数据", key=f"restore_builtin_{context_key}"):
            restore_builtin_request_source()
            st.success("已恢复为内置演示任务负载。")
            st.rerun()


def is_agent_source() -> bool:
    return current_source_kind() == SOURCE_AGENT


def agent_tool_display_for_task(task_type: Any) -> str:
    text = str(task_type)
    for item in st.session_state.get("agent_tool_history") or []:
        if str(item.get("task_type")) == text:
            return str(item.get("display_name") or item.get("tool_name") or "")
    return ""


def render_flow(steps: List[str]):
    html = ["<div class='flow-wrap'>"]
    for i, step in enumerate(steps, start=1):
        html.append(
            f"<div class='flow-step'><div class='flow-num'>{i}</div><div class='flow-title'>{step}</div></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_capability_cards(cards: List[tuple[str, str]]):
    cols = st.columns(len(cards))
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='cap-card'><strong>{title}</strong><span>{body}</span></div>",
                unsafe_allow_html=True,
            )


def render_metric_cards(cards: List[tuple[str, str, str, bool]]):
    cols = st.columns(len(cards))
    for col, (name, value, caption, emphasis) in zip(cols, cards):
        cls = "metric-card emphasis" if emphasis else "metric-card"
        with col:
            st.markdown(
                f"<div class='{cls}'><div class='metric-name'>{name}</div><div class='metric-value'>{value}</div><div class='metric-caption'>{caption}</div></div>",
                unsafe_allow_html=True,
            )


def format_metric(value: Any, digits: int = 3, fallback: str = "待生成") -> str:
    try:
        if value is None:
            return fallback
        val = float(value)
        if not np.isfinite(val):
            return fallback
        return f"{val:.{digits}f}"
    except Exception:
        return fallback


def current_demo_metrics() -> Dict[str, Any]:
    result = st.session_state.get("last_result")
    if isinstance(result, dict) and isinstance(result.get("summary"), dict):
        summary = result["summary"]
        return {
            "保障率": format_metric(summary.get("mean_hit_rate")),
            "冗余率": format_metric(summary.get("mean_waste_rate")),
            "综合评价": format_metric(summary.get("mean_reward")),
            "说明": f"来自当前方法：{summary.get('method_label', '最近一次运行')}",
        }
    try:
        df = get_current_request_df()
        rows = build_naive_state_rows(prepare_request_dataframe(df).head(240))
        if rows:
            return {
                "保障率": format_metric(np.mean([safe_float(r.get("hit_rate", 0.0)) for r in rows])),
                "冗余率": format_metric(np.mean([safe_float(r.get("waste_rate", 0.0)) for r in rows])),
                "综合评价": format_metric(np.mean([safe_float(r.get("reward", 0.0)) for r in rows])),
                "说明": f"来自{current_source_label()}的即时规则演示结果",
            }
    except Exception:
        pass
    return {"保障率": "待生成", "冗余率": "待生成", "综合评价": "待生成", "说明": "运行预测后显示当前演示结果"}


DECISION_EXPORT_COLS = [
    "t",
    "game_name",
    "access_key_id",
    "current_arrivals",
    "resource_queue",
    "waiting_queue",
    "baseline_action",
    "raw_action",
    "allocation",
    "reward",
    "hit_rate",
    "waste_rate",
    "next_resource_queue",
    "next_waiting_queue",
    "action_source",
    "retrieved_case_ids",
]

CASE_EXPORT_COLS = [
    "case_id",
    "game_name",
    "access_key_id",
    "decision_step",
    "regime_tag",
    "teacher_allocation",
    "reward",
    "hit_rate",
    "waste_rate",
    "quality_score",
    "arrivals_mean",
    "arrivals_slope",
    "waiting_slope",
    "resource_slope",
]


def decision_export_frame(tdf: pd.DataFrame) -> pd.DataFrame:
    if tdf is None or tdf.empty:
        return pd.DataFrame()
    out = tdf.copy()
    if "action_source" in out.columns:
        out["action_source"] = out["action_source"].map(ui_source_label)
    cols = [c for c in DECISION_EXPORT_COLS if c in out.columns]
    return ui_frame(out[cols])


def comparison_export_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    comp = pd.DataFrame(rows).copy()
    if "method_label" in comp.columns:
        comp = comp.rename(columns={"method_label": "方法"})
    cols = ["方法", "mean_reward", "mean_hit_rate", "mean_waste_rate", "mean_allocation", "mean_return", "num_rows", "num_tasks"]
    cols = [c for c in cols if c in comp.columns]
    return ui_frame(comp[cols])


def case_export_frame(lib: Optional[CaseLibrary]) -> pd.DataFrame:
    if lib is None or not getattr(lib, "cases", []):
        return pd.DataFrame()
    case_df = lib.to_frame()
    cols = [c for c in CASE_EXPORT_COLS if c in case_df.columns]
    return ui_frame(case_df[cols].copy())


def task_load_export_frame(df_req: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df_req is None:
        return pd.DataFrame()
    load_df = prepare_request_dataframe(df_req).copy()
    load_df["series"] = load_df["game_name"].map(display_task_name) + " / " + load_df["access_key_id"].map(display_sequence_name)
    cols = ["log_time", "game_name", "access_key_id", "num", "series"]
    return ui_frame(load_df[[c for c in cols if c in load_df.columns]])


def task_load_counts(df_req: Optional[pd.DataFrame]) -> Dict[str, int]:
    if df_req is None:
        return {"任务类型数量": 0, "请求序列数量": 0, "时间步数量": 0}
    try:
        load_df = prepare_request_dataframe(df_req)
        return {
            "任务类型数量": int(load_df["game_name"].nunique()),
            "请求序列数量": int(load_df["access_key_id"].nunique()),
            "时间步数量": int(len(load_df)),
        }
    except Exception:
        return {"任务类型数量": 0, "请求序列数量": 0, "时间步数量": 0}


def build_demo_report_markdown(
    *,
    result: Optional[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    df_req: Optional[pd.DataFrame],
    lib: Optional[CaseLibrary],
) -> str:
    counts = task_load_counts(df_req)
    case_count = len(lib.cases) if lib is not None and getattr(lib, "cases", []) else 0
    source_label = current_source_label()
    summary = dict(result.get("summary", {})) if isinstance(result, dict) else {}
    method = summary.get("method_label", "尚未运行")
    hit = format_metric(summary.get("mean_hit_rate")) if summary else "尚未运行"
    waste = format_metric(summary.get("mean_waste_rate")) if summary else "尚未运行"
    score = format_metric(summary.get("mean_reward")) if summary else "尚未运行"
    allocation = format_metric(summary.get("mean_allocation"), digits=2) if summary else "尚未运行"

    lines = [
        "# O2Cache 演示报告摘要",
        "",
        "## 项目信息",
        "",
        "- 项目名称：应急智枢---面向灾害应急的多模态智能体辅助决策系统",
        "- 团队名称：O₂Cache",
        "- 平台定位：基于历史案例库与上下文学习的应急服务资源缓存预测平台",
        "",
        "## 本次演示数据",
        "",
        f"- 当前数据来源：{source_label}",
        f"- 任务类型数量：{counts['任务类型数量']}",
        f"- 请求序列数量：{counts['请求序列数量']}",
        f"- 时间步数量：{counts['时间步数量']}",
        f"- 历史案例库规模：{case_count} 个案例",
        "",
        "## 当前运行结果",
        "",
        f"- 当前方法：{method}",
        "- 当前推荐方法：ICL少样本缓存决策",
        f"- 请求保障率：{hit}",
        f"- 资源冗余率：{waste}",
        f"- 综合评价值：{score}",
        f"- 平均资源缓存量：{allocation}",
        "",
    ]

    if comparison_rows:
        comp = comparison_export_frame(comparison_rows)
        lines.extend(["## 策略对比简要结果", ""])
        lines.append(markdown_table(comp))
        lines.append("")
    else:
        lines.extend(["## 策略对比简要结果", "", "尚未生成策略对比结果。", ""])

    if is_agent_source():
        lines.extend(
            [
                "## 多模态智能体任务接入说明",
                "",
                "本次演示使用多模态智能体工具调用轨迹生成任务服务负载，用于展示智能体任务处理过程如何转化为资源缓存预测输入。",
                "该负载为演示模式生成，不代表真实无人机在线控制或真实云资源监控数据。",
                "",
            ]
        )

    lines.extend([
        "## 简短结论",
        "",
        "平台通过历史案例库、相似案例检索与上下文学习机制，为灾害响应多模态任务服务提供资源缓存量预测和资源预热建议。上述结果仅基于当前接入的演示数据和当前运行配置生成。",
        "",
    ])
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "暂无数据。"
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(c, "")).replace("\n", " ") for c in df.columns]
        out.append("| " + " | ".join(values) + " |")
    return "\n".join(out)


UI_LABELS = {
    "log_time": "时间步",
    "timestamp": "时间步",
    "t": "时隙",
    "index": "时隙",
    "num": "请求到达量",
    "current_arrivals": "请求到达量",
    "arrivals": "请求到达量",
    "game_name": "任务类型",
    "access_key_id": "请求序列",
    "sequence_id": "请求序列",
    "series": "任务序列",
    "rows": "记录数",
    "game_types": "任务类型数",
    "sequences": "请求序列数",
    "mean_arrivals": "平均请求到达量",
    "max_arrivals": "最大请求到达量",
    "arrival_mean": "平均请求到达量",
    "arrival_max": "最大请求到达量",
    "arrival_std": "请求到达量标准差",
    "nonzero_ratio": "非零请求占比",
    "seq_len_min": "最短序列长度",
    "seq_len_mean": "平均序列长度",
    "seq_len_max": "最长序列长度",
    "time_start": "起始时间",
    "time_end": "结束时间",
    "length": "序列长度",
    "resource_queue": "空闲算力资源",
    "waiting_queue": "等待服务队列",
    "backlog_pressure": "积压压力",
    "progress": "序列进度",
    "seen_status": "案例覆盖状态",
    "cache_available": "可服务资源",
    "waiting_cost": "等待成本",
    "idle_cost": "冗余成本",
    "next_resource_queue": "下一时隙空闲算力资源",
    "next_waiting_queue": "下一时隙等待服务队列",
    "allocation": "资源缓存量",
    "teacher_allocation": "历史资源缓存量",
    "raw_action": "缓存调整量",
    "teacher_raw_action": "历史缓存调整量",
    "baseline_action": "即时需求缺口",
    "reward": "综合评价值",
    "future_reward_mean": "未来综合评价均值",
    "hit_rate": "请求保障率",
    "waste_rate": "资源冗余率",
    "action_source": "决策来源",
    "retrieved_case_ids": "相似案例编号",
    "case_id": "案例编号",
    "score": "综合相似度",
    "state_similarity": "状态相似度",
    "trend_similarity": "负载趋势相似度",
    "quality_score": "案例质量分",
    "regime_tag": "负载状态类型",
    "decision_step": "历史时隙",
    "arrivals_mean": "近期平均请求到达量",
    "arrivals_slope": "请求到达量趋势",
    "waiting_slope": "等待服务队列趋势",
    "resource_slope": "空闲算力资源趋势",
    "query_regime": "当前负载状态",
    "regime_match": "状态类型匹配",
    "same_game_bonus": "任务类型匹配加权",
    "mean_return": "平均累计评价",
    "mean_reward": "平均单步评价",
    "mean_hit_rate": "平均请求保障率",
    "mean_waste_rate": "平均资源冗余率",
    "mean_allocation": "平均资源缓存量",
    "num_rows": "决策步数",
    "num_tasks": "任务序列数",
    "num_cases": "案例数",
    "num_games": "任务类型数",
    "count": "数量",
    "pc1": "案例特征维度1",
    "pc2": "案例特征维度2",
    "tool_name": "工具标识",
    "display_name": "工具名称",
    "resource_weight": "资源需求权重",
    "estimated_latency_level": "估计时延等级",
    "output_type": "输出类型",
}


SOURCE_LABELS = {
    "baseline": "即时规则分配",
    "heuristic": "启发式状态调节",
    "retriever": "相似案例缓存预测",
    "llm": "ICL少样本缓存决策",
    "llm_hold": "ICL决策保持",
}

TASK_DISPLAY_NAMES = {
    "快速问答服务": "快速问答服务",
    "受灾目标识别": "受灾目标识别",
    "结构化报告生成": "结构化报告生成",
    "灾害图像深度分析": "灾害图像深度分析",
    "危险源态势研判": "危险源态势研判",
    "快速问答业务": "快速问答业务",
    "任务规划业务": "任务规划业务",
    "语义分割业务": "语义分割业务",
    "灾害图像深度分析业务": "灾害图像深度分析业务",
    "结构化报告生成业务": "结构化报告生成业务",
    "结果回传业务": "结果回传业务",
}


def display_task_name(value: Any) -> str:
    text = str(value)
    legacy_prefix = "demo_" + "game_"
    if text.startswith(legacy_prefix):
        return "应急任务" + text.replace(legacy_prefix, "")
    return TASK_DISPLAY_NAMES.get(text, text)


def display_sequence_name(value: Any) -> str:
    text = str(value)
    legacy_prefix = "user_" + "seq_"
    if text.startswith(legacy_prefix):
        return "请求序列" + text.replace(legacy_prefix, "")
    return text


def ui_col(name: str) -> str:
    return UI_LABELS.get(str(name), str(name))


def ui_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.rename(columns={c: ui_col(c) for c in df.columns}).copy()
    if "任务类型" in out.columns:
        out["任务类型"] = out["任务类型"].map(display_task_name)
    if "请求序列" in out.columns:
        out["请求序列"] = out["请求序列"].map(display_sequence_name)
    if "案例覆盖状态" in out.columns:
        out["案例覆盖状态"] = out["案例覆盖状态"].map({"seen": "已覆盖", "unseen": "未覆盖", "cold_start": "冷启动"}).fillna(out["案例覆盖状态"])
    if "决策来源" in out.columns:
        out["决策来源"] = out["决策来源"].map(ui_source_label)
    return out


def ui_source_label(value: Any) -> str:
    text = str(value)
    if text in SOURCE_LABELS:
        return SOURCE_LABELS[text]
    if text.startswith("llm_disabled_fallback") or text.startswith("llm_exception_fallback") or text.startswith("llm_low_conf_fallback"):
        return "相似案例缓存预测回退"
    if text.startswith("llm_fallback"):
        return "ICL接口待确认"
    return text


def load_demo_requests() -> pd.DataFrame:
    return load_dataset(DEMO_DATA / "sample_requests.csv")


def load_demo_history_records(max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    path = DEMO_DATA / "sample_metatrain_history.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                rows.append(normalize_record(json.loads(line)))
    return rows


def chart_line(df: pd.DataFrame, x: str, y: List[str] | str, title: str = ""):
    y_cols = [y] if isinstance(y, str) else list(y)
    display_df = df.rename(columns={c: ui_col(c) for c in [x, *y_cols] if c in df.columns})
    x_label = ui_col(x)
    y_labels = [ui_col(c) for c in y_cols]
    y_arg: List[str] | str = y_labels[0] if isinstance(y, str) else y_labels
    if px is None:
        st.line_chart(display_df.set_index(x_label)[y_arg])
    else:
        fig = px.line(display_df, x=x_label, y=y_arg, title=title)
        style_figure(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)


def chart_bar(df: pd.DataFrame, x: str, y: str, title: str = ""):
    display_df = df.rename(columns={c: ui_col(c) for c in [x, y] if c in df.columns})
    x_label = ui_col(x)
    y_label = ui_col(y)
    if px is None:
        st.bar_chart(display_df.set_index(x_label)[y_label])
    else:
        fig = px.bar(display_df, x=x_label, y=y_label, title=title)
        style_figure(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)


def style_figure(fig, height: int = 360):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=48, b=24),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a", family="Arial, Microsoft YaHei, sans-serif"),
        title=dict(font=dict(size=17, color="#0b3a74")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=["#0b66c3", "#0ea5e9", "#f59e0b", "#16a34a", "#7c3aed", "#ef4444"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef4fb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef4fb", zeroline=False)
    return fig


def render_overview():
    st.markdown(
        """
<div class="hero-band">
  <div class="hero-eyebrow">O₂Cache</div>
  <div class="hero-title">应急智枢---面向灾害应急的多模态智能体辅助决策系统</div>
  <div class="hero-subtitle">基于历史案例库与上下文学习的应急服务资源缓存预测平台</div>
  <div class="hero-copy">本平台面向灾害响应场景下的多模态智能服务请求，展示如何通过历史案例库、相似案例检索和上下文学习机制，预测下一时隙资源缓存量，为模型实例预热、GPU/显存预留和并发资源分配提供辅助决策。</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.subheader("核心流程")
    render_flow(
        [
            "灾害任务请求接入",
            "任务负载状态建模",
            "历史案例库构建",
            "Top-K相似案例检索",
            "ICL少样本缓存决策",
            "资源缓存量与资源预热建议",
            "反馈评价与更新条件分析",
        ]
    )
    st.subheader("多模态智能体任务服务来源")
    section_note(
        "演示模式下，平台可通过工具调用轨迹模拟多模态任务服务负载：无人机 / 救援人员 / 应急中心提交多模态任务 → 智能体进行任务规划与工具调用 → 形成快速问答、语义分割、图像分析、报告生成等服务请求 → 服务请求进入 ICL 资源缓存预测模块 → 平台生成下一时隙资源缓存量与资源预热建议。该流程不表示真实无人机在线控制，也不默认运行真实大模型。"
    )

    st.subheader("关键能力")
    render_capability_cards(
        [
            ("多模态任务负载接入", "接入快速问答、受灾目标识别、结构化报告生成和灾害图像深度分析等任务负载。"),
            ("历史案例库构建", "从历史状态、请求窗口、缓存决策和反馈评价中沉淀可复用案例。"),
            ("相似案例检索与解释", "根据任务类型、负载趋势和系统状态检索 Top-K 相似历史案例。"),
            ("ICL资源缓存决策", "将当前状态与相似案例组织为上下文输入，形成下一时隙资源缓存建议。"),
        ]
    )

    st.subheader("演示指标")
    metrics = current_demo_metrics()
    render_metric_cards(
        [
            ("请求保障率", metrics["保障率"], metrics["说明"], False),
            ("资源冗余率", metrics["冗余率"], metrics["说明"], False),
            ("综合评价值", metrics["综合评价"], "数值越接近 0 表示当前评价函数下效果越优", True),
        ]
    )

    if OVERVIEW_IMAGE.exists():
        st.subheader("系统架构参考")
        st.image(str(OVERVIEW_IMAGE), use_container_width=True)

    with st.expander("流程说明", expanded=False):
        steps = pd.DataFrame(
            [
                ["灾害任务请求接入", "接入不同多模态任务服务的请求负载。"],
                ["任务负载状态建模", "根据请求到达量、空闲算力资源和等待服务队列形成当前时隙状态。"],
                ["历史案例库构建", "从历史决策轨迹中沉淀包含状态、动作和反馈评价的高质量案例。"],
                ["Top-K相似案例检索", "综合负载趋势、系统状态、任务类型和案例质量检索相似历史案例。"],
                ["ICL少样本缓存决策", "将相似历史案例与当前状态组织为上下文输入，生成下一时隙资源缓存量。"],
                ["资源缓存量与资源预热建议", "展示模型实例预热、GPU/显存预留和并发资源分配的辅助决策结果。"],
                ["反馈评价与更新条件分析", "展示综合评价值、请求保障率和资源冗余率，并分析新案例是否具备入库价值。"],
            ],
            columns=["阶段", "说明"],
        )
        st.dataframe(steps, use_container_width=True, hide_index=True)


def render_multimodal_agent_intake():
    page_title(
        "多模态任务接入",
        "展示灾害图像与文本请求如何通过演示智能体工具调用轨迹，转化为资源缓存预测所需的多类型任务负载。",
    )
    section_note(
        "本页为演示模式，不加载真实大模型权重，不执行真实图像分割或视觉推理。平台通过演示智能体工具调用轨迹，将多模态任务处理过程转化为资源缓存预测所需的任务负载。"
    )
    render_data_source_status("agent_intake")

    st.subheader("智能体任务演示")
    col1, col2 = st.columns([2, 1])
    with col1:
        user_task = st.text_area(
            "用户任务输入",
            value=DEFAULT_USER_TASK,
            height=90,
            help="输入灾害响应场景下的文本请求。演示模式会生成固定工具调用轨迹。",
        )
    with col2:
        image_name = st.text_input(
            "可选图片名称",
            value="灾害现场图像_示例",
            help="仅作为演示标识，不会读取或上传真实图像。",
        )
        run_agent = st.button("生成演示工具调用轨迹", type="primary")

    if run_agent or st.session_state.agent_tool_history is None:
        history = run_demo_agent(user_task=user_task, image_name=image_name)
        load_df = events_to_load_dataframe(history, sequence_id="智能体演示序列001")
        st.session_state.agent_tool_history = history
        st.session_state.agent_load_df = load_df

    tool_history = st.session_state.get("agent_tool_history") or []
    load_df = st.session_state.get("agent_load_df")
    if not tool_history or load_df is None:
        st.stop()

    st.markdown(
        "<div class='soft-panel'><strong>演示边界</strong><br>"
        "当前页面展示的是多模态智能体任务服务来源的轻量接入方式。真实 Qwen3-VL 与语义分割模型可作为高级模式扩展；"
        "资源缓存预测平台接收的是由工具调用轨迹生成的服务负载，不直接控制无人机飞行或真实云资源。</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("工具调用步骤", len(tool_history))
    c2.metric("任务类型", int(load_df["game_name"].nunique()))
    c3.metric("请求序列", int(load_df["sequence_id"].nunique()))
    c4.metric("最大资源权重", f"{float(load_df['resource_weight'].max()):.1f}")

    st.subheader("工具调用轨迹")
    event_df = chinese_event_frame(tool_history)
    st.dataframe(event_df, use_container_width=True, hide_index=True)

    with st.expander("各工具作用说明", expanded=True):
        for item in tool_history:
            st.markdown(
                f"**{item.get('step')}. {item.get('display_name')}**：{item.get('summary')} "
                f"对应任务类型为 `{item.get('task_type')}`，资源需求权重为 `{item.get('resource_weight')}`。"
            )

    st.subheader("结构化灾害分析摘要")
    st.markdown(
        """
- 任务规划步骤将用户请求拆分为语义分割、图像分析、报告生成和结果回传。
- 语义分割业务表示对灾害图像中的道路、水体、建筑受损区域进行结构化识别。
- 灾害图像深度分析业务表示对受灾区域、通行风险和救援优先级进行综合判断。
- 结构化报告生成业务表示将分析结果整理成面向应急中心的报告摘要。
- 这些工具调用会形成多类型云服务请求，可作为后续 ICL 资源缓存预测的输入负载。
"""
    )

    st.subheader("任务负载转化")
    st.caption("请求到达量由工具资源权重与工具调用次数生成，仅用于演示负载，不代表真实无人机数据或真实云资源监控数据。")
    load_display = chinese_load_frame(load_df)
    st.dataframe(load_display, use_container_width=True, hide_index=True)
    if px:
        fig = px.line(load_display, x="时间步", y="请求到达量", color="任务类型", markers=True, title="智能体工具调用转化的演示请求负载")
        style_figure(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(load_df.set_index("timestamp")["arrivals"])

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("将该负载送入 ICL 资源缓存预测流程", type="primary"):
            set_request_source(
                load_df,
                kind=SOURCE_AGENT,
                label=AGENT_SOURCE_LABEL,
                note="演示模式下由智能体工具调用轨迹生成的多类型任务服务负载。",
                metadata={"tool_history": tool_history},
            )
            st.success("已接入智能体演示负载。请继续进入“历史案例库”构建案例库，或进入“ICL缓存决策”运行预测。")
    with col2:
        st.info("送入后，平台主线仍保持：历史案例库构建 → 相似案例检索 → ICL少样本缓存决策 → 调度效果评估 → 单步可解释分析 → 结果导出。")


def render_data_import():
    page_title("灾害任务接入", "接入灾害场景下多模态任务服务的请求负载，并完成基础质量检查。")
    render_data_source_status("data_import")
    section_note(
        "这些数据用于模拟灾害响应场景下不同智能服务的请求负载，包括快速问答、受灾目标识别、结构化报告生成、灾害图像深度分析等业务类型。"
    )
    with st.expander("上传文件格式说明", expanded=False):
        st.markdown(
            """
任务负载文件支持 `CSV`、`JSON`、`JSONL`。最小字段包括：时间步、请求到达量、请求序列、任务类型。

平台兼容字段名：`log_time`、`num`、`access_key_id`、`game_name`；也可使用 `timestamp`、`requests`、`sequence_id`、`task_type` 等别名。完整说明见 `docs/数据格式与参数说明.md`。
"""
        )
    mode_options = ["使用内置演示任务负载", "上传任务负载文件", "高级：读取本机数据文件"]
    if st.session_state.get("request_df") is not None and current_source_kind() != SOURCE_BUILTIN:
        mode_options = ["继续使用当前任务负载", *mode_options]
    mode = st.radio(
        "任务负载来源",
        mode_options,
        horizontal=True,
        help="选择本次演示使用内置灾害任务负载，还是加载自定义任务负载文件。",
    )
    df = None
    source_kind = SOURCE_BUILTIN
    source_label = BUILTIN_SOURCE_LABEL
    source_note = "平台内置灾害任务负载样例。"
    source_meta: Dict[str, Any] = {}
    if mode.startswith("继续"):
        df = get_current_request_df()
        source_kind = current_source_kind()
        source_label = current_source_label()
        source_note = str(st.session_state.get("request_source_note", ""))
        source_meta = dict(st.session_state.get("request_source_metadata", {}))
    elif mode.startswith("使用"):
        df = load_demo_requests()
        source_kind = SOURCE_BUILTIN
        source_label = BUILTIN_SOURCE_LABEL
        source_note = "平台内置灾害任务负载样例。"
    elif mode.startswith("上传"):
        f = st.file_uploader(
            "上传任务负载文件",
            type=["csv", "jsonl", "json"],
            help="上传描述灾害任务请求到达量的文件，至少包含时间步、请求到达量、请求序列和任务类型。",
        )
        if f is not None:
            df = load_dataset(f)
            source_kind = SOURCE_UPLOAD
            source_label = UPLOAD_SOURCE_LABEL
            source_note = f"用户上传文件：{getattr(f, 'name', '任务负载文件')}"
            source_meta = {"file_name": getattr(f, "name", "")}
    else:
        path = st.text_input(
            "本机数据文件路径",
            value="",
            placeholder="选择已准备的任务负载文件路径",
            help="读取本机已有任务负载文件，适合队内调试或固定演示环境。",
        )
        if path and Path(path).exists():
            df = load_dataset(path)
            source_kind = SOURCE_LOCAL
            source_label = LOCAL_SOURCE_LABEL
            source_note = "从本机路径读取的任务负载文件。"
            source_meta = {"path": path}
        elif path:
            st.warning("路径不存在。")
    if df is None:
        st.stop()
    report = validate_dataset(df)
    should_clear = (
        st.session_state.get("request_source_kind") != source_kind
        or st.session_state.get("request_source_note") != source_note
        or st.session_state.get("request_df") is None
    )
    set_request_source(
        df,
        kind=source_kind,
        label=source_label,
        note=source_note,
        metadata=source_meta,
        clear_outputs=should_clear,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("记录数", report.summary.get("num_rows", 0))
    c2.metric("任务类型", report.summary.get("num_game_types", 0))
    c3.metric("请求序列", report.summary.get("num_sequences", 0))
    c4.metric("最大请求到达量", f"{report.summary.get('arrival_max',0):.2f}")
    if report.missing_fields:
        st.error("任务负载数据缺少必要信息，请检查时间步、请求到达量、请求序列和任务类型是否完整。")
    if report.warnings:
        for w in report.warnings:
            st.warning(str(w).replace("log_time", "时间步").replace("num", "请求到达量"))
    if report.ok:
        st.success("数据字段校验通过。")

    with st.expander("字段识别结果（高级）", expanded=False):
        recognized = {ui_col(k): v for k, v in report.mapped_columns.items()}
        st.write("平台已识别任务类型、请求序列、时间步与请求到达量等基础信息。")
        st.json(recognized)

    st.subheader("任务负载预览")
    try:
        preview_df = prepare_request_dataframe(df).head(30)
    except Exception:
        preview_df = df.head(30)
    st.dataframe(ui_frame(preview_df), use_container_width=True)
    if report.ok:
        pdf = prepare_request_dataframe(df)
        stats = dataset_statistics(pdf)
        st.subheader("负载概况")
        render_metric_cards(
            [
                ("任务类型数", str(stats.get("game_types", 0)), "当前接入的多模态服务类别数量", False),
                ("请求序列数", str(stats.get("sequences", 0)), "用于演示连续负载变化的请求序列", False),
                ("平均请求到达量", format_metric(stats.get("mean_arrivals"), digits=2), "各时隙请求到达量均值", False),
                ("最大请求到达量", format_metric(stats.get("max_arrivals"), digits=2), "当前任务负载中的峰值请求量", True),
            ]
        )
        col1, col2 = st.columns(2)
        with col1:
            vc = pdf["game_name"].value_counts().reset_index()
            vc.columns = ["game_name", "rows"]
            chart_bar(vc, "game_name", "rows", "任务类型记录数分布")
        with col2:
            seq_len = pdf.groupby(["game_name", "access_key_id"]).size().reset_index(name="length")
            if px:
                fig = px.histogram(ui_frame(seq_len), x="序列长度", nbins=20, title="请求序列长度分布")
                style_figure(fig, height=340)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(seq_len["length"])
        st.subheader("请求负载时间序列样例")
        one = pdf.groupby(["game_name", "access_key_id"], sort=False).head(80)
        if px:
            one = one.copy()
            one["series"] = one["game_name"].map(display_task_name) + " / " + one["access_key_id"].map(display_sequence_name)
            fig = px.line(ui_frame(one), x="时间步", y="请求到达量", color="任务序列", title="典型请求序列的到达量变化")
            style_figure(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(one[["log_time", "num"]].set_index("log_time"))


def render_reward_workbench():
    page_title("调度效果评估", "分析不同资源缓存量对等待服务队列、资源冗余和综合评价值的影响。")
    render_data_source_status("reward")
    section_note(
        "本页围绕请求保障率、资源冗余率和综合评价值展示调度效果。策略对比仅基于当前接入的演示数据计算，不外推到未测试场景。"
    )
    if is_agent_source():
        st.info("本次评估基于多模态智能体工具调用生成的任务服务负载。")

    df_req = get_current_request_df()
    lib = st.session_state.get("case_library")
    st.subheader("策略结果对比")
    max_compare_steps = st.number_input("每条序列对比步数", 20, 500, 120, step=20, help="策略对比时每条任务序列运行的时间步数量。")
    if st.button("生成策略对比", type="primary"):
        compare_specs = [
            ("即时规则分配基线", "Baseline"),
            ("启发式状态调节基线", "Rule-Heuristic"),
        ]
        if lib is not None and getattr(lib, "cases", []):
            compare_specs.extend([
                ("相似案例缓存预测（消融基线）", "retriever"),
                ("ICL少样本缓存决策（本地案例演示）", "llm"),
            ])
        compare_rows = []
        for label, strategy in compare_specs:
            cfg = StrategyConfig(strategy=strategy, window=5, k=5)
            result = run_strategy_on_requests(df_req, config=cfg, case_library=lib if lib is not None else None, llm_client=None, max_steps_per_sequence=int(max_compare_steps))
            row = dict(result.get("summary", {}))
            row["method_label"] = label
            compare_rows.append(row)
        st.session_state["comparison_results"] = compare_rows

    compare_rows = st.session_state.get("comparison_results", [])
    if compare_rows:
        comp = pd.DataFrame(compare_rows)
        show_cols = ["method_label", "mean_reward", "mean_hit_rate", "mean_waste_rate", "mean_allocation", "num_rows"]
        comp_display = comp[[c for c in show_cols if c in comp.columns]].rename(columns={"method_label": "方法"})
        st.dataframe(ui_frame(comp_display), use_container_width=True, hide_index=True)
        if px:
            metric_df = ui_frame(comp_display)
            fig = px.bar(metric_df, x="方法", y=["平均单步评价", "平均请求保障率", "平均资源冗余率"], barmode="group", title="当前演示数据下的策略指标对比")
            style_figure(fig, height=380)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("点击“生成策略对比”后，将基于当前任务负载计算各策略的演示指标。")

    st.subheader("单状态敏感性分析")
    col = st.columns(3)
    resource = col[0].number_input("空闲算力资源", min_value=0.0, value=10.0, step=1.0, help="单状态敏感性分析中的当前可用资源。")
    waiting = col[1].number_input("等待服务队列", min_value=0.0, value=0.0, step=1.0, help="单状态敏感性分析中的当前积压请求量。")
    arrivals = col[2].number_input("当前请求到达量", min_value=0.0, value=30.0, step=1.0, help="单状态敏感性分析中的新增请求量。")
    col = st.columns(5)
    max_allocation = col[0].number_input("单时隙最大资源缓存量", min_value=1.0, value=800.0, step=10.0, help="单状态分析中允许的最大缓存量。")
    residual_scale = col[1].number_input("缓存调整尺度", min_value=1.0, value=200.0, step=10.0, help="影响缓存调整量转换为资源量的幅度。")
    waiting_penalty = col[2].number_input("等待惩罚系数", min_value=0.0, value=3.0, step=0.5, help="等待队列越长，对综合评价的惩罚越强。")
    idle_penalty = col[3].number_input("冗余惩罚系数", min_value=0.0, value=1.0, step=0.5, help="资源冗余越高，对综合评价的惩罚越强。")
    reward_scale = col[4].number_input("评价尺度", min_value=1.0, value=100.0, step=10.0, help="综合评价值的缩放参数，便于观察曲线变化。")

    baseline = myopic_base_allocation(resource, waiting, arrivals, max_allocation=max_allocation)
    st.metric("即时需求缺口", f"{baseline:.2f}")
    raw_values = np.linspace(-1, 1, 101)
    rows = [
        evaluate_raw_action(
            raw_action=float(r),
            resource_queue=resource,
            waiting_queue=waiting,
            arrivals=arrivals,
            residual_scale=residual_scale,
            max_allocation=max_allocation,
            waiting_penalty=waiting_penalty,
            idle_penalty=idle_penalty,
            reward_scale=reward_scale,
        )
        for r in raw_values
    ]
    df = pd.DataFrame(rows)
    chart_line(df, "raw_action", ["reward", "hit_rate", "waste_rate"], "不同缓存调整量下的综合评价、请求保障率和资源冗余率")
    chart_line(df, "raw_action", ["allocation", "next_waiting_queue", "next_resource_queue"], "不同缓存调整量下的资源缓存量与下一时隙状态")
    st.dataframe(ui_frame(df.round(4)), use_container_width=True)


def render_case_library():
    page_title("历史案例库", "从历史决策轨迹中构建包含任务负载状态、资源缓存动作和反馈评价的案例库。")
    render_data_source_status("case_library")
    section_note(
        "每个历史案例由历史状态、近期请求窗口、资源缓存决策和反馈评价组成。高质量案例会作为 ICL 缓存决策的上下文参考，帮助当前时隙形成更有依据的资源缓存建议。"
    )
    if is_agent_source():
        st.info("当前案例库构建基于智能体工具调用产生的多模态任务负载演示数据。该数据用于演示工具调用如何进入资源缓存预测流程，不代表真实无人机在线数据。")
    with st.expander("历史决策轨迹格式说明", expanded=False):
        st.markdown(
            """
历史决策轨迹推荐使用 `JSONL`，每行表示一个历史时隙。记录中应包含任务类型、请求序列、当前状态、资源缓存量、请求保障率、资源冗余率和综合评价值等信息。

平台会按连续轨迹切分历史案例，并使用后续反馈评价计算案例质量。完整字段说明见 `docs/数据格式与参数说明.md`。
"""
        )
    mode_options = ["使用内置历史决策轨迹", "上传历史决策轨迹", "高级：读取本机历史轨迹"]
    if is_agent_source():
        mode_options = ["根据当前智能体负载生成演示历史轨迹", *mode_options]
    mode = st.radio(
        "历史决策轨迹来源",
        mode_options,
        horizontal=True,
        help="选择用于构建案例库的历史决策轨迹来源。",
    )
    max_lines = st.number_input(
        "最多读取记录数（0 表示不限制）",
        min_value=0,
        value=2500,
        step=500,
        help="限制读取的历史轨迹行数，用于控制构建速度和案例库规模。",
    )
    window = st.slider("最近负载窗口长度", 2, 20, 5, help="构建案例时回看多少个历史时隙，用于描述近期负载趋势。")
    future_horizon = st.slider("反馈评价观察窗口", 1, 20, 5, help="评估一个历史决策后续影响的时间范围，用于计算案例质量。")
    max_cases = st.number_input(
        "最大案例数（0 表示不限制）",
        min_value=0,
        value=0,
        step=1000,
        help="限制最终进入案例库的案例数量，案例过多时可提高页面响应速度。",
    )

    records: Optional[List[Dict[str, Any]]] = None
    if mode.startswith("根据当前智能体"):
        current_df = get_current_request_df()
        records = build_naive_state_rows(prepare_request_dataframe(current_df))
    elif mode.startswith("使用"):
        records = load_demo_history_records(max_lines=None if max_lines == 0 else int(max_lines))
    elif mode.startswith("上传"):
        f = st.file_uploader(
            "上传历史决策轨迹",
            type=["jsonl"],
            help="上传历史决策轨迹文件，每一行应为一个 JSON 记录。",
        )
        if f is not None:
            content = f.read().decode("utf-8")
            records = [normalize_record(json.loads(line)) for i, line in enumerate(content.splitlines()) if line.strip() and (max_lines == 0 or i < max_lines)]
    else:
        path = st.text_input(
            "本机历史轨迹文件路径",
            value="",
            placeholder="选择已准备的历史决策轨迹文件路径",
            help="读取本机已有历史决策轨迹，适合固定演示环境。",
        )
        if path and Path(path).exists():
            records = []
            with Path(path).open("r", encoding="utf-8") as fp:
                for i, line in enumerate(fp):
                    if max_lines and i >= max_lines:
                        break
                    if line.strip():
                        records.append(normalize_record(json.loads(line)))
        elif path:
            st.warning("路径不存在。")
    if records is None:
        st.stop()

    if st.button("构建/刷新历史案例库", type="primary") or st.session_state.case_library is None:
        lib = CaseLibrary(window=int(window), future_horizon=int(future_horizon), max_cases=None if max_cases == 0 else int(max_cases))
        lib.build_from_records(records)
        st.session_state.history_records = records
        st.session_state.case_library = lib
    lib: CaseLibrary = st.session_state.case_library
    if lib is None or not lib.cases:
        st.warning("当前未构建出案例。请检查历史轨迹是否足够长，且最近负载窗口长度是否过大。")
        st.stop()
    stats = lib.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("案例数", stats.get("num_cases", 0))
    c2.metric("任务类型", stats.get("num_games", 0))
    c3.metric("平均质量", f"{stats.get('quality_mean',0):.3f}")
    c4.metric("平均综合评价", f"{stats.get('reward_mean',0):.3f}")

    df = lib.to_frame()
    case_display_cols = [
        "case_id",
        "game_name",
        "access_key_id",
        "decision_step",
        "regime_tag",
        "teacher_allocation",
        "reward",
        "hit_rate",
        "waste_rate",
        "quality_score",
        "arrivals_mean",
        "arrivals_slope",
        "waiting_slope",
        "resource_slope",
    ]
    case_display_df = df[[c for c in case_display_cols if c in df.columns]].copy()
    tab1, tab2, tab3 = st.tabs(["统计图", "案例地图", "案例表"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            reg = df["regime_tag"].value_counts().reset_index()
            reg.columns = ["regime_tag", "count"]
            chart_bar(reg, "regime_tag", "count", "负载状态类型分布")
        with col2:
            game = df["game_name"].value_counts().head(20).reset_index()
            game.columns = ["game_name", "count"]
            chart_bar(game, "game_name", "count", "任务类型分布（前20）")
        col1, col2 = st.columns(2)
        with col1:
            if px:
                fig = px.histogram(ui_frame(df), x="案例质量分", nbins=30, title="案例质量分布")
                style_figure(fig, height=330)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            if px:
                fig = px.histogram(ui_frame(df), x="历史资源缓存量", nbins=30, title="历史资源缓存量分布")
                style_figure(fig, height=330)
                st.plotly_chart(fig, use_container_width=True)
    with tab2:
        map_df = case_map_frame(lib)
        if px:
            map_display = ui_frame(map_df)
            map_display["案例点大小"] = np.maximum(map_df["quality_score"].astype(float) + 1.1, 0.1)
            fig = px.scatter(map_display, x="案例特征维度1", y="案例特征维度2", color="负载状态类型", size="案例点大小", hover_data=["案例编号", "任务类型", "案例质量分", "历史资源缓存量"], title="案例库特征地图")
            style_figure(fig, height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(ui_frame(map_df[["pc1", "pc2", "regime_tag", "quality_score"]]))
    with tab3:
        st.dataframe(ui_frame(case_display_df), use_container_width=True)
        csv = ui_frame(case_display_df).to_csv(index=False, encoding="utf-8-sig")
        st.download_button("导出历史案例表", csv, file_name="O2Cache_历史案例表.csv", mime="text/csv")


def render_retrieval():
    page_title("相似案例检索", "展示当前任务状态如何检索 Top-K 相似历史案例，并解释相似度来源。")
    render_data_source_status("retrieval")
    section_note(
        "Top-K 相似历史案例是当前资源缓存决策的重要依据。平台会同时比较任务类型、近期请求趋势、空闲算力资源、等待服务队列和历史案例质量。"
    )
    lib: Optional[CaseLibrary] = st.session_state.get("case_library")
    if lib is None or not lib.cases:
        st.warning("请先在“历史案例库”页面构建案例库。")
        st.stop()
    df_req = get_current_request_df()
    df_req = prepare_request_dataframe(df_req)
    series_options = df_req.groupby(["game_name", "access_key_id"]).size().reset_index()[["game_name", "access_key_id"]]
    labels = [f"{display_task_name(r.game_name)} / {display_sequence_name(r.access_key_id)}" for r in series_options.itertuples()]
    selected = st.selectbox("选择任务请求序列", labels, help="选择当前要分析的灾害任务序列。")
    idx = labels.index(selected)
    game = str(series_options.iloc[idx]["game_name"])
    seq = str(series_options.iloc[idx]["access_key_id"])
    seq_df = df_req[(df_req["game_name"] == game) & (df_req["access_key_id"] == seq)].reset_index(drop=True)
    if len(seq_df) < 3:
        st.warning("当前请求序列过短，暂不适合进行相似案例检索。请接入至少 3 个时间步的任务负载。")
        st.stop()
    max_window = max(2, min(20, len(seq_df) - 1))
    window = st.slider("最近负载窗口长度", 2, max_window, min(5, max_window), key="ret_window", help="当前检索时回看的近期请求窗口，影响负载趋势相似度。")
    step = st.slider("选择当前时隙", int(window), len(seq_df) - 1, min(int(window), len(seq_df) - 1), help="指定当前决策发生在哪个时间步。")
    k = st.slider("Top-K", 1, 12, 5, help="返回的相似历史案例数量。")
    # Use baseline reconstruction for recent rows.
    baseline_rows = build_naive_state_rows(seq_df.iloc[: step + 1])
    current = baseline_rows[-1]["state_summary"]
    recent = baseline_rows[max(0, len(baseline_rows) - window - 1) : -1]
    cases = lib.query(game_name=game, recent_window=recent, current_state_summary=current, k=k)
    raw = lib.fallback_raw_action(game_name=game, recent_window=recent, current_state_summary=current, k=k)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前请求到达量", f"{current['current_arrivals']:.2f}")
    col2.metric("等待服务队列", f"{current['waiting_queue']:.2f}")
    col3.metric("即时需求缺口", f"{current['baseline_action']:.2f}")
    col4.metric("相似案例缓存调整量", f"{raw:.3f}")
    if is_agent_source():
        tool_name = agent_tool_display_for_task(game)
        if tool_name:
            st.info(f"当前任务来源工具：{tool_name}。")

    st.subheader("Top-K 相似历史案例")
    case_table = []
    for c in cases:
        case_table.append({
            "case_id": c.get("case_id"),
            "game_name": c.get("game_name"),
            "score": c.get("score"),
            "state_similarity": c.get("state_similarity"),
            "trend_similarity": c.get("trend_similarity"),
            "quality_score": c.get("quality_score"),
            "regime_tag": c.get("regime_tag"),
            "regime_match": c.get("regime_match"),
            "same_game_bonus": c.get("same_game_bonus"),
            "teacher_raw_action": c.get("teacher_raw_action"),
            "teacher_allocation": c.get("teacher_allocation"),
            "future_reward_mean": c.get("future_outcome", {}).get("reward_mean"),
        })
    cdf = pd.DataFrame(case_table)
    st.dataframe(ui_frame(cdf), use_container_width=True)
    if cases:
        st.markdown(
            "<div class='soft-panel'><strong>为什么这些案例相似</strong><br>"
            "检索分数综合了当前系统状态、近期请求趋势、负载状态类型、任务类型匹配和案例质量。"
            "状态相似度用于比较空闲算力资源与等待服务队列，负载趋势相似度用于比较近期请求变化形态，案例质量用于优先选择反馈评价较好的历史经验。</div>",
            unsafe_allow_html=True,
        )
    if not cdf.empty and px:
        cdf_display = ui_frame(cdf)
        fig = px.bar(cdf_display, x="案例编号", y=["状态相似度", "负载趋势相似度", "案例质量分"], title="相似度组成")
        style_figure(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("当前负载窗口与相似历史窗口对比")
    frames = [line_df_for_window("当前负载窗口", recent)]
    for i, c in enumerate(cases[:3], start=1):
        frames.append(line_df_for_window(f"相似案例 {i}: {c.get('case_id')}", c.get("recent_window", [])))
    wdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not wdf.empty and px:
        wdf_display = ui_frame(wdf.rename(columns={"label": "窗口"}))
        fig = px.line(wdf_display, x="时隙", y="请求到达量", color="窗口", title="请求到达量窗口对比")
        style_figure(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("当前状态详情"):
        st.dataframe(ui_frame(pd.DataFrame([current])), use_container_width=True, hide_index=True)


def _make_llm_client_from_ui() -> Optional[OpenAICompatibleClient]:
    enable = st.checkbox(
        "启用外部 ICL 推理模型接口（高级配置）",
        value=False,
        help="外部接口只用于演示 ICL 决策过程；不启用时平台仍可完成本地演示。",
    )
    if not enable:
        return None
    st.caption("外部模型接口用于演示检索增强的 ICL 决策过程；未配置时，平台使用本地案例检索与规则化推理完成演示。")
    base_url = st.text_input("服务地址（高级）", value="https://api.openai.com/v1", help="兼容 OpenAI 接口格式的服务地址。")
    model = st.text_input("模型名称（高级）", value="gpt-4o-mini", help="外部 ICL 推理模型名称。")
    api_key = st.text_input("访问密钥（仅本次会话使用，不保存）", type="password", help="访问密钥仅保存在当前会话中，不写入文件或日志。")
    force_json = st.checkbox("强制结构化输出", value=False, help="要求外部接口返回结构化结果，便于解析资源缓存建议。")
    if not api_key:
        st.warning("未填写访问密钥时，将自动回退到相似案例缓存预测，平台仍可继续演示。")
        return None
    return OpenAICompatibleClient(base_url=base_url, api_key=api_key, model=model, force_json_mode=force_json)


def render_prediction():
    page_title("ICL缓存决策", "在同一任务负载上选择不同资源缓存方法，生成下一时隙资源缓存量与资源预热建议。")
    render_data_source_status("prediction")
    st.markdown(
        "<div class='decision-chain'><strong>决策链路：</strong>"
        "当前状态 + Top-K相似案例 + 上下文输入模板 → ICL少样本缓存决策 → 资源缓存量与资源预热建议。"
        "未配置外部 ICL 推理模型时，平台使用本地相似案例缓存预测完成演示。</div>",
        unsafe_allow_html=True,
    )
    if is_agent_source():
        st.info("当前预测基于多模态智能体工具调用生成的任务服务负载。")
    df_req = get_current_request_df()
    try:
        state_preview = build_naive_state_rows(prepare_request_dataframe(df_req).head(12))
        if state_preview:
            with st.expander("当前状态与上下文输入模板", expanded=False):
                st.markdown("**当前状态**")
                st.dataframe(ui_frame(pd.DataFrame([state_preview[0].get("state_summary", {})])), use_container_width=True, hide_index=True)
                st.markdown("**上下文输入模板**")
                st.code("当前状态 + Top-K相似历史案例 + 反馈评价摘要 → 预测下一时隙资源缓存量", language="text")
    except Exception:
        pass
    lib = st.session_state.get("case_library")
    if lib is None:
        st.info("当前尚未构建案例库。即时规则与启发式基线可以运行；相似案例缓存预测与 ICL 少样本缓存决策需要先构建案例库。")
    method_label = st.selectbox("方法", list(METHOD_OPTIONS.keys()), index=3, help="选择本次资源缓存量的计算方式。")
    strategy = METHOD_OPTIONS[method_label]
    st.caption(METHOD_DESCRIPTIONS[method_label])
    col = st.columns(5)
    window = col[0].number_input("最近负载窗口", 2, 20, 5, help="预测时回看的请求负载长度，用于形成当前上下文状态。")
    k = col[1].number_input("Top-K", 1, 20, 5, help="ICL 决策参考的相似历史案例数量。")
    max_steps = col[2].number_input("每条序列最多步数", 5, 1000, 120, help="每条任务序列最多运行的时间步数量，用于控制演示耗时。")
    waiting_penalty = col[3].number_input("等待惩罚系数", 0.0, 10.0, 3.0, 0.5, help="等待服务队列对综合评价的惩罚强度。")
    idle_penalty = col[4].number_input("冗余惩罚系数", 0.0, 10.0, 1.0, 0.5, help="资源冗余对综合评价的惩罚强度。")
    col = st.columns(4)
    initial_resource = col[0].number_input("初始空闲算力资源", 0.0, 10000.0, 10.0, help="每条序列开始时的可用服务能力。")
    initial_waiting = col[1].number_input("初始等待服务队列", 0.0, 10000.0, 0.0, help="每条序列开始时已经积压的请求量。")
    residual_scale = col[2].number_input("缓存调整尺度", 1.0, 2000.0, 200.0, help="将缓存调整量转换为实际资源量的尺度。")
    max_allocation = col[3].number_input("单时隙最大资源缓存量", 1.0, 10000.0, 800.0, help="单个时间步允许建议的最大资源缓存量。")
    llm_client = None
    if strategy == "llm":
        with st.expander("ICL推理模型接口（高级配置）", expanded=False):
            llm_client = _make_llm_client_from_ui()
    if strategy in {"retriever", "llm"} and (lib is None or not getattr(lib, "cases", [])):
        st.warning("相似案例检索与 ICL 方法需要先构建历史案例库。")
    run = st.button("运行预测", type="primary")
    if run:
        cfg = StrategyConfig(
            strategy=strategy,
            window=int(window),
            k=int(k),
            residual_scale=float(residual_scale),
            max_allocation=float(max_allocation),
            initial_resource=float(initial_resource),
            initial_waiting=float(initial_waiting),
            waiting_penalty=float(waiting_penalty),
            idle_penalty=float(idle_penalty),
            reward_scale=100.0,
        )
        with st.spinner("正在执行资源缓存预测……"):
            result = run_strategy_on_requests(df_req, config=cfg, case_library=lib if lib is not None else None, llm_client=llm_client, max_steps_per_sequence=int(max_steps))
        result["summary"]["method_label"] = method_label
        st.session_state.last_result = result
        outdir = DEMO_OUTPUTS / f"last_{strategy.lower().replace('+','_').replace('-','_')}"
        save_run_outputs(outdir, summary=result["summary"], trace=result["trace"])
        st.session_state.last_run_dir = str(outdir)
        st.success("资源缓存决策已完成，结果已准备导出。")
    result = st.session_state.get("last_result")
    if result is None:
        st.stop()
    summary = result["summary"]
    trace = result["trace"]
    st.markdown(f"**当前方法：** {summary.get('method_label', summary.get('strategy', ''))}")
    cols = st.columns(5)
    cols[0].metric("平均累计评价", f"{summary.get('mean_return',0):.3f}")
    cols[1].metric("平均单步评价", f"{summary.get('mean_reward',0):.3f}")
    cols[2].metric("平均请求保障率", f"{summary.get('mean_hit_rate',0):.3f}")
    cols[3].metric("平均资源冗余率", f"{summary.get('mean_waste_rate',0):.3f}")
    cols[4].metric("平均资源缓存量", f"{summary.get('mean_allocation',0):.2f}")
    st.markdown(
        "<div class='soft-panel'><strong>资源预热建议说明</strong><br>"
        "资源缓存量用于辅助判断下一时隙应预留的服务能力，可映射到模型实例预热、GPU/显存预留和并发配额调整。"
        "页面展示的是当前演示数据和所选方法下的计算结果。</div>",
        unsafe_allow_html=True,
    )
    tdf = trace_to_frame(trace)
    if not tdf.empty:
        display_tdf = tdf.copy()
        display_tdf["action_source"] = display_tdf["action_source"].map(ui_source_label)
        chart_line(display_tdf.reset_index(), "index", ["allocation", "waiting_queue", "next_waiting_queue"], "资源缓存量与等待服务队列变化")
        chart_line(display_tdf.reset_index(), "index", ["reward", "hit_rate", "waste_rate"], "综合评价、请求保障率与资源冗余率")
        st.subheader("决策来源统计")
        src = display_tdf["action_source"].value_counts().reset_index()
        src.columns = ["action_source", "count"]
        chart_bar(src, "action_source", "count", "决策来源分布")
        st.subheader("逐步结果表")
        decision_df = decision_export_frame(display_tdf)
        st.dataframe(decision_df, use_container_width=True)
        st.download_button("导出运行结果", decision_df.to_csv(index=False, encoding="utf-8-sig"), file_name="O2Cache_运行结果.csv", mime="text/csv")


def render_step_explanation():
    page_title("单步可解释分析", "复盘某一时隙资源缓存决策的任务状态、相似案例依据、决策来源和反馈评价。")
    render_data_source_status("step_explanation")
    result = st.session_state.get("last_result")
    if result is None:
        st.warning("请先在“ICL缓存决策”页面运行一次预测。")
        st.stop()
    trace = result.get("trace", [])
    if not trace:
        st.stop()
    idx = st.slider("选择时隙", 0, len(trace) - 1, 0, help="选择已运行结果中的某一个时间步，用于查看该步的状态、相似案例和反馈评价。")
    row = trace[idx]
    if is_agent_source():
        tool_name = agent_tool_display_for_task(row.get("game_name"))
        if tool_name:
            st.info(f"该时隙所属任务类型：{display_task_name(row.get('game_name'))}；来源工具：{tool_name}。")
    s = row.get("state_summary", {})
    ns = row.get("next_state_summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("请求到达量", f"{safe_float(s.get('current_arrivals',0)):.2f}")
    c2.metric("等待服务队列", f"{safe_float(s.get('waiting_queue',0)):.2f} → {safe_float(ns.get('waiting_queue',0)):.2f}")
    c3.metric("资源缓存量", f"{safe_float(row.get('allocation',0)):.2f}")
    c4.metric("综合评价值", f"{safe_float(row.get('reward',0)):.3f}")
    st.subheader("状态与决策")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**当前状态**")
        st.dataframe(ui_frame(pd.DataFrame([s])), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**执行决策与反馈**")
        feedback = {k: row.get(k) for k in ["raw_action", "allocation", "reward", "hit_rate", "waste_rate", "action_source"]}
        feedback["action_source"] = ui_source_label(feedback.get("action_source"))
        st.dataframe(ui_frame(pd.DataFrame([feedback])), use_container_width=True, hide_index=True)
    st.subheader("相似历史案例")
    retrieved = row.get("retrieved_cases", [])
    if retrieved:
        small = pd.DataFrame([
            {
                "case_id": c.get("case_id"),
                "score": c.get("score"),
                "state_similarity": c.get("state_similarity"),
                "trend_similarity": c.get("trend_similarity"),
                "quality_score": c.get("quality_score"),
                "teacher_raw_action": c.get("teacher_raw_action"),
                "teacher_allocation": c.get("teacher_allocation"),
                "regime_tag": c.get("regime_tag"),
            }
            for c in retrieved
        ])
        st.dataframe(ui_frame(small), use_container_width=True)
    else:
        st.info("该时隙未记录相似案例明细，可能为即时规则分配基线、启发式状态调节基线，或尚未启用案例库。")
    if row.get("llm_result") is not None:
        with st.expander("ICL推理模型输出详情（高级）", expanded=False):
            st.json(row.get("llm_result"))


def render_result_replay():
    page_title("结果导出", "导出本次演示形成的运行结果、策略对比结果、单步决策解释和演示报告摘要。")
    render_data_source_status("result_export")
    result = st.session_state.get("last_result")
    df_req = get_current_request_df()
    lib = st.session_state.get("case_library")
    comparison_rows = st.session_state.get("comparison_results", [])

    if result is None:
        st.info("尚未生成资源缓存决策结果。请先在“ICL缓存决策”页面运行一次预测。")
    else:
        summary = dict(result.get("summary", {}))
        trace = result.get("trace", [])
        tdf = trace_to_frame(trace)
        if not tdf.empty:
            decision_df = decision_export_frame(tdf)
            st.subheader("运行结果")
            cols = st.columns(4)
            cols[0].metric("平均单步评价", f"{summary.get('mean_reward', 0):.3f}")
            cols[1].metric("平均请求保障率", f"{summary.get('mean_hit_rate', 0):.3f}")
            cols[2].metric("平均资源冗余率", f"{summary.get('mean_waste_rate', 0):.3f}")
            cols[3].metric("平均资源缓存量", f"{summary.get('mean_allocation', 0):.2f}")
            st.dataframe(decision_df.head(80), use_container_width=True)
            st.download_button("导出运行结果", decision_df.to_csv(index=False, encoding="utf-8-sig"), file_name="O2Cache_运行结果.csv", mime="text/csv")

            explanation_cols = ["t", "game_name", "access_key_id", "current_arrivals", "allocation", "reward", "hit_rate", "waste_rate", "action_source", "retrieved_case_ids"]
            explanation_df = tdf[[c for c in explanation_cols if c in tdf.columns]].copy()
            explanation_df = decision_export_frame(explanation_df)
            st.download_button("导出单步决策解释", explanation_df.to_csv(index=False, encoding="utf-8-sig"), file_name="O2Cache_单步决策解释.csv", mime="text/csv")

    st.subheader("策略对比结果")
    comp_df = comparison_export_frame(comparison_rows)
    if not comp_df.empty:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.download_button("导出策略对比结果", comp_df.to_csv(index=False, encoding="utf-8-sig"), file_name="O2Cache_策略对比结果.csv", mime="text/csv")
    else:
        st.info("尚未生成策略对比结果。可在“调度效果评估”页面点击“生成策略对比”。")

    if result is not None:
        report_md = build_demo_report_markdown(result=result, comparison_rows=comparison_rows, df_req=df_req, lib=lib)
        st.download_button("生成演示报告摘要", report_md, file_name="O2Cache_演示报告摘要.md", mime="text/markdown")
    else:
        st.info("运行一次 ICL 缓存决策后，可以生成包含本次指标的演示报告摘要。")

    if result is not None:
        summary = dict(result.get("summary", {}))
        if summary:
            st.download_button(
                "导出指标摘要",
                json.dumps({ui_col(k): v for k, v in summary.items()}, ensure_ascii=False, indent=2),
                file_name="O2Cache_指标摘要.json",
                mime="application/json",
            )

    if df_req is not None:
        try:
            load_df = task_load_export_frame(df_req)
            st.subheader("任务负载曲线")
            st.download_button("导出任务负载数据", load_df.to_csv(index=False, encoding="utf-8-sig"), file_name="O2Cache_任务负载数据.csv", mime="text/csv")
            if px:
                fig = px.line(load_df, x="时间步", y="请求到达量", color="任务序列", title="任务负载曲线")
                style_figure(fig, height=360)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("当前任务负载数据暂不能导出，请检查数据格式。")

    case_df = case_export_frame(lib)
    if not case_df.empty:
        st.subheader("历史案例库辅助材料")
        st.download_button("导出历史案例表", case_df.to_csv(index=False, encoding="utf-8-sig"), file_name="O2Cache_历史案例表.csv", mime="text/csv")

    st.subheader("演示材料辅助文件")
    st.write("可将上述导出结果用于整理评审演示中的任务负载曲线、策略对比结果、单步决策解释表和演示报告摘要。")


def render_docs():
    page_title("平台说明", "说明平台演示流程与数据含义。")
    st.write(
        "平台展示灾害任务请求接入、任务负载状态建模、历史案例库构建、Top-K 相似案例检索、"
        "ICL 少样本缓存决策、资源缓存量与资源预热建议、反馈评价与案例库更新条件分析等流程。"
    )


pages = {
    "系统总览": render_overview,
    "多模态任务接入": render_multimodal_agent_intake,
    "灾害任务接入": render_data_import,
    "历史案例库": render_case_library,
    "相似案例检索": render_retrieval,
    "ICL缓存决策": render_prediction,
    "调度效果评估": render_reward_workbench,
    "单步可解释分析": render_step_explanation,
    "结果导出": render_result_replay,
}

with st.sidebar:
    st.header("O₂Cache")
    st.subheader("应急智枢")
    st.caption("ICL资源缓存预测演示平台")
    page = st.radio("导航", list(pages.keys()))
    st.divider()
    st.caption(f"当前数据来源：{current_source_label()}")
    if st.session_state.request_df is not None:
        st.success("任务负载已接入")
    if st.session_state.case_library is not None:
        st.success(f"历史案例库：{len(st.session_state.case_library.cases)} 个案例")
    if st.session_state.last_result is not None:
        st.success("已有资源缓存决策结果")

pages[page]()
