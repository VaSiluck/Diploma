from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.view_model import STATE_LABELS


RISK_NORMAL_MAX = 0.30
RISK_WARNING_MAX = 0.70

STATE_COLORS = {
    "normal": "#16a34a",
    "watch": "#f59e0b",
    "alert": "#dc2626",
    "incident": "#111827",
    "warning": "#f59e0b",
    "risk": "#dc2626",
}

ANCHOR_LABELS = {
    "explicit_raw_log_keyword_or_severity": ("явный лог-якорь", "#2563eb"),
    "explicit_log_signal": ("явный лог-якорь", "#2563eb"),
    "run_end_proxy": ("proxy по концу run", "#7c3aed"),
    "run_end_proxy_no_log_events": ("proxy по концу run", "#7c3aed"),
    "normal_no_failure": ("нормальный запуск", "#15803d"),
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def format_score(score: float, digits: int = 2) -> str:
    score = float(np.clip(safe_float(score, 0.0), 0.0, 1.0))
    if score >= 1.0 - 0.5 * 10 ** (-digits):
        return f">{1.0 - 10 ** (-digits):.{digits}f}"
    if score <= 0.5 * 10 ** (-digits):
        return f"<{10 ** (-digits):.{digits}f}"
    return f"{score:.{digits}f}"


def format_metric(value: Any, digits: int = 3, empty: str = "n/a") -> str:
    value = safe_float(value, np.nan)
    if not np.isfinite(value):
        return empty
    return f"{value:.{digits}f}"


def format_delta(value: Any, digits: int = 3) -> str:
    value = safe_float(value, np.nan)
    if not np.isfinite(value):
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}"


def anchor_badge_parts(anchor_policy: Any) -> tuple[str, str]:
    return ANCHOR_LABELS.get(str(anchor_policy or ""), ("неизвестный якорь", "#64748b"))


def event_anchor_text(row: pd.Series) -> str:
    policy = str(row.get("anchor_policy", ""))
    scenario_type = str(row.get("scenario_type", ""))
    if policy == "normal_no_failure" or scenario_type == "normal":
        return "Нет сбоя"
    return "Якорь события"


def state_from_score(score: float) -> str:
    score = float(np.clip(score, 0.0, 1.0))
    if score >= RISK_WARNING_MAX:
        return "risk"
    if score >= RISK_NORMAL_MAX:
        return "warning"
    return "normal"


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = str(color).strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(109,93,252,{alpha})"
    try:
        red = int(color[0:2], 16)
        green = int(color[2:4], 16)
        blue = int(color[4:6], 16)
    except ValueError:
        return f"rgba(109,93,252,{alpha})"
    return f"rgba({red},{green},{blue},{alpha})"


def inject_css() -> str:
    return """
    <style>
    :root {
        color-scheme: light;
        --blue:#2563eb;
        --blue-dark:#1d4ed8;
        --ink:#102033;
        --muted:#64748b;
        --line:#e6edf5;
        --panel:#ffffff;
        --page:#f6f8fb;
        --soft:#f8fafc;
        --shadow:0 10px 26px rgba(16,32,51,0.06);
    }
    .stApp {
        background: var(--page);
        color: var(--ink);
    }
    .stApp * {
        color-scheme: light;
    }
    header[data-testid="stHeader"], div[data-testid="stToolbar"], #MainMenu, footer {
        display:none;
        visibility:hidden;
    }
    .block-container {
        width: min(1840px, 98vw);
        max-width: min(1840px, 98vw);
        padding-top: 0.75rem;
        padding-bottom: 1.1rem;
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        box-shadow: 0 1px 2px rgba(16,32,51,0.035) !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span {
        color: var(--ink);
    }
    h3 {
        font-size: 1.06rem;
        margin-top: 0.05rem;
        margin-bottom: 0.42rem;
    }
    .section-title {
        font-size:16px;
        font-weight:800;
        line-height:1.2;
        color:var(--ink);
        margin:0 0 10px 0;
    }
    .header-grid {
        display:grid;
        grid-template-columns:minmax(280px, 1.35fr) minmax(720px, 3.6fr);
        gap:12px;
        align-items:stretch;
        margin-bottom:10px;
    }
    .app-head {
        display:flex;
        align-items:center;
        gap:12px;
        min-height:76px;
        border:1px solid var(--line);
        border-radius:8px;
        background:#fff;
        padding:12px 14px;
        box-shadow:var(--shadow);
    }
    .logo-shield {
        width:44px;
        height:44px;
        border-radius:8px;
        background:linear-gradient(180deg,#2f80ed,#1457b8);
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-size:24px;
        font-weight:800;
        box-shadow:0 7px 18px rgba(29,110,232,0.22);
    }
    .app-title {
        font-size:25px;
        font-weight:750;
        line-height:1.0;
        letter-spacing:0;
    }
    .app-subtitle {
        color:var(--muted);
        font-size:12.5px;
        margin-top:5px;
        max-width:520px;
        line-height:1.35;
    }
    .status-grid {
        display:grid;
        grid-template-columns: repeat(5, minmax(130px, 1fr));
        gap:10px;
    }
    .top-card {
        border:1px solid var(--line);
        border-radius:8px;
        background:#fff;
        padding:10px 12px;
        min-height:76px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        min-width:0;
        box-shadow:var(--shadow);
    }
    .top-label {
        color:#536176;
        font-size:12px;
        line-height:1.05;
    }
    .top-value {
        font-size:18px;
        line-height:1.2;
        font-weight:750;
        margin-top:4px;
        color:var(--ink);
        overflow-wrap:anywhere;
    }
    .top-sub {
        color:#64748b;
        font-size:11px;
        margin-top:2px;
        line-height:1.2;
    }
    .top-risk {
        color:#dc2626;
        font-size:26px;
        font-weight:800;
        line-height:1;
    }
    .control-title {
        font-size:15px;
        font-weight:780;
        color:var(--ink);
        margin-bottom:7px;
    }
    .replay-meta {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:12px;
        color:#536176;
        font-size:12px;
        margin-top:5px;
        line-height:1.25;
    }
    .replay-step-value {
        color:#0f73e8;
        font-size:13px;
        font-weight:800;
        white-space:nowrap;
    }
    .case-note {
        border:1px solid #d8e8ff;
        background:#f4f9ff;
        color:#255a99;
        border-radius:8px;
        padding:7px 10px;
        font-size:12px;
        margin-top:8px;
        line-height:1.35;
    }
    .mode-note {
        border:1px solid #d8e8ff;
        background:#f4f9ff;
        color:#255a99;
        border-radius:999px;
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding:5px 10px;
        font-size:12px;
        line-height:1.2;
        margin:2px 0 9px 0;
    }
    .autoplay-tick {
        display:none;
    }
    div[data-testid="stSegmentedControl"] div[role="radiogroup"] {
        display:flex;
        flex-wrap:wrap;
        gap:8px;
    }
    div[data-testid="stSegmentedControl"] button,
    button[data-testid^="stBaseButton-segmented_control"] {
        flex:0 1 auto !important;
        width:auto !important;
        min-height:36px;
        min-width:96px;
        border:1px solid #d7e2ef !important;
        border-radius:7px !important;
        background:#ffffff !important;
        color:#26384d !important;
        white-space:normal !important;
        line-height:1.12 !important;
        padding:7px 10px !important;
        box-shadow:none !important;
    }
    button[data-testid="stBaseButton-segmented_controlActive"] {
        background:#eef6ff !important;
        border-color:#2563eb !important;
        color:#0f4fb3 !important;
        box-shadow:inset 0 0 0 1px rgba(37,99,235,0.18) !important;
    }
    .stButton > button {
        border-radius:6px;
        border:1px solid #d7e2ef;
        background:#fff;
        color:#26384d;
        min-height:36px;
        font-size:12px;
        font-weight:650;
        white-space:normal;
        overflow-wrap:anywhere;
        line-height:1.15;
        box-shadow:none;
    }
    .stButton > button[data-testid="stBaseButton-primary"],
    .stButton > button[kind="primary"] {
        background:#2563eb !important;
        border-color:#2563eb !important;
        color:#fff !important;
    }
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background:#ffffff !important;
        border-color:#cbd7e6 !important;
        color:#26384d !important;
    }
    .stButton > button:disabled,
    .stButton > button[disabled] {
        background:#f3f6fa !important;
        border-color:#d8e1ed !important;
        color:#8fa0b5 !important;
        opacity:1 !important;
    }
    .stButton > button:hover {
        border-color:#93c5fd;
        color:#0f73e8;
    }
    div[data-testid="stSlider"] {
        padding-top:2px;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        margin-top:0;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div {
        background:#d7e0ed !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {
        background:#2563eb !important;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background:#2563eb !important;
        border-color:#2563eb !important;
        box-shadow:0 0 0 4px rgba(37,99,235,0.13);
    }
    .chip-row {
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin:7px 0 6px 0;
    }
    .chip {
        border:1px solid #dbe5f1;
        background:#fff;
        border-radius:999px;
        padding:4px 9px;
        font-size:12px;
        color:#46566a;
    }
    .chip strong {
        color:#102033;
    }
    .evidence-table {
        border:1px solid var(--line);
        border-radius:8px;
        overflow:hidden;
        background:#fff;
    }
    .evidence-head, .evidence-row {
        display:grid;
        grid-template-columns:28px minmax(0, 1fr) 46px 62px 58px;
        gap:6px;
        align-items:center;
        padding:8px 8px;
        border-bottom:1px solid #eef3f8;
        font-size:12px;
        min-width:0;
    }
    .evidence-head > div,
    .evidence-row > div {
        min-width:0;
        overflow-wrap:anywhere;
    }
    .evidence-head {
        background:#f8fafc;
        color:#536176;
        font-weight:750;
    }
    .evidence-row:last-child {
        border-bottom:none;
    }
    .rank {
        width:22px;
        height:22px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        background:#0f73e8;
        color:white;
        font-weight:750;
        font-size:12px;
    }
    .bar-track {
        height:8px;
        border-radius:999px;
        background:#e2e8f0;
        overflow:hidden;
    }
    .bar-fill {
        height:100%;
        background:#0f73e8;
        border-radius:999px;
    }
    .delta-up { color:#dc2626; font-weight:750; }
    .delta-down { color:#0f766e; font-weight:750; }
    .delta-flat { color:#64748b; font-weight:650; }
    .side-row {
        display:grid;
        grid-template-columns: 1fr auto;
        gap:8px;
        border-bottom:1px solid #edf2f7;
        padding:8px 0;
        font-size:13px;
    }
    .side-row:last-child {
        border-bottom:none;
    }
    .side-k {
        color:#536176;
    }
    .side-v {
        font-weight:750;
        color:#102033;
        text-align:right;
    }
    .explain-box {
        border:1px solid #e6edf5;
        border-radius:8px;
        padding:10px 11px;
        color:#334155;
        background:#fbfdff;
        line-height:1.45;
        font-size:13px;
    }
    .panel-caption {
        color:#64748b;
        font-size:12px;
        line-height:1.4;
        margin-top:6px;
    }
    .small-muted {
        color:#64748b;
        font-size:12px;
        line-height:1.25;
    }
    .kpi-card {
        border:1px solid #e6edf5;
        border-radius:8px;
        background:#fff;
        padding:10px 11px 7px 11px;
        min-height:198px;
    }
    .kpi-tm-head {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:10px;
        min-height:50px;
    }
    .kpi-tm-title {
        display:flex;
        align-items:center;
        gap:8px;
        min-width:0;
    }
    .kpi-tm-dot {
        width:9px;
        height:9px;
        border-radius:50%;
        flex:0 0 auto;
        box-shadow:0 0 0 4px rgba(37,99,235,0.08);
    }
    .kpi-tm-name {
        font-size:13px;
        font-weight:800;
        color:#102033;
        line-height:1.1;
    }
    .kpi-tm-subtitle {
        color:#64748b;
        font-size:11.3px;
        line-height:1.22;
        margin-top:3px;
        overflow-wrap:anywhere;
    }
    .kpi-tm-value {
        font-size:22px;
        line-height:1.0;
        font-weight:850;
        color:#102033;
        text-align:right;
        white-space:nowrap;
        font-variant-numeric:tabular-nums;
    }
    .kpi-tm-delta {
        color:#64748b;
        font-size:11.5px;
        line-height:1.25;
        margin-top:5px;
        text-align:right;
        white-space:nowrap;
    }
    .kpi-tm-meter {
        height:5px;
        border-radius:999px;
        background:#e8eef6;
        overflow:hidden;
        margin:7px 0 2px 0;
    }
    .kpi-tm-fill {
        height:100%;
        border-radius:999px;
    }
    .kpi-title {
        display:flex;
        justify-content:space-between;
        align-items:baseline;
        gap:8px;
        font-size:12px;
        font-weight:750;
        min-height:31px;
    }
    .kpi-title span:first-child {
        overflow-wrap:anywhere;
    }
    .kpi-value {
        color:#4f46e5;
        font-size:12px;
        font-weight:750;
        white-space:nowrap;
    }
    .baseline-table {
        border:1px solid #e6edf5;
        border-radius:8px;
        overflow:hidden;
        background:#fff;
        font-size:11.5px;
    }
    .baseline-head,
    .baseline-row {
        display:grid;
        grid-template-columns:minmax(0, 1fr) 36px 42px 52px;
        gap:5px;
        align-items:center;
        padding:7px 6px;
        border-bottom:1px solid #eef3f8;
    }
    .baseline-head > div,
    .baseline-row > div {
        min-width:0;
        overflow-wrap:anywhere;
    }
    .baseline-head {
        background:#111827;
        color:#fff;
        font-weight:750;
    }
    .baseline-row:last-child {
        border-bottom:none;
    }
    .baseline-row strong {
        color:#111827;
    }
    .baseline-num {
        font-variant-numeric:tabular-nums;
        text-align:right;
        font-weight:700;
    }
    .honest-card {
        border:1px solid #e6edf5;
        border-radius:8px;
        background:#fff;
        overflow:hidden;
    }
    .honest-row,
    .honest-head {
        display:grid;
        grid-template-columns:minmax(0, 1.2fr) 46px 54px 44px 50px 58px;
        gap:6px;
        align-items:center;
        padding:8px 8px;
        border-bottom:1px solid #eef3f8;
        font-size:11.5px;
    }
    .honest-head {
        background:#f8fafc;
        color:#536176;
        font-weight:800;
    }
    .honest-row:last-child {
        border-bottom:none;
    }
    .honest-model {
        font-weight:760;
        color:#102033;
        min-width:0;
        overflow-wrap:anywhere;
    }
    .honest-num {
        text-align:right;
        font-variant-numeric:tabular-nums;
        color:#102033;
        font-weight:700;
    }
    .honest-muted {
        color:#64748b;
        font-size:11.5px;
        line-height:1.32;
        padding:8px 9px;
        background:#fbfdff;
        border-top:1px solid #eef3f8;
    }
    .delta-good { color:#15803d; }
    .delta-bad { color:#dc2626; }
    .delta-even { color:#64748b; }
    .verdict-badge {
        display:inline-flex;
        align-items:center;
        border-radius:999px;
        padding:3px 8px;
        font-size:11.5px;
        font-weight:800;
        border:1px solid #e2e8f0;
        background:#f8fafc;
        color:#334155;
        margin-bottom:7px;
    }
    .verdict-bad {
        border-color:#fecaca;
        background:#fff1f2;
        color:#dc2626;
    }
    .verdict-good {
        border-color:#bbf7d0;
        background:#f0fdf4;
        color:#15803d;
    }
    .verdict-even {
        border-color:#dbe5f1;
        background:#f8fafc;
        color:#475569;
    }
    .window-score-box {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:8px;
        margin-top:9px;
    }
    .window-score {
        border:1px solid #e6edf5;
        border-radius:8px;
        padding:8px;
        background:#fbfdff;
    }
    .window-score-label {
        color:#64748b;
        font-size:11.5px;
    }
    .window-score-value {
        color:#102033;
        font-size:18px;
        line-height:1.1;
        font-weight:850;
        margin-top:3px;
        font-variant-numeric:tabular-nums;
    }
    .reliability-grid {
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:8px;
    }
    .reliability-tile {
        border:1px solid #e6edf5;
        border-radius:8px;
        background:#fff;
        padding:9px;
    }
    .reliability-label {
        font-size:12px;
        color:#64748b;
    }
    .reliability-value {
        font-size:17px;
        font-weight:850;
        color:#102033;
        margin-top:2px;
    }
    .anchor-pill {
        display:inline-flex;
        align-items:center;
        gap:6px;
        border-radius:999px;
        padding:4px 9px;
        font-size:12px;
        font-weight:750;
        background:#f8fafc;
        border:1px solid #dbe5f1;
        color:#334155;
    }
    .anchor-dot {
        width:7px;
        height:7px;
        border-radius:50%;
        display:inline-block;
    }
    @media (max-width: 1100px) {
        .header-grid {
            grid-template-columns:1fr;
        }
        .status-grid {
            grid-template-columns:repeat(3, minmax(150px, 1fr));
        }
    }
    @media (max-width: 760px) {
        .block-container {
            width: min(100vw, 100%);
            max-width: min(100vw, 100%);
            padding-left:0.75rem;
            padding-right:0.75rem;
        }
        .status-grid {
            grid-template-columns:1fr;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-wrap:wrap !important;
        }
        div[data-testid="column"] {
            min-width:min(100%, 320px) !important;
            width:100% !important;
        }
        .evidence-head, .evidence-row {
            grid-template-columns:28px 1fr;
        }
        .evidence-head div:nth-child(n+3),
        .evidence-row div:nth-child(n+3) {
            display:none;
        }
    }
    </style>
    """


def top_card(label: str, value: str, sub: str = "", tone: str = "default") -> str:
    color = {
        "default": "#102033",
        "blue": "#0f73e8",
        "warning": "#d97706",
        "danger": "#dc2626",
        "success": "#15803d",
    }.get(tone, "#102033")
    return (
        "<div class='top-card'>"
        f"<div class='top-label'>{html.escape(label)}</div>"
        f"<div class='top-value' style='color:{color}'>{html.escape(value)}</div>"
        f"<div class='top-sub'>{html.escape(sub)}</div>"
        "</div>"
    )


def chip_row(items: list[tuple[str, str]]) -> str:
    chips = [
        f"<span class='chip'>{html.escape(label)}: <strong>{html.escape(value)}</strong></span>"
        for label, value in items
    ]
    return f"<div class='chip-row'>{''.join(chips)}</div>"


def replay_dots(total: int, current: int) -> str:
    dots = []
    for idx in range(max(total, 1)):
        cls = "current" if idx == current else ("done" if idx < current else "")
        dots.append(f"<span class='replay-dot {cls}' title='Шаг {idx + 1}'></span>")
    return f"<div class='replay-dots'>{''.join(dots)}</div>"


def build_risk_figure(run_df: pd.DataFrame, current_idx: int, preview: bool) -> go.Figure:
    fig = go.Figure()
    run_df = run_df.sort_values("window_ts").reset_index(drop=True)
    current_idx = int(np.clip(current_idx, 0, max(len(run_df) - 1, 0)))
    cur = run_df.iloc[current_idx]

    x = pd.to_datetime(run_df["window_ts"], utc=True, errors="coerce")
    score_col = "alert_score" if "alert_score" in run_df.columns else ("model_score_raw" if "model_score_raw" in run_df.columns else ("model_score" if "model_score" in run_df.columns else "risk_score"))
    risk = pd.to_numeric(run_df[score_col], errors="coerce").fillna(0.0).clip(0, 1)
    rank = pd.to_numeric(run_df.get("model_score_rank", pd.Series(np.nan, index=run_df.index)), errors="coerce")
    current_ts = pd.Timestamp(cur["window_ts"])
    current_score = safe_float(cur.get(score_col))
    watch_threshold = safe_float(cur.get("watch_threshold", RISK_NORMAL_MAX), RISK_NORMAL_MAX)
    alert_threshold = safe_float(cur.get("alert_threshold", cur.get("threshold_risk", RISK_WARNING_MAX)), RISK_WARNING_MAX)
    watch_threshold = float(np.clip(watch_threshold, 0.0, 1.0))
    alert_threshold = float(np.clip(alert_threshold, watch_threshold, 1.0))

    window_end = pd.to_datetime(cur.get("window_end"), utc=True, errors="coerce")
    macro_sec = int(max(safe_float(cur.get("macro_window_sec", 60), 60), safe_float(cur.get("micro_window_sec", 10), 10)))
    if pd.isna(window_end) or pd.Timestamp(window_end) <= current_ts:
        window_end = current_ts + pd.Timedelta(seconds=macro_sec)
    window_end = pd.Timestamp(window_end)
    if pd.notna(x.max()):
        window_end = min(window_end, pd.Timestamp(x.max()))
    if window_end <= current_ts:
        window_end = current_ts + pd.Timedelta(seconds=int(safe_float(cur.get("micro_window_sec", 10), 10)))

    fig.add_hrect(y0=0.0, y1=watch_threshold, fillcolor="#ecfdf5", opacity=0.42, line_width=0, layer="below")
    fig.add_hrect(y0=watch_threshold, y1=alert_threshold, fillcolor="#fffbeb", opacity=0.38, line_width=0, layer="below")
    fig.add_hrect(y0=alert_threshold, y1=1.05, fillcolor="#fff1f2", opacity=0.34, line_width=0, layer="below")

    kpi_missing = pd.to_numeric(run_df.get("kpi_available", 1), errors="coerce").fillna(1).eq(0).to_numpy()
    start = None
    has_kpi_missing = bool(kpi_missing.any())
    for idx, flag in enumerate(kpi_missing):
        if flag and start is None:
            start = idx
        if start is not None and ((not flag) or idx == len(kpi_missing) - 1):
            end = idx if flag else idx - 1
            fig.add_vrect(
                x0=x.iloc[start],
                x1=x.iloc[end],
                fillcolor="#d1d5db",
                opacity=0.22,
                line_width=0,
                layer="below",
            )
            start = None

    fig.add_vrect(
        x0=current_ts,
        x1=window_end,
        fillcolor="#dbeafe",
        opacity=0.70,
        line_width=0,
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=risk,
            mode="lines",
            name="Оценка модели" if not preview else "Proxy-оценка",
            line={"color": "#2f80ed", "width": 3.0, "shape": "spline", "smoothing": 0.55},
            fill="tozeroy",
            fillcolor="rgba(47,128,237,0.13)",
            customdata=np.stack([rank.fillna(np.nan).to_numpy()], axis=-1),
            hovertemplate="%{x|%H:%M:%S}<br>raw score=%{y:.3f}<br>ранг validation=%{customdata[0]:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=risk,
            mode="markers",
            name="Окна replay",
            marker={"size": 4.0, "color": "#2f80ed", "opacity": 0.42},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    for value, label, color, dash in [
        (watch_threshold, "Порог наблюдения", "#f59e0b", "dot"),
        (alert_threshold, "Порог тревоги", "#dc2626", "dash"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[value] * len(x),
                mode="lines",
                name=label,
                line={"color": color, "width": 1.1, "dash": dash},
                hoverinfo="skip",
            )
        )
    if has_kpi_missing:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="KPI отсутствует",
                marker={"size": 9, "symbol": "square", "color": "rgba(148,163,184,0.55)"},
                hoverinfo="skip",
            )
        )

    fig.add_vline(x=current_ts, line_color="#0f73e8", line_width=1.5)
    fig.add_trace(
        go.Scatter(
            x=[current_ts],
            y=[current_score],
            mode="markers+text",
            text=[format_score(current_score)],
            textposition="middle right",
            name="Выбранное окно",
            marker={"size": 12, "color": "#ffffff", "line": {"color": "#0f73e8", "width": 2.4}},
            hoverinfo="skip",
        )
    )
    fig.add_annotation(
        x=current_ts,
        y=1.0,
        text=f"Выбранное окно<br>{current_ts.strftime('%H:%M')}-{window_end.strftime('%H:%M')}",
        showarrow=False,
        yanchor="bottom",
        bgcolor="#0f73e8",
        bordercolor="#0f73e8",
        borderpad=5,
        font={"size": 10, "color": "#ffffff"},
    )

    alert_state = run_df.get("alert_state", pd.Series("", index=run_df.index)).fillna("").astype(str).str.lower()
    alert_mask = alert_state.eq("alert")
    if not alert_mask.any():
        alert_mask = pd.to_numeric(run_df.get("alert_fired", 0), errors="coerce").fillna(0).astype(int).eq(1)
    if alert_mask.any():
        alert_ts = pd.Timestamp(pd.to_datetime(run_df.loc[alert_mask, "window_ts"], utc=True, errors="coerce").iloc[0])
        fig.add_vline(x=alert_ts, line_color="#f59e0b", line_width=1.8, line_dash="dash")
        fig.add_annotation(
            x=alert_ts,
            y=0.88,
            text=f"ТРЕВОГА<br>{alert_ts.strftime('%H:%M')}",
            showarrow=False,
            font={"size": 10, "color": "#b45309"},
            yanchor="bottom",
            bgcolor="#fffbeb",
            bordercolor="#f59e0b",
            borderpad=4,
        )

    anchor_policy = str(cur.get("anchor_policy", ""))
    anchor_ts = pd.to_datetime(cur.get("event_anchor", cur.get("fail_start")), utc=True, errors="coerce")
    if anchor_policy != "normal_no_failure" and pd.notna(anchor_ts):
        anchor_label, anchor_color = anchor_badge_parts(anchor_policy)
        line_dash = "solid" if anchor_policy in {"explicit_raw_log_keyword_or_severity", "explicit_log_signal"} else "dash"
        fig.add_vline(x=pd.Timestamp(anchor_ts), line_color=anchor_color, line_width=1.2, line_dash=line_dash)
        fig.add_annotation(
            x=pd.Timestamp(anchor_ts),
            y=0.98,
            text=f"ИНЦИДЕНТ<br>{anchor_label}<br>{pd.Timestamp(anchor_ts).strftime('%H:%M')}",
            showarrow=False,
            font={"size": 10, "color": anchor_color},
            yanchor="bottom",
        )

    x_left = x.min()
    if pd.notna(x_left):
        normal_y = max(watch_threshold * 0.5, 0.04)
        watch_y = watch_threshold + max((alert_threshold - watch_threshold) * 0.5, 0.03)
        alert_y = alert_threshold + max((1.0 - alert_threshold) * 0.5, 0.03)
        fig.add_annotation(xref="paper", x=0.017, y=min(alert_y, 1.02), text="Тревога", showarrow=False, bgcolor="#dc2626", font={"color": "white", "size": 11}, borderpad=4)
        fig.add_annotation(xref="paper", x=0.017, y=min(watch_y, 1.0), text="Наблюдение", showarrow=False, bgcolor="#fff7ed", font={"color": "#d97706", "size": 11}, bordercolor="#facc15", borderpad=4)
        fig.add_annotation(xref="paper", x=0.017, y=normal_y, text="Норма", showarrow=False, bgcolor="#ecfdf5", font={"color": "#15803d", "size": 11}, bordercolor="#86efac", borderpad=4)

    fig.update_layout(
        height=470,
        margin={"l": 48, "r": 18, "t": 44, "b": 34},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=True,
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "center",
            "x": 0.52,
            "font": {"size": 10},
            "itemsizing": "constant",
        },
        font={"color": "#334155"},
    )
    fig.update_xaxes(
        tickformat="%H:%M",
        gridcolor="#edf2f7",
        zeroline=False,
        nticks=7,
        tickfont={"size": 11},
        showline=True,
        linecolor="#dbe5f1",
        fixedrange=True,
    )
    fig.update_yaxes(
        title_text="Оценка модели, 0-1 (не вероятность сбоя)",
        range=[0, 1.06],
        gridcolor="#edf2f7",
        zeroline=False,
        tickfont={"size": 11},
        showline=True,
        linecolor="#dbe5f1",
        fixedrange=True,
    )
    return fig


def build_fusion_figure(log_share: float, kpi_share: float | None = None, share_available: bool = True) -> go.Figure:
    if not share_available:
        log_share = 0.5
        kpi_share = 0.5
    else:
        log_share = float(np.clip(log_share, 0.0, 1.0))
        kpi_share = max(0.0, 1.0 - log_share) if kpi_share is None else float(np.clip(kpi_share, 0.0, 1.0))
        total = max(log_share + kpi_share, 1e-9)
        log_share = log_share / total
        kpi_share = kpi_share / total
    colors = ["#1d6ee8", "#6d28d9"] if share_available else ["#94a3b8", "#cbd5e1"]
    annotation = "Баланс<br>сигналов" if share_available else "Баланс<br>н/д"
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Логи", "KPI"],
                values=[log_share, max(kpi_share, 1e-6)],
                hole=0.68,
                sort=False,
                textinfo="none",
                marker={"colors": colors},
            )
        ]
    )
    fig.update_layout(
        height=166,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        paper_bgcolor="#ffffff",
        annotations=[{"text": annotation, "showarrow": False, "font": {"size": 12, "color": "#334155"}}],
    )
    return fig


def build_kpi_sparkline(item: dict[str, Any], current_ts: pd.Timestamp) -> go.Figure:
    x = pd.to_datetime(pd.Series(item["x"]), utc=True, errors="coerce")
    y = pd.to_numeric(pd.Series(item["y"]), errors="coerce")
    valid = x.notna() & y.notna()
    fig = go.Figure()
    accent = str(item.get("accent", "#6d5dfc"))
    if int(valid.sum()) >= 2:
        fig.add_trace(
            go.Scatter(
                x=x[valid],
                y=y[valid],
                mode="lines",
                line={"color": accent, "width": 1.8, "shape": "spline", "smoothing": 0.45},
                fill="tozeroy",
                fillcolor=_hex_to_rgba(accent, 0.13),
                showlegend=False,
                hovertemplate="%{x|%H:%M}<br>%{y:.3g}<extra></extra>",
            )
        )
        current_value = pd.to_numeric(pd.Series([item.get("value")]), errors="coerce").iloc[0]
        if pd.notna(current_value) and pd.notna(current_ts):
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(current_ts)],
                    y=[float(current_value)],
                    mode="markers",
                    marker={"size": 7, "color": "#ffffff", "line": {"color": accent, "width": 2}},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    else:
        fig.add_annotation(
            text="Нет данных",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 11, "color": "#94a3b8"},
        )
    if pd.notna(current_ts):
        fig.add_vline(x=pd.Timestamp(current_ts), line_color="#334155", line_dash="dot", line_width=1.0)
    yaxis_options: dict[str, Any] = {
        "showticklabels": False,
        "showgrid": True,
        "gridcolor": "#edf2f7",
        "zeroline": False,
        "fixedrange": True,
    }
    if item.get("unit") == "%":
        yaxis_options["range"] = [0, 100]
    fig.update_layout(
        height=118,
        margin={"l": 5, "r": 5, "t": 6, "b": 2},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor="#edf2f7", zeroline=False, fixedrange=True)
    fig.update_yaxes(**yaxis_options)
    return fig


def render_log_table_html(log_df: pd.DataFrame) -> str:
    rows = []
    for _, row in log_df.iterrows():
        rank = int(safe_float(row.get("Ранг", 0), 0))
        label = str(row.get("Шаблон", ""))
        freq = int(safe_float(row.get("Частота", 0), 0))
        delta = safe_float(row.get("Δ к baseline", 0.0), 0.0)
        contrib = float(np.clip(safe_float(row.get("Вклад", 0.0), 0.0), 0, 1))
        delta_cls = "delta-up" if delta > 5 else ("delta-down" if delta < -5 else "delta-flat")
        if abs(delta) >= 1000:
            delta_text = f"{delta / 1000:+.1f}k%"
        else:
            delta_text = f"{delta:+.0f}%"
        freq_txt = f"{freq:,}".replace(",", " ")
        rows.append(
            "<div class='evidence-row'>"
            f"<div class='rank'>{rank}</div>"
            f"<div>{html.escape(label)}</div>"
            f"<div>{freq_txt}</div>"
            f"<div class='{delta_cls}'>{delta_text}</div>"
            "<div>"
            "<div class='bar-track'>"
            f"<div class='bar-fill' style='width:{contrib * 100:.1f}%'></div>"
            "</div>"
            f"<div class='small-muted'>{contrib * 100:.1f}%</div>"
            "</div>"
            "</div>"
        )
    return (
        "<div class='evidence-table'>"
        "<div class='evidence-head'><div></div><div>Шаблон</div><div>Частота</div><div>Δ baseline</div><div>Вклад</div></div>"
        f"{''.join(rows)}"
        "</div>"
    )


def render_baseline_table_html(table: pd.DataFrame, max_rows: int = 6) -> str:
    if table.empty:
        return "<div class='explain-box'>Baseline-сводка недоступна.</div>"

    show = table.head(max_rows).copy()
    rows = []
    for _, row in show.iterrows():
        model = str(row.get("Модель", row.get("baseline", "n/a")))
        target = str(row.get("Цель", row.get("target", "n/a")))
        ap = safe_float(row.get("AP", row.get("average_precision", np.nan)), np.nan)
        roc = safe_float(row.get("ROC-AUC", row.get("roc_auc", np.nan)), np.nan)
        ap_text = "n/a" if not np.isfinite(ap) else f"{ap:.3f}"
        roc_text = "n/a" if not np.isfinite(roc) else f"{roc:.3f}"
        rows.append(
            "<div class='baseline-row'>"
            f"<div><strong>{html.escape(model)}</strong></div>"
            f"<div>{html.escape(target)}</div>"
            f"<div class='baseline-num'>{ap_text}</div>"
            f"<div class='baseline-num'>{roc_text}</div>"
            "</div>"
        )

    return (
        "<div class='baseline-table'>"
        "<div class='baseline-head'><div>Модель</div><div>Цель</div><div>AP</div><div>ROC-AUC</div></div>"
        f"{''.join(rows)}"
        "</div>"
    )


def _diagnostic_for_target(report: dict[str, Any], target: str) -> dict[str, Any]:
    for row in report.get("baseline_diagnostics", []) if isinstance(report, dict) else []:
        if str(row.get("target")) == target:
            return dict(row)
    return {}


def _derive_diagnostic_from_table(table: pd.DataFrame, target: str) -> dict[str, Any]:
    if table.empty:
        return {}
    rows = table.copy()
    if "status" in rows.columns:
        rows = rows.loc[rows["status"].fillna("ok").eq("ok")].copy()
    if "slice" in rows.columns:
        rows = rows.loc[rows["slice"].fillna("all").eq("all")].copy()
    model_col = "model" if "model" in rows.columns else ("baseline" if "baseline" in rows.columns else "")
    if "target" not in rows.columns and model_col:
        metric_col = "run_alert_f1" if "run_alert_f1" in rows.columns else ("f1" if "f1" in rows.columns else "")
        if not metric_col:
            return {}
        fused_rows = rows.loc[rows[model_col].astype(str).eq("fused")].copy()
        baseline_rows = rows.loc[~rows[model_col].astype(str).isin(["fused", "full_lr"])].copy()
        if fused_rows.empty:
            return {}
        fused = fused_rows.iloc[0]
        best = baseline_rows.sort_values(metric_col, ascending=False).iloc[0] if not baseline_rows.empty else None
        fused_metric = safe_float(fused.get(metric_col), np.nan)
        best_metric = safe_float(best.get(metric_col), np.nan) if best is not None else np.nan
        delta = fused_metric - best_metric if np.isfinite(fused_metric) and np.isfinite(best_metric) else np.nan
        verdict = "хуже baseline" if np.isfinite(delta) and delta < -0.02 else ("лучше baseline" if np.isfinite(delta) and delta > 0.02 else "паритет")
        return {
            "target": "run_level_alert",
            "central_average_precision": safe_float(fused.get("average_precision"), np.nan),
            "central_roc_auc": safe_float(fused.get("roc_auc"), np.nan),
            "central_f1": fused_metric,
            "central_recall": safe_float(fused.get("failure_run_recall", fused.get("recall")), np.nan),
            "central_precision": safe_float(fused.get("run_alert_precision", fused.get("precision")), np.nan),
            "central_false_alert_run_rate": safe_float(fused.get("false_alert_run_rate"), np.nan),
            "central_alert_density": safe_float(fused.get("alert_density"), np.nan),
            "random_reference_ap": safe_float(fused.get("prevalence"), np.nan),
            "best_baseline": str(best.get(model_col, "baseline")) if best is not None else "baseline",
            "best_baseline_average_precision": safe_float(best.get("average_precision"), np.nan) if best is not None else np.nan,
            "best_baseline_roc_auc": safe_float(best.get("roc_auc"), np.nan) if best is not None else np.nan,
            "best_baseline_f1": best_metric,
            "best_baseline_recall": safe_float(best.get("failure_run_recall", best.get("recall")), np.nan) if best is not None else np.nan,
            "best_baseline_precision": safe_float(best.get("run_alert_precision", best.get("precision")), np.nan) if best is not None else np.nan,
            "best_baseline_false_alert_run_rate": safe_float(best.get("false_alert_run_rate"), np.nan) if best is not None else np.nan,
            "best_baseline_alert_density": safe_float(best.get("alert_density"), np.nan) if best is not None else np.nan,
            "delta_vs_best_baseline_ap": delta,
            "verdict": verdict,
        }
    if "target" not in rows.columns:
        return {}
    rows = rows.loc[rows["target"].astype(str).eq(target)].copy()
    if rows.empty:
        return {}
    central_names = {
        "central-fusion-model",
        "hades-like-strict-intensity-fusion",
        "hades-like-state-score",
        "final-hades-like-model",
        "final-multimodal-strict-intensity-fusion",
        "final-multimodal-state-score",
    }
    central_mask = rows["baseline"].astype(str).isin(central_names)
    central_rows = rows.loc[central_mask].copy()
    non_central = rows.loc[~central_mask].copy()
    if central_rows.empty:
        return {}
    central = central_rows.iloc[0]
    best = non_central.sort_values("average_precision", ascending=False).iloc[0] if not non_central.empty else None
    central_ap = safe_float(central.get("average_precision"), np.nan)
    best_ap = safe_float(best.get("average_precision"), np.nan) if best is not None else np.nan
    delta = central_ap - best_ap if np.isfinite(central_ap) and np.isfinite(best_ap) else np.nan
    verdict = "хуже baseline" if np.isfinite(delta) and delta < -0.02 else ("лучше baseline" if np.isfinite(delta) and delta > 0.02 else "паритет")
    return {
        "target": target,
        "central_average_precision": central_ap,
        "central_roc_auc": safe_float(central.get("roc_auc"), np.nan),
        "central_f1": safe_float(central.get("f1"), np.nan),
        "central_recall": safe_float(central.get("recall"), np.nan),
        "random_reference_ap": safe_float(central.get("prevalence"), np.nan),
        "best_baseline": str(best.get("baseline", "baseline")) if best is not None else "baseline",
        "best_baseline_average_precision": best_ap,
        "best_baseline_roc_auc": safe_float(best.get("roc_auc"), np.nan) if best is not None else np.nan,
        "best_baseline_f1": safe_float(best.get("f1"), np.nan) if best is not None else np.nan,
        "best_baseline_recall": safe_float(best.get("recall"), np.nan) if best is not None else np.nan,
        "delta_vs_best_baseline_ap": delta,
        "verdict": verdict,
    }


def render_honest_baseline_card_html(
    table: pd.DataFrame,
    report: dict[str, Any],
    target: str,
    current: pd.Series | None = None,
) -> str:
    diag = _diagnostic_for_target(report, target) or _derive_diagnostic_from_table(table, target)
    if not diag:
        return "<div class='explain-box'>Baseline-сравнение недоступно.</div>"

    is_run_level = str(diag.get("target", "")) == "run_level_alert"
    target_short = "risk" if target == "y_risk" else "warning"
    target_label = "alert-layer" if is_run_level else ("risk" if target == "y_risk" else "warning")
    delta = safe_float(diag.get("delta_vs_best_baseline_ap"), np.nan)
    verdict = str(diag.get("verdict", "паритет"))
    verdict_cls = "verdict-bad" if "хуже" in verdict else ("verdict-good" if "лучше" in verdict else "verdict-even")
    delta_cls = "delta-bad" if np.isfinite(delta) and delta < -0.02 else ("delta-good" if np.isfinite(delta) and delta > 0.02 else "delta-even")

    rows = [
        (
            "random / prevalence",
            diag.get("random_reference_ap"),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            "delta-even",
        ),
        (
            str(diag.get("best_baseline", "best baseline")),
            diag.get("best_baseline_average_precision"),
            diag.get("best_baseline_roc_auc"),
            diag.get("best_baseline_f1"),
            diag.get("best_baseline_recall"),
            0.0,
            "delta-even",
        ),
        (
            "fused alert",
            diag.get("central_average_precision"),
            diag.get("central_roc_auc"),
            diag.get("central_f1"),
            diag.get("central_recall"),
            delta,
            delta_cls,
        ),
    ]

    html_rows = []
    for model, ap, roc, f1, recall, row_delta, cls in rows:
        html_rows.append(
            "<div class='honest-row'>"
            f"<div class='honest-model'>{html.escape(model)}</div>"
            f"<div class='honest-num'>{format_metric(ap)}</div>"
            f"<div class='honest-num'>{format_metric(roc)}</div>"
            f"<div class='honest-num'>{format_metric(f1)}</div>"
            f"<div class='honest-num'>{format_metric(recall)}</div>"
            f"<div class='honest-num {cls}'>{format_delta(row_delta) if np.isfinite(safe_float(row_delta, np.nan)) else '—'}</div>"
            "</div>"
        )

    current_html = ""
    if current is not None and not is_run_level:
        central_col = "risk_score" if target == "y_risk" else "warning_score"
        baseline_col = f"baseline_{target_short}_score"
        baseline_model_col = f"baseline_{target_short}_model"
        central_score = safe_float(current.get(central_col), np.nan)
        baseline_score = safe_float(current.get(baseline_col), np.nan)
        if np.isfinite(central_score) and np.isfinite(baseline_score):
            baseline_model = str(current.get(baseline_model_col, diag.get("best_baseline", "baseline")))
            current_html = (
                "<div class='window-score-box'>"
                "<div class='window-score'>"
                "<div class='window-score-label'>central score окна</div>"
                f"<div class='window-score-value'>{format_score(central_score)}</div>"
                "</div>"
                "<div class='window-score'>"
                f"<div class='window-score-label'>{html.escape(baseline_model)} score окна</div>"
                f"<div class='window-score-value'>{format_score(baseline_score)}</div>"
                "</div>"
                "</div>"
            )

    verdict_message = report.get("training_verdict", {}).get("message", "") if isinstance(report, dict) else ""
    if not verdict_message:
        verdict_message = "Сравнение строится по test split; F1@10% и Recall@10% считаются для верхних 10% окон по score, чтобы сравнивать ранжирование без притворной калибровки."
    delta_label = "Δ F1" if is_run_level else "Δ AP"

    return (
        f"<span class='verdict-badge {verdict_cls}'>{html.escape(verdict)}</span>"
        "<div class='honest-card'>"
        f"<div class='honest-head'><div>Модель</div><div>AP</div><div>ROC-AUC</div><div>F1@10%</div><div>Recall@10%</div><div>{html.escape(delta_label)}</div></div>"
        f"{''.join(html_rows)}"
        f"<div class='honest-muted'>Цель: <strong>{html.escape(target_label)}</strong>. {html.escape(verdict_message)}</div>"
        "</div>"
        f"{current_html}"
    )


def render_reliability_html(report: dict[str, Any], calibration_bins: pd.DataFrame) -> str:
    strict_rows = report.get("strict_research_summary", []) if isinstance(report, dict) else []
    if strict_rows:
        rows_html = []
        for row in strict_rows:
            target = "risk" if str(row.get("target")) == "y_risk" else "warning"
            delta = safe_float(row.get("delta_ap"), np.nan)
            cls = "delta-good" if np.isfinite(delta) and delta > 0 else ("delta-bad" if np.isfinite(delta) and delta < 0 else "delta-even")
            rows_html.append(
                "<div class='honest-row'>"
                f"<div class='honest-model'>{html.escape(target)}</div>"
                f"<div class='honest-num'>{format_metric(row.get('model_average_precision'))}</div>"
                f"<div class='honest-num'>{html.escape(str(row.get('best_baseline', 'baseline')))}</div>"
                f"<div class='honest-num'>{format_metric(row.get('best_baseline_average_precision'))}</div>"
                f"<div class='honest-num {cls}'>{format_delta(delta)}</div>"
                "</div>"
            )
        return (
            "<div class='honest-card'>"
            "<div class='honest-head'><div>Цель</div><div>AP модели</div><div>Лучший baseline</div><div>AP baseline</div><div>Δ AP</div></div>"
            f"{''.join(rows_html)}"
            "<div class='honest-muted'>Строгая исследовательская проверка: explicit anchors, gap10. Этот блок отделен от продуктового replay alert-layer.</div>"
            "</div>"
        )

    diagnostics = report.get("baseline_diagnostics", []) if isinstance(report, dict) else []
    if not diagnostics:
        return "<div class='explain-box'>Диагностика надежности недоступна.</div>"

    ece_by_target: dict[str, float] = {}
    if not calibration_bins.empty and {"target", "n", "abs_calibration_error"}.issubset(calibration_bins.columns):
        for target, group in calibration_bins.groupby("target"):
            n = pd.to_numeric(group["n"], errors="coerce").fillna(0)
            err = pd.to_numeric(group["abs_calibration_error"], errors="coerce").fillna(0)
            denom = float(n.sum())
            if denom > 0:
                ece_by_target[str(target)] = float((n * err).sum() / denom)

    tiles = []
    for row in diagnostics:
        target = str(row.get("target"))
        label = "Warning" if target == "y_warning" else "Risk"
        ap = safe_float(row.get("central_average_precision"), np.nan)
        random_ap = safe_float(row.get("random_reference_ap"), np.nan)
        roc = safe_float(row.get("central_roc_auc"), np.nan)
        ece = ece_by_target.get(target, np.nan)
        tiles.append(
            "<div class='reliability-tile'>"
            f"<div class='reliability-label'>{html.escape(label)} AP vs prevalence</div>"
            f"<div class='reliability-value'>{format_metric(ap)} / {format_metric(random_ap)}</div>"
            f"<div class='small-muted'>ROC-AUC {format_metric(roc)} · calib.err {format_metric(ece)}</div>"
            "</div>"
        )

    return (
        f"<div class='reliability-grid'>{''.join(tiles)}</div>"
        "<div class='honest-muted'>Score показывает ранжирование модели и не является откалиброванной вероятностью события.</div>"
    )


def state_badge(state: str) -> str:
    state = state if state in STATE_LABELS else "normal"
    color = STATE_COLORS[state]
    return (
        f"<span style='display:inline-flex;border:1px solid {color}33;background:{color}12;"
        f"color:{color};border-radius:999px;padding:3px 8px;font-weight:750;font-size:12px'>"
        f"{html.escape(STATE_LABELS[state])}</span>"
    )
