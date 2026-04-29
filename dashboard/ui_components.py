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
    "warning": "#f59e0b",
    "risk": "#dc2626",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


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
    risk = pd.to_numeric(run_df["risk_score"], errors="coerce").fillna(0.0).clip(0, 1)
    current_ts = pd.Timestamp(cur["window_ts"])
    current_score = safe_float(cur.get("risk_score"))

    window_end = pd.to_datetime(cur.get("window_end"), utc=True, errors="coerce")
    macro_sec = int(max(safe_float(cur.get("macro_window_sec", 60), 60), safe_float(cur.get("micro_window_sec", 10), 10)))
    if pd.isna(window_end) or pd.Timestamp(window_end) <= current_ts:
        window_end = current_ts + pd.Timedelta(seconds=macro_sec)
    window_end = pd.Timestamp(window_end)
    if pd.notna(x.max()):
        window_end = min(window_end, pd.Timestamp(x.max()))
    if window_end <= current_ts:
        window_end = current_ts + pd.Timedelta(seconds=int(safe_float(cur.get("micro_window_sec", 10), 10)))

    fig.add_hrect(y0=0.0, y1=RISK_NORMAL_MAX, fillcolor="#ecfdf5", opacity=0.62, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_NORMAL_MAX, y1=RISK_WARNING_MAX, fillcolor="#fffbeb", opacity=0.56, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_WARNING_MAX, y1=1.05, fillcolor="#fff1f2", opacity=0.58, line_width=0, layer="below")

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
            name="Оценка риска" if not preview else "Proxy-риск",
            line={"color": "#2f80ed", "width": 3.0, "shape": "spline", "smoothing": 0.55},
            fill="tozeroy",
            fillcolor="rgba(47,128,237,0.13)",
            hovertemplate="%{x|%H:%M:%S}<br>риск=%{y:.2f}<extra></extra>",
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
        (RISK_NORMAL_MAX, "Норма (<0.3)", "#10b981", "dot"),
        (RISK_WARNING_MAX, "Риск (>0.7)", "#dc2626", "dot"),
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
            text=[f"{current_score:.2f}"],
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

    fail_ts = pd.to_datetime(cur.get("fail_start"), utc=True, errors="coerce")
    if pd.notna(fail_ts):
        fig.add_vline(x=pd.Timestamp(fail_ts), line_color="#7c3aed", line_width=1.2, line_dash="dash")
        fig.add_annotation(
            x=pd.Timestamp(fail_ts),
            y=0.98,
            text=f"Начало сбоя<br>{pd.Timestamp(fail_ts).strftime('%H:%M')}",
            showarrow=False,
            font={"size": 11, "color": "#7c3aed"},
            yanchor="bottom",
        )

    x_left = x.min()
    if pd.notna(x_left):
        fig.add_annotation(xref="paper", x=0.017, y=0.86, text="Риск", showarrow=False, bgcolor="#dc2626", font={"color": "white", "size": 11}, borderpad=4)
        fig.add_annotation(xref="paper", x=0.017, y=0.50, text="Предупреждение", showarrow=False, bgcolor="#fff7ed", font={"color": "#d97706", "size": 11}, bordercolor="#facc15", borderpad=4)
        fig.add_annotation(xref="paper", x=0.017, y=0.17, text="Норма", showarrow=False, bgcolor="#ecfdf5", font={"color": "#15803d", "size": 11}, bordercolor="#86efac", borderpad=4)

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
        title_text="Оценка риска",
        range=[0, 1.06],
        gridcolor="#edf2f7",
        zeroline=False,
        tickfont={"size": 11},
        showline=True,
        linecolor="#dbe5f1",
        fixedrange=True,
    )
    return fig


def build_fusion_figure(log_share: float, kpi_available: bool) -> go.Figure:
    log_share = float(np.clip(log_share, 0.0, 1.0))
    kpi_share = 0.0 if not kpi_available else max(0.0, 1.0 - log_share)
    if not kpi_available:
        log_share = 1.0
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Логи", "KPI"],
                values=[log_share, max(kpi_share, 1e-6)],
                hole=0.68,
                sort=False,
                textinfo="none",
                marker={"colors": ["#1d6ee8", "#6d28d9"]},
            )
        ]
    )
    fig.update_layout(
        height=166,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        paper_bgcolor="#ffffff",
        annotations=[{"text": "Вклад<br>fusion", "showarrow": False, "font": {"size": 12, "color": "#334155"}}],
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


def state_badge(state: str) -> str:
    state = state if state in STATE_LABELS else "normal"
    color = STATE_COLORS[state]
    return (
        f"<span style='display:inline-flex;border:1px solid {color}33;background:{color}12;"
        f"color:{color};border-radius:999px;padding:3px 8px;font-weight:750;font-size:12px'>"
        f"{html.escape(STATE_LABELS[state])}</span>"
    )
