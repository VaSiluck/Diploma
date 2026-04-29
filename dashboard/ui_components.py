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


def inject_css() -> str:
    return """
    <style>
    :root {
        --blue:#1d6ee8;
        --blue-dark:#1557bd;
        --ink:#102033;
        --muted:#64748b;
        --line:#e6edf5;
        --panel:#ffffff;
        --page:#f6f8fb;
    }
    .stApp {
        background: var(--page);
        color: var(--ink);
    }
    header[data-testid="stHeader"], div[data-testid="stToolbar"], #MainMenu, footer {
        display:none;
        visibility:hidden;
    }
    .block-container {
        max-width: 1480px;
        padding-top: 0.65rem;
        padding-bottom: 0.8rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        border-radius: 8px !important;
        background: #ffffff !important;
        box-shadow: 0 1px 2px rgba(16,32,51,0.04) !important;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span {
        color: var(--ink);
    }
    h3 {
        font-size: 1.02rem;
        margin-top: 0.1rem;
        margin-bottom: 0.35rem;
    }
    .app-head {
        display:flex;
        align-items:center;
        gap:12px;
        min-height:58px;
    }
    .logo-shield {
        width:42px;
        height:42px;
        border-radius:10px;
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
        font-size:27px;
        font-weight:750;
        line-height:1.0;
        letter-spacing:0;
    }
    .app-subtitle {
        color:var(--muted);
        font-size:12px;
        margin-top:5px;
    }
    .top-card {
        border:1px solid var(--line);
        border-radius:8px;
        background:#fff;
        padding:10px 12px;
        min-height:58px;
        display:flex;
        flex-direction:column;
        justify-content:center;
    }
    .top-label {
        color:#536176;
        font-size:12px;
        line-height:1.05;
        white-space:nowrap;
    }
    .top-value {
        font-size:18px;
        line-height:1.2;
        font-weight:750;
        margin-top:4px;
        color:var(--ink);
    }
    .top-sub {
        color:#64748b;
        font-size:11px;
        margin-top:2px;
    }
    .top-risk {
        color:#dc2626;
        font-size:26px;
        font-weight:800;
        line-height:1;
    }
    .case-note {
        border:1px solid #d8e8ff;
        background:#f4f9ff;
        color:#255a99;
        border-radius:8px;
        padding:8px 10px;
        font-size:12px;
        margin-top:4px;
    }
    .stButton > button {
        border-radius:6px;
        border:1px solid #d7e2ef;
        background:#fff;
        color:#26384d;
        min-height:34px;
        font-size:12px;
        font-weight:650;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }
    .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
        background:#0f73e8 !important;
        border-color:#0f73e8 !important;
        color:#fff !important;
    }
    .stButton > button:hover {
        border-color:#93c5fd;
        color:#0f73e8;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background:#0f73e8 !important;
        border-color:#0f73e8 !important;
        box-shadow:0 0 0 4px rgba(15,115,232,0.13);
    }
    .replay-dots {
        display:grid;
        grid-template-columns: repeat(auto-fit, minmax(0, 1fr));
        gap:8px;
        align-items:center;
        margin-top:6px;
    }
    .replay-dot {
        height:7px;
        border-radius:999px;
        background:#d4dce8;
    }
    .replay-dot.done {
        background:#5aa2f2;
    }
    .replay-dot.current {
        height:13px;
        background:#0f73e8;
        box-shadow:0 0 0 3px rgba(15,115,232,0.15);
    }
    .chip-row {
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin:5px 0 2px 0;
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
        grid-template-columns:34px 1.55fr .75fr .9fr .9fr;
        gap:8px;
        align-items:center;
        padding:8px 10px;
        border-bottom:1px solid #eef3f8;
        font-size:12px;
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
    .small-muted {
        color:#64748b;
        font-size:12px;
    }
    .kpi-card {
        border:1px solid #e6edf5;
        border-radius:8px;
        background:#fff;
        padding:8px 9px 6px 9px;
        min-height:188px;
    }
    .kpi-title {
        display:flex;
        justify-content:space-between;
        align-items:baseline;
        gap:8px;
        font-size:12px;
        font-weight:750;
    }
    .kpi-value {
        color:#4f46e5;
        font-size:12px;
        font-weight:750;
        white-space:nowrap;
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

    fig.add_hrect(y0=0.0, y1=RISK_NORMAL_MAX, fillcolor="#ecfdf5", opacity=0.72, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_NORMAL_MAX, y1=RISK_WARNING_MAX, fillcolor="#fffbeb", opacity=0.72, line_width=0, layer="below")
    fig.add_hrect(y0=RISK_WARNING_MAX, y1=1.05, fillcolor="#fff1f2", opacity=0.72, line_width=0, layer="below")

    kpi_missing = pd.to_numeric(run_df.get("kpi_available", 1), errors="coerce").fillna(1).eq(0).to_numpy()
    start = None
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

    fig.add_trace(
        go.Scatter(
            x=x,
            y=risk,
            mode="lines+markers",
            name="Оценка риска" if not preview else "Proxy-риск",
            line={"color": "#2f80ed", "width": 2.4, "shape": "spline", "smoothing": 0.45},
            marker={"size": 4.5, "color": "#2f80ed"},
            hovertemplate="%{x|%H:%M:%S}<br>риск=%{y:.2f}<extra></extra>",
        )
    )

    for value, label, color, dash in [
        (RISK_NORMAL_MAX, "Норма (<0.3)", "#10b981", "dot"),
        (RISK_WARNING_MAX, "Аномалия (>0.7)", "#dc2626", "dot"),
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

    current_ts = pd.Timestamp(cur["window_ts"])
    fig.add_vrect(
        x0=current_ts,
        x1=current_ts + pd.Timedelta(seconds=int(safe_float(cur.get("micro_window_sec", 10), 10))),
        fillcolor="#dbeafe",
        opacity=0.72,
        line_width=0,
        layer="below",
    )
    fig.add_vline(x=current_ts, line_color="#0f73e8", line_width=1.5)
    fig.add_trace(
        go.Scatter(
            x=[current_ts],
            y=[safe_float(cur.get("risk_score"))],
            mode="markers+text",
            text=[f"{safe_float(cur.get('risk_score')):.2f}"],
            textposition="middle right",
            name="Выбранное окно",
            marker={"size": 10, "color": "#ffffff", "line": {"color": "#0f73e8", "width": 2}},
            hoverinfo="skip",
        )
    )

    fail_ts = pd.to_datetime(cur.get("fail_start"), utc=True, errors="coerce")
    if pd.notna(fail_ts):
        fig.add_vline(x=pd.Timestamp(fail_ts), line_color="#7c3aed", line_width=1.2, line_dash="dash")
        fig.add_annotation(
            x=pd.Timestamp(fail_ts),
            y=0.98,
            text="Начало сбоя",
            showarrow=False,
            font={"size": 11, "color": "#7c3aed"},
            yanchor="bottom",
        )

    x_left = x.min()
    if pd.notna(x_left):
        fig.add_annotation(x=x_left, y=0.86, text="Риск", showarrow=False, bgcolor="#dc2626", font={"color": "white", "size": 11}, borderpad=4)
        fig.add_annotation(x=x_left, y=0.50, text="Предупреждение", showarrow=False, bgcolor="#fff7ed", font={"color": "#d97706", "size": 11}, bordercolor="#facc15", borderpad=4)
        fig.add_annotation(x=x_left, y=0.17, text="Норма", showarrow=False, bgcolor="#ecfdf5", font={"color": "#15803d", "size": 11}, bordercolor="#86efac", borderpad=4)

    fig.update_layout(
        height=350,
        margin={"l": 16, "r": 8, "t": 8, "b": 8},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "x": 0.18, "y": 1.09, "font": {"size": 11}},
        font={"color": "#334155"},
    )
    fig.update_xaxes(tickformat="%H:%M", gridcolor="#edf2f7", zeroline=False)
    fig.update_yaxes(title_text="Оценка риска", range=[0, 1.04], gridcolor="#edf2f7", zeroline=False)
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
        height=142,
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
    if int(valid.sum()) >= 2:
        fig.add_trace(
            go.Scatter(
                x=x[valid],
                y=y[valid],
                mode="lines",
                line={"color": "#6d5dfc", "width": 1.6, "shape": "spline", "smoothing": 0.45},
                fill="tozeroy",
                fillcolor="rgba(109,93,252,0.12)",
                showlegend=False,
                hovertemplate="%{x|%H:%M}<br>%{y:.3g}<extra></extra>",
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
    fig.update_layout(
        height=112,
        margin={"l": 3, "r": 3, "t": 4, "b": 2},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )
    fig.update_xaxes(showticklabels=False, showgrid=True, gridcolor="#edf2f7", zeroline=False, fixedrange=True)
    fig.update_yaxes(showticklabels=False, showgrid=True, gridcolor="#edf2f7", zeroline=False, fixedrange=True)
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
        freq_txt = f"{freq:,}".replace(",", " ")
        rows.append(
            "<div class='evidence-row'>"
            f"<div class='rank'>{rank}</div>"
            f"<div>{html.escape(label)}</div>"
            f"<div>{freq_txt}</div>"
            f"<div class='{delta_cls}'>{delta:+.0f}%</div>"
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


def state_badge(state: str) -> str:
    state = state if state in STATE_LABELS else "normal"
    color = STATE_COLORS[state]
    return (
        f"<span style='display:inline-flex;border:1px solid {color}33;background:{color}12;"
        f"color:{color};border-radius:999px;padding:3px 8px;font-weight:750;font-size:12px'>"
        f"{html.escape(STATE_LABELS[state])}</span>"
    )
