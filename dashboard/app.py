from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from dashboard.runtime.contract import (
    DataContractError,
    checks_to_frame,
    load_compact_replay_bundle,
    load_contract_bundle,
)
from dashboard.runtime.inference import artifact_checks_frame, inspect_inference_bundle
from dashboard.ui.components import (
    anchor_badge_parts,
    build_fusion_figure,
    build_kpi_sparkline,
    build_risk_figure,
    chip_row,
    event_anchor_text,
    inject_css,
    render_honest_baseline_card_html,
    render_log_table_html,
    render_reliability_html,
    format_score,
    safe_float,
    state_from_score,
    top_card,
)
from dashboard.runtime.view_model import (
    CASE_ORDER,
    STATE_LABELS,
    build_dashboard_view,
    build_kpi_evidence,
    build_log_evidence,
    nearest_step,
    run_frame_for_case,
)

MAX_REPLAY_STEPS = 180


@st.cache_data(show_spinner=False)
def _load_contract_cached():
    return load_contract_bundle()


@st.cache_data(show_spinner=False)
def _inspect_inference_cached(cache_version: str = "prediction-store-v1"):
    return inspect_inference_bundle()


def _short_run(run_id: str, limit: int = 34) -> str:
    value = str(run_id)
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _format_time_window(row: pd.Series) -> str:
    start = pd.to_datetime(row.get("window_ts"), utc=True, errors="coerce")
    end = pd.to_datetime(row.get("window_end"), utc=True, errors="coerce")
    if pd.isna(start):
        return "n/a"
    if pd.isna(end):
        end = start + pd.Timedelta(seconds=int(safe_float(row.get("micro_window_sec", 10), 10)))
    return f"{pd.Timestamp(start).strftime('%H:%M:%S')}–{pd.Timestamp(end).strftime('%H:%M:%S')}"


def _init_state(cases: pd.DataFrame) -> None:
    if cases.empty:
        return
    if "selected_case_id" not in st.session_state:
        st.session_state.selected_case_id = str(cases.iloc[0]["case_id"])
    if "replay_step" not in st.session_state:
        st.session_state.replay_step = 0
    if "last_case_id" not in st.session_state:
        st.session_state.last_case_id = ""


def _selected_case(cases: pd.DataFrame) -> pd.Series:
    selected = cases.loc[cases["case_id"].eq(st.session_state.selected_case_id)]
    if selected.empty:
        st.session_state.selected_case_id = str(cases.iloc[0]["case_id"])
        selected = cases.loc[cases["case_id"].eq(st.session_state.selected_case_id)]
    return selected.iloc[0]


def current_context(cases: pd.DataFrame, timeline: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, int]:
    case = _selected_case(cases)
    run_df = run_frame_for_case(timeline, str(case["run_id"]), max_steps=MAX_REPLAY_STEPS)
    if run_df.empty:
        st.error("Для выбранного сценария нет окон.")
        st.stop()

    if st.session_state.last_case_id != st.session_state.selected_case_id:
        st.session_state.replay_step = nearest_step(run_df, pd.Timestamp(case["focus_window_ts"]))
        st.session_state.last_case_id = st.session_state.selected_case_id

    st.session_state.replay_step = int(np.clip(st.session_state.replay_step, 0, len(run_df) - 1))
    return case, run_df, st.session_state.replay_step


def _move_replay(delta: int, total_steps: int) -> None:
    st.session_state.replay_step = int(np.clip(st.session_state.replay_step + delta, 0, max(total_steps - 1, 0)))


def render_header(case: pd.Series, current: pd.Series, step_idx: int, total_steps: int, mode: str) -> None:
    state = str(current.get("alert_state", current.get("pred_state", "normal"))).lower()
    risk = safe_float(current.get("alert_score", current.get("model_score_raw", current.get("model_score", 0.0))), 0.0)
    tone = "danger" if state in {"alert", "incident"} else ("warning" if state == "watch" else "success")
    mode_value = "Scoring" if mode == "model_scoring" else "Preview"
    mode_sub = "финальная модель" if mode == "model_scoring" else "без model bundle"
    state_tone = tone
    risk_color = "#dc2626" if tone == "danger" else ("#d97706" if tone == "warning" else "#15803d")
    state_sub = f"шаг {step_idx + 1} / {total_steps}"
    lead_value = safe_float(current.get("time_to_failure_sec", current.get("run_event_lead_time_sec")), np.nan)
    if state in {"alert", "incident"} and np.isfinite(lead_value):
        state_sub = f"до инцидента: {lead_value:.0f} сек"
    status_cards = "".join(
        [
            top_card("Режим", mode_value, mode_sub, tone="blue"),
            top_card("Сценарий", _short_run(str(case["run_id"]), limit=19), f"Run #{int(case.get('run_number', 0)):03d}"),
            top_card("Окно", _format_time_window(current), "выбранное evidence-окно"),
            top_card("Состояние alert-слоя", STATE_LABELS.get(state, state), state_sub, tone=state_tone),
            (
                "<div class='top-card'>"
                "<div class='top-label'>Оценка alert-слоя</div>"
                f"<div class='top-risk' style='color:{risk_color}'>{format_score(risk)}</div>"
                "<div class='top-sub'>raw score, не вероятность</div>"
                "</div>"
            ),
        ]
    )
    st.markdown(
        f"""
        <div class="header-grid">
          <div class="app-head">
            <div class="logo-shield">∨</div>
            <div>
              <div class="app-title">LCMAN Dashboard</div>
              <div class="app-subtitle">Лабораторный интерфейс replay-инференса риска по логам и KPI</div>
            </div>
          </div>
          <div class="status-grid">{status_cards}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_controls(cases: pd.DataFrame, timeline: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, int]:
    case, run_df, _ = current_context(cases, timeline)

    with st.container(border=True):
        top = st.columns([2.25, 1.15, 1.8], gap="medium")
        with top[0]:
            st.markdown("<div class='control-title'>Демо-сценарии</div>", unsafe_allow_html=True)
            options = [str(row["case_id"]) for _, row in cases.iterrows()]
            short_labels = {
                "risk_escalation": "рост риска",
                "early_warning": "ранний сигнал",
                "degraded_fallback": "неполные данные",
                "normal_reference": "норма",
            }
            label_by_case = {
                str(row["case_id"]): short_labels.get(str(row["kind"]), str(row["label"]))
                for _, row in cases.iterrows()
            }
            if st.session_state.selected_case_id not in options:
                st.session_state.selected_case_id = options[0]
            selected = st.segmented_control(
                "Демо-сценарии",
                options,
                default=st.session_state.selected_case_id,
                format_func=lambda case_id: label_by_case.get(str(case_id), str(case_id)),
                key="case_segment",
                label_visibility="collapsed",
                width="stretch",
            )
            if selected and str(selected) != st.session_state.selected_case_id:
                st.session_state.selected_case_id = str(selected)
                st.session_state.autoplay = False
                st.rerun()
            st.markdown(f"<div class='case-note'>{html.escape(str(case['note']))}</div>", unsafe_allow_html=True)

        with top[1]:
            st.markdown("<div class='control-title'>Replay</div>", unsafe_allow_html=True)
            c_prev, c_next = st.columns(2, gap="small")
            with c_prev:
                if st.button("←", key="prev", use_container_width=True, disabled=st.session_state.replay_step <= 0, help="Предыдущий шаг"):
                    _move_replay(-1, len(run_df))
                    st.rerun()
            with c_next:
                if st.button("→", key="next", use_container_width=True, disabled=st.session_state.replay_step >= len(run_df) - 1, help="Следующий шаг"):
                    _move_replay(1, len(run_df))
                    st.rerun()

        with top[2]:
            real_windows = len(timeline.loc[timeline["run_id"].eq(str(case["run_id"]))])
            st.markdown(
                (
                    "<div class='replay-meta'>"
                    "<span><strong>Шкала replay</strong></span>"
                    f"<span class='replay-step-value'>{st.session_state.replay_step + 1} / {len(run_df)}</span>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            slider_value = st.slider(
                "Шаг replay",
                min_value=1,
                max_value=len(run_df),
                value=st.session_state.replay_step + 1,
                label_visibility="collapsed",
                key=f"slider_{st.session_state.selected_case_id}",
            )
            if slider_value - 1 != st.session_state.replay_step:
                st.session_state.replay_step = slider_value - 1
                st.session_state.autoplay = False
                st.rerun()
            st.caption(f"{st.session_state.replay_step + 1}/{len(run_df)} · реальные окна run: {real_windows}")

    case = _selected_case(cases)
    run_df = run_frame_for_case(timeline, str(case["run_id"]), max_steps=MAX_REPLAY_STEPS)
    step_idx = int(np.clip(st.session_state.replay_step, 0, len(run_df) - 1))
    return case, run_df, step_idx


def render_risk_panel(run_df: pd.DataFrame, step_idx: int, preview: bool) -> None:
    with st.container(border=True):
        st.markdown("<div class='section-title'>Оценка модели во времени</div>", unsafe_allow_html=True)
        fig = build_risk_figure(run_df, step_idx, preview=preview)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        cur = run_df.iloc[step_idx]
        anchor_ts = pd.to_datetime(cur.get("event_anchor", cur.get("fail_start")), utc=True, errors="coerce")
        anchor_title = event_anchor_text(cur)
        anchor_value = "n/a"
        if anchor_title != "Нет сбоя" and pd.notna(anchor_ts):
            anchor_value = pd.Timestamp(anchor_ts).strftime("%H:%M:%S")
        anchor_label, _ = anchor_badge_parts(cur.get("anchor_policy"))
        st.markdown(
            chip_row(
                [
                    ("Логи", "доступны" if int(safe_float(cur.get("log_available"), 0)) else "нет"),
                    ("KPI", "доступны" if int(safe_float(cur.get("kpi_available"), 0)) else "нет"),
                    (anchor_title, anchor_value),
                    ("Provenance", anchor_label),
                    ("Режим оценки", "proxy-preview" if preview else str(cur.get("score_source", "model_store"))),
                ]
            ),
            unsafe_allow_html=True,
        )


def render_log_panel(run_df: pd.DataFrame, step_idx: int) -> None:
    with st.container(border=True):
        st.markdown("<div class='section-title'>Лог-доказательства</div>", unsafe_allow_html=True)
        log_df = build_log_evidence(run_df, step_idx)
        st.markdown(render_log_table_html(log_df), unsafe_allow_html=True)
        st.markdown(
            "<div class='panel-caption'>Паттерны считаются по накопленному replay-контексту выбранного run.</div>",
            unsafe_allow_html=True,
        )
        excerpt = str(run_df.iloc[step_idx].get("log_text", "")).strip()
        excerpt = " ".join(excerpt.split())
        if len(excerpt) > 700:
            excerpt = excerpt[:700] + "…"
        with st.expander("Фрагмент логов выбранного окна", expanded=False):
            st.code(excerpt or "__NO_LOG__", language="text")


def render_kpi_panel(run_df: pd.DataFrame, step_idx: int, kpi_columns: list[str]) -> None:
    with st.container(border=True):
        st.markdown("<div class='section-title'>KPI-доказательства</div>", unsafe_allow_html=True)
        cur = run_df.iloc[step_idx]
        kpi_available = bool(int(safe_float(cur.get("kpi_available", 0), 0)))
        if not kpi_available:
            st.warning("KPI недоступны в текущем окне; ниже показан контекст run для degraded-mode.")
        evidence = build_kpi_evidence(run_df, kpi_columns, step_idx, top_n=8)
        if not evidence:
            st.info("Для выбранного run нет достаточных KPI-признаков в dashboard-срезе contract.")
            return

        def render_kpi_card(item: dict, idx: int, prefix: str) -> None:
            accent = str(item.get("accent", "#2563eb"))
            value_text = str(item.get("value_text", f"{safe_float(item.get('value')):.3g}"))
            y_values = pd.to_numeric(pd.Series(item.get("y", [])), errors="coerce")
            if item.get("unit") == "%":
                fill_width = float(np.clip(safe_float(item.get("value")), 0.0, 100.0))
            else:
                ymax = float(y_values.max()) if int(y_values.notna().sum()) else 0.0
                fill_width = float(np.clip((safe_float(item.get("value")) / max(ymax, 1e-6)) * 100.0, 0.0, 100.0))
            st.markdown(
                (
                    "<div class='kpi-tm-head'>"
                    "<div class='kpi-tm-title'>"
                    f"<span class='kpi-tm-dot' style='background:{accent}'></span>"
                    "<div>"
                    f"<div class='kpi-tm-name'>{html.escape(str(item.get('title', item.get('label', 'KPI'))))}</div>"
                    f"<div class='kpi-tm-subtitle'>{html.escape(str(item.get('subtitle', '')))}</div>"
                    "</div>"
                    "</div>"
                    "<div>"
                    f"<div class='kpi-tm-value' style='color:{accent}'>{html.escape(value_text)}</div>"
                    f"<div class='kpi-tm-delta'>Δ baseline: {html.escape(str(item.get('delta_text', 'n/a')))} · {html.escape(str(item.get('signal', 'n/a')))}</div>"
                    "</div>"
                    "</div>"
                    "<div class='kpi-tm-meter'>"
                    f"<div class='kpi-tm-fill' style='width:{fill_width:.1f}%; background:{accent}'></div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            fig = build_kpi_sparkline(item, current_ts=current_ts)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"{prefix}_{idx}_{item['column']}_{step_idx}",
            )

        current_ts = pd.Timestamp(cur["window_ts"])
        primary = evidence[:4]
        extra = evidence[4:]
        cols = st.columns(2, gap="small")
        for idx, item in enumerate(primary):
            with cols[idx % 2]:
                with st.container(border=True):
                    render_kpi_card(item, idx, "kpi_main")

        if extra:
            with st.expander("Дополнительные KPI-карточки", expanded=False):
                extra_cols = st.columns(2, gap="small")
                for idx, item in enumerate(extra):
                    with extra_cols[idx % 2]:
                        with st.container(border=True):
                            render_kpi_card(item, idx, "kpi_extra")
        st.markdown(
            "<div class='panel-caption'>Карточки агрегируют центральные KPI-семейства: CPU, память, диск и сеть. Серые/пустые состояния означают отсутствие данных в выбранном окне.</div>",
            unsafe_allow_html=True,
        )


def render_side_panel(current: pd.Series, view, inference) -> None:
    risk = safe_float(current.get("alert_score", current.get("model_score_raw", current.get("model_score", 0.0))), 0.0)
    rank_score = safe_float(current.get("model_score_rank", np.nan), np.nan)
    strict_risk = safe_float(current.get("risk_score_raw", current.get("risk_score", np.nan)), np.nan)
    horizon_300 = safe_float(current.get("model_score_raw_300s", current.get("model_score_raw", np.nan)), np.nan)
    horizon_120 = safe_float(current.get("model_score_120s", np.nan), np.nan)
    horizon_60 = safe_float(current.get("model_score_60s", np.nan), np.nan)
    horizon_30 = safe_float(current.get("risk_score_30s", current.get("risk_score_raw", np.nan)), np.nan)
    warning = safe_float(current.get("warning_score", risk), risk)
    state = str(current.get("alert_state", current.get("pred_state", state_from_score(risk)))).lower()
    model_label = str(view.dashboard_report.get("model_name", "финальная мультимодальная модель") or "финальная мультимодальная модель")
    runtime = view.dashboard_report.get("runtime", {}) if isinstance(view.dashboard_report, dict) else {}
    int8_status = str(runtime.get("onnx_int8", {}).get("status", "n/a")) if isinstance(runtime, dict) else "n/a"
    kpi_available = bool(int(safe_float(current.get("kpi_available", 0), 0)))
    log_available = bool(int(safe_float(current.get("log_available", 0), 0)))
    log_share = safe_float(current.get("fusion_log_share", np.nan), np.nan)
    kpi_share = safe_float(current.get("fusion_kpi_share", np.nan), np.nan)
    share_available = bool(int(safe_float(current.get("fusion_share_available", 0), 0)))
    share_available = share_available and np.isfinite(log_share) and np.isfinite(kpi_share) and (log_share + kpi_share) > 0
    if share_available:
        total_share = max(log_share + kpi_share, 1e-9)
        log_share = float(np.clip(log_share / total_share, 0.0, 1.0))
        kpi_share = float(np.clip(kpi_share / total_share, 0.0, 1.0))
    else:
        log_share, kpi_share = 0.5, 0.5
    log_only_score = safe_float(current.get("log_only_score", np.nan), np.nan)
    kpi_only_score = safe_float(current.get("kpi_only_score", np.nan), np.nan)
    anchor_label, anchor_color = anchor_badge_parts(current.get("anchor_policy"))

    with st.container(border=True):
        st.markdown("<div class='section-title'>Сводка alert-слоя</div>", unsafe_allow_html=True)
        rows = [
            ("Оценка alert-слоя", format_score(risk)),
            ("Ранг среди validation-окон", "n/a" if not np.isfinite(rank_score) else format_score(rank_score)),
            ("Strict risk-channel", "n/a" if not np.isfinite(strict_risk) else format_score(strict_risk)),
            ("Horizon 300s", "n/a" if not np.isfinite(horizon_300) else format_score(horizon_300)),
            ("Horizon 120s", "n/a" if not np.isfinite(horizon_120) else format_score(horizon_120)),
            ("Horizon 60s", "n/a" if not np.isfinite(horizon_60) else format_score(horizon_60)),
            ("Risk 30s", "n/a" if not np.isfinite(horizon_30) else format_score(horizon_30)),
            ("ONNX INT8", int8_status),
            ("Порог наблюдения", format_score(safe_float(current.get("watch_threshold", np.nan), np.nan))),
            ("Порог тревоги", format_score(safe_float(current.get("alert_threshold", current.get("threshold_risk", np.nan)), np.nan))),
            ("Состояние alert-слоя", STATE_LABELS.get(state, state)),
            ("Диагностический warning", format_score(warning)),
            ("Разметка", anchor_label),
            ("Модель", "Preview" if view.mode != "model_scoring" else f"Финальная модель: {model_label}"),
            ("Событие run", str(current.get("run_event_label", "n/a") or "n/a")),
            ("Lead time", "n/a" if not np.isfinite(safe_float(current.get("time_to_failure_sec", current.get("run_event_lead_time_sec")), np.nan)) else f"{safe_float(current.get('time_to_failure_sec', current.get('run_event_lead_time_sec')), np.nan):.0f} сек"),
        ]
        html_rows = "".join(
            f"<div class='side-row'><div class='side-k'>{html.escape(k)}</div><div class='side-v'>{html.escape(v)}</div></div>"
            for k, v in rows
        )
        html_rows += (
            "<div class='anchor-pill'>"
            f"<span class='anchor-dot' style='background:{anchor_color}'></span>"
            f"{html.escape(event_anchor_text(current))}: {html.escape(anchor_label)}"
            "</div>"
        )
        st.markdown(html_rows, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='section-title'>Сводка по объединению</div>", unsafe_allow_html=True)
        left, right = st.columns([0.86, 1.14], gap="small")
        with left:
            st.plotly_chart(
                build_fusion_figure(log_share, kpi_share, share_available),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with right:
            log_share_text = f"{log_share * 100:.0f}%" if share_available else "н/д"
            kpi_share_text = f"{kpi_share * 100:.0f}%" if share_available else "н/д"
            log_score_text = format_score(log_only_score) if np.isfinite(log_only_score) else "н/д"
            kpi_score_text = format_score(kpi_only_score) if np.isfinite(kpi_only_score) else "н/д"
            st.markdown(
                (
                    f"<div class='side-row'><div class='side-k'>Баланс log-head</div><div class='side-v'>{log_share_text}</div></div>"
                    f"<div class='side-row'><div class='side-k'>Баланс KPI-head</div><div class='side-v'>{kpi_share_text}</div></div>"
                    f"<div class='side-row'><div class='side-k'>log-only score</div><div class='side-v'>{log_score_text}</div></div>"
                    f"<div class='side-row'><div class='side-k'>KPI-only score</div><div class='side-v'>{kpi_score_text}</div></div>"
                    f"<div class='side-row'><div class='side-k'>Режим</div><div class='side-v'>{html.escape(str(current.get('fusion_display_mode', 'log+kpi' if log_available and kpi_available else 'fallback')))}</div></div>"
                ),
                unsafe_allow_html=True,
            )
        st.caption("Это диагностический баланс auxiliary log-head/KPI-head. Он не является причинной атрибуцией и не доказывает качество отдельного baseline.")

    with st.container(border=True):
        st.markdown("<div class='section-title'>Пояснение</div>", unsafe_allow_html=True)
        if view.mode == "model_scoring":
            model_rows = int(view.timeline["score_source"].eq("model_store").sum()) if "score_source" in view.timeline.columns else 0
            text = (
                f"Dashboard использует финальный replay bundle для {model_rows} окон. "
                "Оценка модели отделена от решения alert-слоя и фактического инцидента."
            )
        else:
            text = (
                "Compact model bundle отсутствует, поэтому интерфейс не утверждает модельный prediction. "
                "Текущий replay показывает, как будет выглядеть лабораторный инференс на contract-окнах после запуска notebook 03 в train-режиме."
            )
        st.markdown(f"<div class='explain-box'>{html.escape(text)}</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("<div class='section-title'>Сравнение с baseline</div>", unsafe_allow_html=True)
        target = st.segmented_control(
            "Цель baseline",
            ["y_risk", "y_warning"],
            default="y_risk",
            format_func=lambda value: "risk" if value == "y_risk" else "warning",
            key="baseline_target",
            label_visibility="collapsed",
            width="stretch",
        )
        st.markdown(
            render_honest_baseline_card_html(view.baseline_table.copy(), view.dashboard_report, str(target), current),
            unsafe_allow_html=True,
        )
        if inference.missing_required:
            st.caption("Central fusion не сравнивается с baseline, пока отсутствует compact model bundle.")

    with st.container(border=True):
        st.markdown("<div class='section-title'>Надежность модели</div>", unsafe_allow_html=True)
        st.markdown(render_reliability_html(view.dashboard_report, view.calibration_bins), unsafe_allow_html=True)


def render_about(bundle, inference, view) -> None:
    st.markdown("---")
    with st.expander("Model / About / Data contract", expanded=False):
        st.markdown("#### Contract artifacts")
        st.dataframe(checks_to_frame(bundle.artifact_checks), use_container_width=True, hide_index=True)

        st.markdown("#### Inference bundle readiness")
        st.dataframe(artifact_checks_frame(inference.artifact_checks), use_container_width=True, hide_index=True)

        st.markdown("#### Contract summary")
        summary_rows = [
            {"Показатель": "rows", "Значение": str(view.stats["rows"])},
            {"Показатель": "runs", "Значение": str(view.stats["runs"])},
            {"Показатель": "workloads", "Значение": str(view.stats["workloads"])},
            {"Показатель": "micro window, сек", "Значение": str(view.stats["micro_window_sec"])},
            {"Показатель": "macro window, сек", "Значение": str(view.stats["macro_window_sec"])},
            {"Показатель": "warning share", "Значение": f"{view.stats['warning_share']:.3f}"},
            {"Показатель": "risk share", "Значение": f"{view.stats['risk_share']:.3f}"},
            {"Показатель": "log coverage", "Значение": f"{view.stats['log_available_share']:.3f}"},
            {"Показатель": "KPI coverage", "Значение": f"{view.stats['kpi_available_share']:.3f}"},
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        debug = inference.debug_metadata.get("model_config", {})
        if debug:
            st.markdown("#### Dry-run model config")
            st.json(debug, expanded=False)

        if view.dashboard_report:
            st.markdown("#### Dashboard diagnostics")
            st.json(view.dashboard_report, expanded=False)

        quality = view.dashboard_report.get("bundle_quality_report", {}) if isinstance(view.dashboard_report, dict) else {}
        if quality:
            st.markdown("#### Bundle quality summary")
            score_distribution = quality.get("score_distribution", {})
            state_counts = quality.get("alert_state_counts", {})
            raw_summary = score_distribution.get("test_model_score_raw", score_distribution.get("test_horizon_300s", {}))
            rank_summary = score_distribution.get("test_model_score_rank", score_distribution.get("validation_horizon_300s", {}))
            quality_rows = [
                {"Показатель": "proxy windows", "Значение": str(quality.get("proxy_windows", "n/a"))},
                {"Показатель": "alert states", "Значение": str(state_counts)},
                {"Показатель": "raw score p50", "Значение": str(score_distribution.get("test_model_score_raw", {}).get("p50", "n/a"))},
                {"Показатель": "rank score p50", "Значение": str(score_distribution.get("test_model_score_rank", {}).get("p50", "n/a"))},
            ]
            st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)

        if not view.seed_metrics.empty:
            st.markdown("#### Seed metrics")
            st.dataframe(view.seed_metrics, use_container_width=True, hide_index=True)

        if not view.calibration_bins.empty:
            st.markdown("#### Calibration bins")
            st.dataframe(view.calibration_bins, use_container_width=True, hide_index=True)

        if inference.model_card:
            st.markdown("#### Model card")
            st.markdown(inference.model_card)


def render_setup_required(inference) -> None:
    st.markdown(
        """
        <div class="header-grid">
          <div class="app-head">
            <div class="logo-shield">L</div>
            <div>
              <div class="app-title">LCMAN Dashboard</div>
              <div class="app-subtitle">Финальный replay-интерфейс требует bundle из dashboard/artifacts.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("Финальный bundle не найден. Дашборд больше не использует proxy-режим и старые outputs.")
    st.code(
        "\n".join(
            [
                "python -m dashboard.pipeline.build_contract --raw-data raw_data",
                "python -m dashboard.training.baselines --contract-path dashboard/artifacts/contract/strict_intensity_contract.csv.gz",
                "python -m dashboard.training.final_model --device cuda --contract-path dashboard/artifacts/contract/strict_intensity_contract.csv.gz",
                "python -m dashboard.export.bundle --mode protocol_fixed",
                "streamlit run dashboard/app.py",
            ]
        ),
        language="powershell",
    )
    st.dataframe(artifact_checks_frame(inference.artifact_checks), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="LCMAN Dashboard",
        page_icon="L",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(inject_css(), unsafe_allow_html=True)

    try:
        inference = _inspect_inference_cached()
        if inference.ready and not inference.dashboard_replay_store.empty:
            bundle = load_compact_replay_bundle(inference.dashboard_replay_store, inference.model_dir)
        else:
            render_setup_required(inference)
            st.stop()
    except DataContractError as exc:
        st.error(f"Ошибка data contract: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Dashboard не смог загрузить артефакты: {exc}")
        st.stop()

    view = build_dashboard_view(bundle, inference)
    if view.timeline.empty or view.cases.empty:
        st.error("Недостаточно данных для replay dashboard.")
        st.stop()

    _init_state(view.cases)
    case, run_df, step_idx = current_context(view.cases, view.timeline)
    current = run_df.iloc[step_idx]

    render_header(case, current, step_idx, len(run_df), view.mode)
    case, run_df, step_idx = render_controls(view.cases, view.timeline)
    current = run_df.iloc[step_idx]
    source_label = "replay_bundle" if bundle.contract_path.name.startswith("replay_bundle") else ("dashboard_replay_store" if bundle.contract_path.name.startswith("dashboard_replay_store") else "strict_intensity contract")
    st.markdown(
        chip_row(
            [
                ("Кейс", str(case["label"])),
                ("run_id", str(case["run_id"])),
                ("Источник", source_label),
                ("Bundle", "готов" if inference.ready else "не обучен"),
            ]
        ),
        unsafe_allow_html=True,
    )
    mode_note = (
        view.mode_note
        if view.mode != "model_scoring"
        else view.mode_note
    )
    st.markdown(f"<div class='mode-note'>{html.escape(mode_note)}</div>", unsafe_allow_html=True)

    main_cols = st.columns([3.3, 1.0], gap="medium")
    with main_cols[0]:
        render_risk_panel(run_df, step_idx, preview=view.mode != "model_scoring")
        lower = st.columns([1.05, 1.75], gap="medium")
        with lower[0]:
            render_log_panel(run_df, step_idx)
        with lower[1]:
            render_kpi_panel(run_df, step_idx, view.kpi_columns)

    with main_cols[1]:
        render_side_panel(current, view, inference)

    render_about(bundle, inference, view)


if __name__ == "__main__":
    main()
