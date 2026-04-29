from __future__ import annotations

import html
import time

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.contract_pipeline import (
    DataContractError,
    checks_to_frame,
    load_contract_bundle,
)
from dashboard.inference import artifact_checks_frame, inspect_inference_bundle
from dashboard.ui_components import (
    build_fusion_figure,
    build_kpi_sparkline,
    build_risk_figure,
    chip_row,
    inject_css,
    replay_dots,
    render_log_table_html,
    safe_float,
    state_from_score,
    top_card,
)
from dashboard.view_model import (
    CASE_ORDER,
    STATE_LABELS,
    build_dashboard_view,
    build_kpi_evidence,
    build_log_evidence,
    nearest_step,
    run_frame_for_case,
)


REFRESH_SECONDS = 0.65


@st.cache_data(show_spinner=False)
def _load_contract_cached():
    return load_contract_bundle()


@st.cache_data(show_spinner=False)
def _inspect_inference_cached():
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
    if "autoplay" not in st.session_state:
        st.session_state.autoplay = False
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
    run_df = run_frame_for_case(timeline, str(case["run_id"]), max_steps=36)
    if run_df.empty:
        st.error("Для выбранного сценария нет окон.")
        st.stop()

    if st.session_state.last_case_id != st.session_state.selected_case_id:
        st.session_state.replay_step = nearest_step(run_df, pd.Timestamp(case["focus_window_ts"]))
        st.session_state.last_case_id = st.session_state.selected_case_id

    st.session_state.replay_step = int(np.clip(st.session_state.replay_step, 0, len(run_df) - 1))
    return case, run_df, st.session_state.replay_step


def render_header(case: pd.Series, current: pd.Series, step_idx: int, total_steps: int, mode: str) -> None:
    state = str(current.get("pred_state", "normal"))
    risk = safe_float(current.get("risk_score", 0.0), 0.0)
    tone = "danger" if risk >= 0.70 else ("warning" if risk >= 0.30 else "success")
    mode_value = "Scoring" if mode == "model_scoring" else "Preview"
    mode_sub = "central fusion" if mode == "model_scoring" else "без model bundle"

    cols = st.columns([2.7, 1.02, 1.65, 1.28, 0.92, 1.55, 0.86], gap="small")
    with cols[0]:
        st.markdown(
            """
            <div class="app-head">
              <div class="logo-shield">∨</div>
              <div>
                <div class="app-title">LCMAN Dashboard</div>
                <div class="app-subtitle">Лабораторный интерфейс инференса риска для multimodal telemetry</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(top_card("Режим", mode_value, mode_sub, tone="blue"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(top_card("Сценарий", _short_run(str(case["run_id"])), f"Run #{int(case.get('run_number', 0)):03d}"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(top_card("Окно", _format_time_window(current), "выбранное evidence-окно"), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(top_card("Шаг", f"{step_idx + 1} / {total_steps}", "replay"), unsafe_allow_html=True)
    with cols[5]:
        state_tone = "danger" if state == "risk" else ("warning" if state == "warning" else "success")
        st.markdown(top_card("Текущее состояние", STATE_LABELS.get(state, state), "по текущему окну", tone=state_tone), unsafe_allow_html=True)
    with cols[6]:
        st.markdown(
            (
                "<div class='top-card'>"
                "<div class='top-label'>Оценка риска</div>"
                f"<div class='top-risk' style='color:{'#dc2626' if tone == 'danger' else ('#d97706' if tone == 'warning' else '#15803d')}'>{risk:.2f}</div>"
                "<div class='top-sub'>текущий шаг</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_controls(cases: pd.DataFrame, timeline: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, int]:
    case, run_df, _ = current_context(cases, timeline)

    top = st.columns([3.15, 2.2, 2.65], gap="small")
    with top[0]:
        st.markdown("### Демо-сценарии")
        by_kind = {str(row["kind"]): row for _, row in cases.iterrows()}
        btn_cols = st.columns([1.25, 1.14, 1.02, 0.75], gap="small")
        for idx, kind in enumerate(CASE_ORDER):
            row = by_kind.get(kind)
            with btn_cols[idx]:
                if row is None:
                    st.button(kind, disabled=True, use_container_width=True, key=f"case_disabled_{kind}")
                    continue
                active = str(row["case_id"]) == st.session_state.selected_case_id
                if st.button(
                    str(row["label"]),
                    key=f"case_{kind}",
                    type="primary" if active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_case_id = str(row["case_id"])
                    st.session_state.autoplay = False
                    st.rerun()
        st.markdown(f"<div class='case-note'>{html.escape(str(case['note']))}</div>", unsafe_allow_html=True)

    with top[1]:
        st.markdown("### Управление replay")
        c_prev, c_play, c_pause, c_next = st.columns(4, gap="small")
        with c_prev:
            if st.button("◀", key="prev", use_container_width=True, disabled=st.session_state.replay_step <= 0):
                st.session_state.replay_step -= 1
                st.session_state.autoplay = False
                st.rerun()
        with c_play:
            if st.button("▶ Play", key="play", use_container_width=True):
                st.session_state.autoplay = True
                st.rerun()
        with c_pause:
            if st.button("Ⅱ Pause", key="pause", use_container_width=True, type="primary" if st.session_state.autoplay else "secondary"):
                st.session_state.autoplay = False
                st.rerun()
        with c_next:
            if st.button("▶", key="next", use_container_width=True, disabled=st.session_state.replay_step >= len(run_df) - 1):
                st.session_state.replay_step += 1
                st.session_state.autoplay = False
                st.rerun()

    with top[2]:
        st.markdown("### Шкала replay")
        st.markdown(replay_dots(len(run_df), st.session_state.replay_step), unsafe_allow_html=True)
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
        st.caption(f"{st.session_state.replay_step + 1}/{len(run_df)} · реальные окна run: {len(timeline.loc[timeline['run_id'].eq(str(case['run_id']))])}")

    case = _selected_case(cases)
    run_df = run_frame_for_case(timeline, str(case["run_id"]), max_steps=36)
    step_idx = int(np.clip(st.session_state.replay_step, 0, len(run_df) - 1))
    return case, run_df, step_idx


def render_risk_panel(run_df: pd.DataFrame, step_idx: int, preview: bool) -> None:
    with st.container(border=True):
        st.markdown("### Оценка риска во времени")
        fig = build_risk_figure(run_df, step_idx, preview=preview)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        cur = run_df.iloc[step_idx]
        fail_ts = pd.to_datetime(cur.get("fail_start"), utc=True, errors="coerce")
        st.markdown(
            chip_row(
                [
                    ("Логи", "доступны" if int(safe_float(cur.get("log_available"), 0)) else "нет"),
                    ("KPI", "доступны" if int(safe_float(cur.get("kpi_available"), 0)) else "нет"),
                    ("Начало сбоя", pd.Timestamp(fail_ts).strftime("%H:%M:%S") if pd.notna(fail_ts) else "n/a"),
                    ("Режим оценки", "proxy-preview" if preview else "model-scoring"),
                ]
            ),
            unsafe_allow_html=True,
        )


def render_log_panel(run_df: pd.DataFrame, step_idx: int) -> None:
    st.markdown("### Лог-доказательства")
    log_df = build_log_evidence(run_df, step_idx)
    st.markdown(render_log_table_html(log_df), unsafe_allow_html=True)
    st.caption("Паттерны считаются по накопленному replay-контексту и ближайшим реальным окнам contract.")
    excerpt = str(run_df.iloc[step_idx].get("log_text", "")).strip()
    excerpt = " ".join(excerpt.split())
    if len(excerpt) > 700:
        excerpt = excerpt[:700] + "…"
    with st.expander("Фрагмент логов выбранного окна", expanded=False):
        st.code(excerpt or "__NO_LOG__", language="text")


def render_kpi_panel(run_df: pd.DataFrame, step_idx: int, kpi_columns: list[str]) -> None:
    st.markdown("### KPI-доказательства")
    cur = run_df.iloc[step_idx]
    kpi_available = bool(int(safe_float(cur.get("kpi_available", 0), 0)))
    if not kpi_available:
        st.warning("В текущем окне KPI недоступны: доказательства ниже показывают контекст run, а не прямое подтверждение решения.")
    evidence = build_kpi_evidence(run_df, kpi_columns, step_idx, top_n=6)
    if not evidence:
        st.info("Для выбранного run нет достаточных KPI-признаков в dashboard-срезе contract.")
        return
    cols = st.columns(2, gap="small")
    current_ts = pd.Timestamp(cur["window_ts"])
    for idx, item in enumerate(evidence):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(
                    (
                        f"<div class='kpi-title'><span>{html.escape(str(item['label']))}</span>"
                        f"<span class='kpi-value'>{safe_float(item['value']):.3g}</span></div>"
                        f"<div class='small-muted'>Δ baseline: {safe_float(item['delta_pct']):+.1f}% · сигнал: {html.escape(str(item['signal']))}</div>"
                    ),
                    unsafe_allow_html=True,
                )
                fig = build_kpi_sparkline(item, current_ts=current_ts)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=f"kpi_{idx}_{item['column']}_{step_idx}")
    st.caption("KPI-карточки используют выбранные агрегаты из `aligned_multirate_windows.csv`; серые зоны на графике риска показывают пропуски KPI.")


def render_side_panel(current: pd.Series, view, inference) -> None:
    risk = safe_float(current.get("risk_score", 0.0), 0.0)
    warning = safe_float(current.get("warning_score", risk), risk)
    state = str(current.get("pred_state", state_from_score(risk)))
    kpi_available = bool(int(safe_float(current.get("kpi_available", 0), 0)))
    log_share = safe_float(current.get("fusion_gate", 0.5), 0.5)
    kpi_share = 0.0 if not kpi_available else max(0.0, 1.0 - log_share)

    with st.container(border=True):
        st.markdown("### Сводка по риску")
        rows = [
            ("Текущая оценка риска", f"{risk:.2f}"),
            ("Warning score", f"{warning:.2f}"),
            ("Состояние", STATE_LABELS.get(state, state)),
            ("Уверенность", "n/a" if view.mode != "model_scoring" else f"{safe_float(current.get('confidence', 0.0)):.2f}"),
            ("Режим", "Preview" if view.mode != "model_scoring" else "Central fusion"),
        ]
        html_rows = "".join(
            f"<div class='side-row'><div class='side-k'>{html.escape(k)}</div><div class='side-v'>{html.escape(v)}</div></div>"
            for k, v in rows
        )
        st.markdown(html_rows, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Сводка по объединению")
        left, right = st.columns([0.86, 1.14], gap="small")
        with left:
            st.plotly_chart(build_fusion_figure(log_share, kpi_available), use_container_width=True, config={"displayModeBar": False})
        with right:
            st.markdown(
                (
                    f"<div class='side-row'><div class='side-k'>Логи</div><div class='side-v'>{log_share * 100:.0f}%</div></div>"
                    f"<div class='side-row'><div class='side-k'>KPI</div><div class='side-v'>{kpi_share * 100:.0f}%</div></div>"
                    f"<div class='side-row'><div class='side-k'>Fallback</div><div class='side-v'>{'да' if not kpi_available else 'нет'}</div></div>"
                ),
                unsafe_allow_html=True,
            )
        st.caption("В preview это диагностическая оценка вклада по доступности модальностей; после train-run здесь будет `fusion_gate` модели.")

    with st.container(border=True):
        st.markdown("### Пояснение")
        if view.mode == "model_scoring":
            text = "Модельный compact bundle найден. Следующий срез подключит прямое восстановление `CentralFusionMultiRate` и сохранённого preprocessing."
        else:
            text = (
                "Compact model bundle отсутствует, поэтому интерфейс не утверждает модельный prediction. "
                "Текущий replay показывает, как будет выглядеть лабораторный инференс на contract-окнах после запуска notebook 03 в train-режиме."
            )
        st.markdown(f"<div class='explain-box'>{html.escape(text)}</div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Сравнение с baseline")
        baseline = view.baseline_table.copy()
        if baseline.empty:
            st.info("Baseline-сводка недоступна.")
        else:
            show = baseline.loc[baseline.get("target", pd.Series(dtype=str)).astype(str).eq("y_risk")].copy()
            if show.empty:
                show = baseline.copy()
            rename = {
                "baseline": "Модель",
                "target": "Цель",
                "average_precision": "AP",
                "roc_auc": "ROC-AUC",
                "f1": "F1",
            }
            show = show.rename(columns=rename)
            keep = [col for col in ["Модель", "Цель", "AP", "ROC-AUC", "F1"] if col in show.columns]
            st.dataframe(show[keep].head(6), use_container_width=True, hide_index=True)
        if inference.missing_required:
            st.caption("Central fusion не сравнивается с baseline, пока отсутствует compact model bundle.")


def render_about(bundle, inference, view) -> None:
    st.markdown("---")
    with st.expander("Model / About / Data contract", expanded=False):
        st.markdown("#### Contract artifacts")
        st.dataframe(checks_to_frame(bundle.artifact_checks), use_container_width=True, hide_index=True)

        st.markdown("#### Inference bundle readiness")
        st.dataframe(artifact_checks_frame(inference.artifact_checks), use_container_width=True, hide_index=True)

        st.markdown("#### Contract summary")
        summary_rows = [
            {"Показатель": "rows", "Значение": view.stats["rows"]},
            {"Показатель": "runs", "Значение": view.stats["runs"]},
            {"Показатель": "workloads", "Значение": view.stats["workloads"]},
            {"Показатель": "micro window, сек", "Значение": view.stats["micro_window_sec"]},
            {"Показатель": "macro window, сек", "Значение": view.stats["macro_window_sec"]},
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

        if inference.model_card:
            st.markdown("#### Model card")
            st.markdown(inference.model_card)


def main() -> None:
    st.set_page_config(
        page_title="LCMAN Dashboard",
        page_icon="L",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(inject_css(), unsafe_allow_html=True)

    try:
        bundle = _load_contract_cached()
        inference = _inspect_inference_cached()
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
    st.markdown(
        chip_row(
            [
                ("Кейс", str(case["label"])),
                ("run_id", str(case["run_id"])),
                ("Источник", "raw_multirate contract"),
                ("Bundle", "готов" if inference.ready else "не обучен"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption(view.mode_note)

    main_cols = st.columns([3.75, 1.28], gap="small")
    with main_cols[0]:
        render_risk_panel(run_df, step_idx, preview=view.mode != "model_scoring")
        lower = st.columns([1.18, 1.82], gap="small")
        with lower[0]:
            with st.container(border=True):
                render_log_panel(run_df, step_idx)
        with lower[1]:
            with st.container(border=True):
                render_kpi_panel(run_df, step_idx, view.kpi_columns)

    with main_cols[1]:
        render_side_panel(current, view, inference)

    render_about(bundle, inference, view)

    if st.session_state.get("autoplay", False):
        if st.session_state.replay_step < len(run_df) - 1:
            time.sleep(REFRESH_SECONDS)
            st.session_state.replay_step += 1
            st.rerun()
        else:
            st.session_state.autoplay = False


if __name__ == "__main__":
    main()
