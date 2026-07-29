"""FastAPI-backed Streamlit defect-inspection dashboard with embedded PyVista views."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000").rstrip("/")
VIEWER_URL = os.getenv("PYVISTA_VIEWER_URL", "http://127.0.0.1:8081").rstrip("/")
VIEWER_TIMEOUT = float(os.getenv("PYVISTA_VIEWER_TIMEOUT", "20"))
TIMEOUT = float(os.getenv("DASHBOARD_API_TIMEOUT", "5"))
SUMMARY_TIMEOUT = float(os.getenv("DASHBOARD_SUMMARY_TIMEOUT", "30"))

SUMMARY = {
    "primary_defect": ["primary_defect", "primary defect", "defect", "defect_type"],
    "stage2_classification": ["stage2_classification", "stage2 classification", "classification"],
    "needs_review": ["needs_review", "needs review", "review_required"],
    "diameter": ["diameter", "equivalent_diameter", "effective_diameter_um", "median_effective_diameter_um"],
    "nominal": ["nominal_diameter", "nominal_diameter_um", "design_diameter", "cad_diameter"],
    "ratio": ["diameter_ratio", "diameter_to_nominal", "diameter_to_nominal_ratio"],
    "deviation": ["max_centerline_deviation", "max_centerline_deviation_um", "maximum_centerline_offset_um"],
    "occupancy": ["occupancy", "material_occupancy", "ct_material_occupancy"],
}
STATION = {
    "position": ["position_fraction", "station_fraction", "position", "fraction"],
    "occupancy": ["material_occupancy", "occupancy", "material_fraction"],
    "diameter": ["equivalent_diameter", "diameter", "effective_diameter_um", "equivalent_radius", "radius"],
    "offset": ["centroid_offset", "centroid_offset_um", "centerline_offset_um", "centerline_deviation"],
}

st.set_page_config(page_title="Lattice NDE Dashboard", layout="wide")


def api_get(path: str, *, request_timeout: float = TIMEOUT, **params: Any) -> dict:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def load_summary() -> pd.DataFrame:
    frame = pd.DataFrame(api_get("/dashboard/struts", request_timeout=SUMMARY_TIMEOUT).get("items", []))
    if frame.empty or "strut_id" not in frame:
        raise ValueError("FastAPI returned no strut summary records")
    frame["strut_id"] = pd.to_numeric(frame["strut_id"], errors="raise").astype(int)
    return frame


def load_stations(strut_id: int) -> pd.DataFrame:
    return pd.DataFrame(api_get(f"/struts/{strut_id}/stations").get("items", []))


def key(value: object) -> str:
    return "".join(char for char in str(value).lower() if char.isalnum())


def column(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    columns = {key(name): name for name in frame.columns}
    return next((columns[key(alias)] for alias in aliases if key(alias) in columns), None)


def severity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    scores = [name for name in ("missing_score", "broken_score", "thin_score", "inflated_score", "bend_score") if name in result]
    if not scores:
        result["severity"] = "Unscored"
        return result
    maximum = result[scores].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0)
    nonzero = maximum[maximum > 0]
    low, medium = nonzero.quantile([0.33, 0.66]).tolist() if not nonzero.empty else (1, 2)
    result["severity"] = np.select([maximum <= low, maximum <= medium], ["Low", "Medium"], default="High")
    return result


def truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "required", "needs review", "needs_review"}


def display(value: object, digits: int = 2, unit: str = "") -> str:
    if value is None or pd.isna(value) or value == "":
        return "N/A"
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{float(number):,.{digits}f} {unit}".strip() if pd.notna(number) else str(value)


def metric(label: str, value: str) -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def publish_selection(strut_id: int) -> None:
    if st.session_state.get("published_strut_id") != strut_id:
        api_post("/select_struts", {"strut_ids": [strut_id]})
        st.session_state.published_strut_id = strut_id


@st.cache_data(ttl=5, show_spinner=False)
def viewer_status() -> tuple[bool, str]:
    """Return cached viewer readiness so reruns do not hammer a starting server."""
    try:
        response = requests.get(VIEWER_URL, timeout=VIEWER_TIMEOUT)
        response.raise_for_status()
        return True, "connected"
    except requests.Timeout:
        return False, f"The viewer did not respond within {VIEWER_TIMEOUT:g} seconds. It may still be building the CT scene."
    except requests.RequestException as exc:
        return False, f"The viewer is not reachable yet: {exc}"


def broker_selection() -> None:
    try:
        active = api_get("/state").get("active_strut_id")
    except requests.RequestException:
        return
    if active is not None and int(active) != st.session_state.get("selected_strut_id"):
        st.session_state.selected_strut_id = int(active)
        st.session_state.published_strut_id = int(active)
        st.rerun()


if hasattr(st, "fragment"):
    @st.fragment(run_every="1s")
    def sync_selection() -> None:
        broker_selection()
else:
    def sync_selection() -> None:
        st.caption("Upgrade Streamlit to enable automatic selection synchronization.")


def overview(summary: pd.DataFrame) -> None:
    st.title("Overview")
    st.caption("High-level view of the defect-inspection workspace.")
    counts = st.columns(3)
    counts[0].metric("Struts", f"{len(summary):,}")
    counts[1].metric("Need review", f"{sum(summary.needs_review.map(truthy)):,}")
    counts[2].metric("Defect classes", f"{summary.primary_defect.nunique():,}")
    st.info("Open Strut Analysis to filter the broker dataset, inspect station evidence, and use the linked 3-D views.")


def workflow() -> None:
    st.title("Inspection Workflow")
    steps = [
        ("CAD + CT scan", "CAD supplies nominal geometry; CT supplies the measured lattice volume."),
        ("Registration", "Aligns CAD and CT coordinates."),
        ("Defect analysis", "Computes strut and station-level morphology metrics."),
        ("Dashboard review", "Synchronizes human review, chat, and the 3-D PyVista views."),
    ]
    for number, (title, description) in enumerate(steps, 1):
        st.markdown(f"<div class='flow-stage'><div class='flow-number'>{number:02d}</div><div class='flow-title'>{title}</div><div>{description}</div></div>", unsafe_allow_html=True)


def trend(stations: pd.DataFrame, title: str, aliases: list[str], position: pd.Series, *, minimum: bool = False) -> None:
    source = column(stations, aliases)
    if source is None:
        return
    values = pd.to_numeric(stations[source], errors="coerce")
    data = pd.DataFrame({"position": position, "value": values}).dropna()
    if data.empty:
        return
    point = data.loc[data.value.idxmin() if minimum else data.value.idxmax()]
    figure = go.Figure(go.Scatter(x=data.position, y=data.value, mode="lines+markers", line={"color": "#38bdf8"}))
    figure.add_trace(go.Scatter(x=[point.position], y=[point.value], mode="markers", marker={"color": "#f97316", "size": 10}))
    figure.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#e5e7eb")
    st.plotly_chart(figure, use_container_width=True)


def analysis(summary: pd.DataFrame) -> None:
    st.title("Strut Analysis")
    st.caption("This Streamlit page is the complete dashboard: numerical analysis, chat, and embedded PyVista macro/micro views stay synchronized.")
    with st.sidebar:
        st.header("Inspection")
        search = st.text_input("Search strut ID", placeholder="e.g. 1234")
        selected_primary = st.multiselect("Primary defect", sorted(summary.primary_defect.unique()), default=sorted(summary.primary_defect.unique()))
        selected_stage2 = st.multiselect("Stage 2 classification", sorted(summary.stage2_classification.unique()), default=sorted(summary.stage2_classification.unique()))
        selected_severity = st.multiselect("Severity", sorted(summary.severity.unique()), default=sorted(summary.severity.unique()))

    filtered = summary[summary.primary_defect.isin(selected_primary) & summary.stage2_classification.isin(selected_stage2) & summary.severity.isin(selected_severity)].copy()
    if search.strip():
        filtered = filtered[filtered.strut_id.astype(str).str.contains(search.strip(), regex=False)]
    filtered = filtered.sort_values("strut_id").reset_index(drop=True)
    if filtered.empty:
        st.warning("No struts match the current filters.")
        return

    event = st.dataframe(filtered, hide_index=True, use_container_width=True, height=280, on_select="rerun", selection_mode="single-row", key="defect_queue")
    rows = getattr(getattr(event, "selection", None), "rows", [])
    available = filtered.strut_id.astype(int).tolist()
    selected = int(st.session_state.get("selected_strut_id", available[0]))
    if rows:
        selected = int(filtered.iloc[rows[0]].strut_id)
    if selected not in available:
        selected = available[0]
    st.session_state.selected_strut_id = selected
    try:
        publish_selection(selected)
    except requests.RequestException as exc:
        st.error(f"Could not publish selection to FastAPI: {exc}")

    selected_row = summary[summary.strut_id == selected].iloc[0]
    st.subheader(f"Selected strut {selected}")
    cards = st.columns(5)
    for container, label, aliases in zip(cards, ["Primary defect", "Stage 2", "Needs review", "Occupancy", "Max deviation"], [SUMMARY["primary_defect"], SUMMARY["stage2_classification"], SUMMARY["needs_review"], SUMMARY["occupancy"], SUMMARY["deviation"]]):
        source = column(summary, aliases)
        with container:
            metric(label, display(selected_row[source]) if source else "N/A")

    st.subheader("Linked 3-D inspection")
    st.caption("The PyVista renderer is embedded below; keep this Streamlit page open for the numerical analysis and chat.")
    viewer_ready, viewer_message = viewer_status()
    if viewer_ready:
        components.iframe(VIEWER_URL, height=720, scrolling=False)
    else:
        st.warning(f"The PyVista viewer is not ready at {VIEWER_URL}: {viewer_message} Numeric analysis and chat remain available.")

    try:
        stations = load_stations(selected)
    except requests.RequestException as exc:
        st.error(f"Could not load station records: {exc}")
        return
    position_source = column(stations, STATION["position"])
    if position_source:
        position = pd.to_numeric(stations[position_source], errors="coerce")
        position = position * 100 if position.dropna().max() <= 1 else position
        charts = st.columns(3)
        with charts[0]:
            trend(stations, "Material occupancy", STATION["occupancy"], position, minimum=True)
        with charts[1]:
            trend(stations, "Measured diameter", STATION["diameter"], position, minimum=True)
        with charts[2]:
            trend(stations, "Centerline offset", STATION["offset"], position)

    with st.expander("View technical/raw data"):
        st.json(selected_row.to_dict())
        st.dataframe(stations, hide_index=True, use_container_width=True)
    chat(selected)


def chat(selected_id: int) -> None:
    st.subheader("Chat")
    st.caption(f"Questions use the active broker selection: strut {selected_id}.")
    for message in st.session_state.setdefault("chat_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask about the selected strut, review queue, or defect summary")
    if not question:
        return
    st.session_state.chat_messages.append({"role": "user", "content": question})
    try:
        response = api_post("/chat", {"message": question, "active_strut_id": selected_id})
        st.session_state.chat_messages.append({"role": "assistant", "content": response.get("reply", "FastAPI returned no reply.")})
    except requests.RequestException as exc:
        st.session_state.chat_messages.append({"role": "assistant", "content": f"Chat is unavailable: {exc}"})
    st.rerun()


st.markdown("""
<style>
.stApp{background:#0b1120}.metric-card,.flow-stage{background:#172033;border:1px solid #293750;border-radius:12px;padding:14px 16px;min-height:76px}.metric-label{color:#94a3b8;font-size:.76rem;text-transform:uppercase;letter-spacing:.06em}.metric-value{color:#f8fafc;font-size:1.15rem;font-weight:650;margin-top:7px}.flow-stage{margin:10px 0;color:#cbd5e1}.flow-number{color:#38bdf8;font-size:.72rem}.flow-title{color:#f8fafc;font-weight:700;margin:4px 0}
</style>
""", unsafe_allow_html=True)

sync_selection()
try:
    summary = severity(load_summary())
except (requests.RequestException, ValueError) as exc:
    st.error(f"Could not load dashboard data from FastAPI at {API_URL}: {exc}")
    st.info("Start FastAPI with: uvicorn part2.stage_4_interactive_dashboard_napari_chatbot.run_fastapi:app --host 127.0.0.1 --port 8000")
    st.stop()
for name, default in (("primary_defect", "Unknown"), ("stage2_classification", "Unknown"), ("needs_review", "Not provided")):
    if name not in summary:
        summary[name] = default
    summary[name] = summary[name].fillna(default).astype(str)

with st.sidebar:
    st.caption("Navigation")
    page = st.radio("Page", ["Overview", "Inspection Workflow", "Strut Analysis"], label_visibility="collapsed")
if page == "Overview":
    overview(summary)
elif page == "Inspection Workflow":
    workflow()
else:
    analysis(summary)
