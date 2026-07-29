"""FastAPI-backed Streamlit defect-inspection dashboard with embedded PyVista views."""
from __future__ import annotations

import os
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components

API_URL = os.getenv("DASHBOARD_API_URL", "http://127.0.0.1:8000").rstrip("/")
VIEWER_URL = os.getenv("PYVISTA_VIEWER_URL", "http://127.0.0.1:8081").rstrip("/")
TIMEOUT = float(os.getenv("DASHBOARD_API_TIMEOUT", "30"))
SUMMARY_TIMEOUT = float(os.getenv("DASHBOARD_SUMMARY_TIMEOUT", "30"))
VIEWER_TIMEOUT = float(os.getenv("PYVISTA_VIEWER_TIMEOUT", "20"))

st.set_page_config(page_title="Lattice NDE Dashboard", layout="wide")

SUMMARY = {
    "primary_defect": ["primary_defect", "primary defect", "defect", "defect_type", "defect type"],
    "stage2_classification": ["stage2_classification", "stage2 classification", "stage2a_classification", "classification"],
    "needs_review": ["needs_review", "needs review", "review_required", "review required"],
    "severity": ["severity", "severity_level", "severity level"],
    "diameter": ["diameter", "equivalent_diameter", "effective_diameter_um", "median_effective_diameter_um", "diameter_um"],
    "nominal": ["nominal_diameter", "nominal diameter", "nominal_diameter_um", "design_diameter", "cad_diameter"],
    "ratio": ["diameter_ratio", "diameter_to_nominal", "diameter_to_nominal_ratio", "diameter / nominal"],
    "deviation": ["max_centerline_deviation", "max_centerline_deviation_um", "centerline_deviation", "maximum_centerline_offset_um"],
    "occupancy": ["occupancy", "material_occupancy", "material fraction", "ct_material_occupancy", "station_material_fraction"],
    "curvature": ["curvature", "tortuosity", "bend_curvature"],
}
STATION = {
    "position": ["position_fraction", "station_fraction", "position", "fraction"],
    "occupancy": ["material_occupancy", "occupancy", "material_fraction", "station_material_fraction"],
    "diameter": ["equivalent_diameter", "diameter", "effective_diameter_um", "diameter_um", "equivalent_radius", "radius", "radius_um"],
    "offset": ["centroid_offset", "centroid_offset_um", "centerline_offset_um", "centerline_deviation", "centerline_deviation_um"],
}

def key(x): return "".join(c for c in str(x).lower() if c.isalnum())
def col(df, aliases):
    lookup = {key(c): c for c in df.columns}
    return next((lookup[key(a)] for a in aliases if key(a) in lookup), None)
def canon(df, aliases):
    df = df.copy()
    for name, options in aliases.items():
        c = col(df, options)
        if c and name not in df: df[name] = df[c]
    return df
def truthy(x): return str(x).strip().lower() in {"true", "1", "yes", "required", "needs review", "needs_review"}
def display(x, unit="", digits=2):
    if x is None or pd.isna(x) or x == "": return "N/A"
    n = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return f"{float(n):,.{digits}f} {unit}".strip() if pd.notna(n) else str(x)
def chart(fig, height=300):
    fig.update_layout(height=height, margin=dict(l=10,r=10,t=42,b=10), paper_bgcolor="#111827", plot_bgcolor="#111827", font_color="#e5e7eb", xaxis_gridcolor="#293750", yaxis_gridcolor="#293750")
    return fig
def metric(label, value, help_text=""):
    title = f' title="{help_text}"' if help_text else ""
    st.markdown(f'<div class="metric-card"{title}><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
def series(df, aliases):    c = col(df, aliases); return c, pd.to_numeric(df[c], errors="coerce") if c else (None, None)


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
    if frame["strut_id"].duplicated().any():
        raise ValueError("FastAPI returned duplicate strut IDs")
    return canon(frame, SUMMARY)


@st.cache_data(ttl=300, show_spinner=False)
def load_stations(strut_id: int) -> pd.DataFrame:
    frame = pd.DataFrame(api_get(f"/struts/{strut_id}/stations").get("items", []))
    if "strut_id" not in frame and not frame.empty:
        frame["strut_id"] = strut_id
    return canon(frame, STATION)


def load_reviews() -> dict[int, dict[str, Any]]:
    return {
        int(item["strut_id"]): item
        for item in api_get("/reviews").get("items", [])
        if item.get("strut_id") is not None
    }


def save_review(strut_id: int, decision: str) -> None:
    api_post(f"/reviews/{strut_id}", {"decision": decision})


def material_status(row: pd.Series) -> tuple[str, str, str]:
    classification = str(row.get("stage2_classification", "Unknown"))
    if classification == "Nominal":
        return "Nominally present", "Material is expected at this CAD strut.", "present"
    if classification == "Missing_Intentional":
        return "Nominally missing", "CAD intentionally expects no material at this strut.", "missing"
    if classification == "Missing_Unintentional":
        return "Unexpectedly missing", "CAD expects material, but this strut is missing.", "missing"
    if classification == "Expected_Missing_But_Material_Present":
        return "Unexpected material", "Material is present where CAD expects the strut to be absent.", "warning"
    return classification.replace("_", " "), "No material-status interpretation is available.", "neutral"


def primary_label(row: pd.Series) -> str:
    value = str(row.get("primary_defect", "Nominal"))
    return "No detected geometry defect" if value in {"", "Nominal", "nan"} else value


def status_card(label: str, value: str, detail: str, tone: str = "neutral") -> None:
    st.markdown(
        f'<div class="status-card status-{tone}"><div class="status-label">{escape(label)}</div>'
        f'<div class="status-value">{escape(value)}</div><div class="status-detail">{escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def render_statistics_banner(summary: pd.DataFrame, reviews: dict[int, dict[str, Any]]) -> None:
    frame = summary.copy()
    frame["__material_status"] = frame.apply(lambda row: material_status(row)[0], axis=1)
    frame["__primary_label"] = frame.apply(primary_label, axis=1)
    missing = frame["__material_status"] == "Nominally missing"
    frame.loc[missing, "__primary_label"] = frame.loc[missing, "stage2_classification"].astype(str).str.replace("_", " ")

    parent_counts = frame["__material_status"].value_counts()
    pie = go.Figure(go.Pie(
        labels=parent_counts.index, values=parent_counts.values, hole=0.45,
        hovertemplate="%{label}<br>%{value:,} struts<br>%{percent:.2%} of all struts<extra></extra>",
    ))
    pie.update_layout(margin=dict(l=0, r=0, t=35, b=0), title="Material status")

    defect_frame = frame[frame["__primary_label"] != "No detected geometry defect"]
    defect_counts = defect_frame["__primary_label"].value_counts().sort_values()
    defect_percent = defect_counts / defect_counts.sum() * 100
    defect_bars = go.Figure(go.Bar(
        x=defect_counts.values,
        y=defect_counts.index,
        orientation="h",
        marker_color="#f97316",
        text=[f"{count:,} ({percent:.1f}%)" for count, percent in zip(defect_counts.values, defect_percent.values)],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:,} struts<br>%{customdata:.2f}% of defect-bearing struts<extra></extra>",
        customdata=defect_percent.values,
    ))
    defect_bars.update_layout(
        height=300, margin=dict(l=10, r=65, t=42, b=10), title="Defect-bearing struts",
        xaxis_title="Strut count", yaxis_title=None,
    )

    automated_review = frame["needs_review"].map(truthy)
    unresolved = int(sum(automated_review & ~frame["strut_id"].astype(int).isin(reviews)))
    final_counts = pd.Series([item["decision"] for item in reviews.values()]).value_counts().to_dict()
    pie_col, defects_col, queue_col = st.columns([1.2, 2.1, 0.7])
    with pie_col: st.plotly_chart(chart(pie, 300), width='stretch')
    with defects_col: st.plotly_chart(chart(defect_bars, 300), width='stretch')
    with queue_col:
        st.markdown("### Review queue")
        metric("Unresolved", f"{unresolved:,}", "Automated review flags without a final saved decision.")
        st.caption(f"Automated flags: {int(automated_review.sum()):,}")
        st.caption(f"Final nominal: {final_counts.get('nominal', 0):,}")
        st.caption(f"Final needs review: {final_counts.get('needs_review', 0):,}")
        st.caption(f"Confirmed defect: {final_counts.get('confirmed_defect', 0):,}")


def render_radius_deviation_analytics(summary: pd.DataFrame) -> None:
    frame = summary.copy()
    frame["__primary_label"] = frame.apply(primary_label, axis=1)
    missing = frame["stage2_classification"].astype(str) != "Nominal"
    frame.loc[missing, "__primary_label"] = frame.loc[missing, "stage2_classification"].astype(str).str.replace("_", " ")
    mean_source = frame.groupby("__primary_label", sort=True).agg(
        radius_um=("median_cross_section_radius_um", "mean"),
        deviation_um=("max_centerline_offset_um", "mean"),
    ).reset_index()
    means = make_subplots(rows=1, cols=2, subplot_titles=("Mean radius", "Mean max deviation"))
    means.add_bar(x=mean_source["__primary_label"], y=mean_source["radius_um"], marker_color="#38bdf8", row=1, col=1)
    means.add_bar(x=mean_source["__primary_label"], y=mean_source["deviation_um"], marker_color="#f97316", row=1, col=2)
    means.update_yaxes(title_text="µm", row=1, col=1)
    means.update_yaxes(title_text="µm", row=1, col=2)
    means.update_xaxes(tickangle=-40)
    means.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=48, b=70), title="Primary-defect class means")

    scatter = go.Figure()
    for label, group in frame.groupby("__primary_label", sort=True):
        scatter.add_trace(go.Scattergl(
            x=group["median_cross_section_radius_um"], y=group["max_centerline_offset_um"], mode="markers", name=label,
            customdata=np.column_stack((group["strut_id"], group["stage2_classification"])),
            hovertemplate="Strut %{customdata[0]}<br>%{customdata[1]}<br>Radius %{x:.1f} µm<br>Max deviation %{y:.1f} µm<extra></extra>",
            marker={"size": 5, "opacity": 0.6},
        ))
    scatter.update_layout(height=320, margin=dict(l=10, r=10, t=42, b=10), title="All-strut radius vs deviation", xaxis_title="Median radius (µm)", yaxis_title="Max deviation (µm)")
    means_col, scatter_col = st.columns(2)
    with means_col: st.plotly_chart(chart(means, 320), width='stretch')
    with scatter_col: st.plotly_chart(chart(scatter, 320), width='stretch')


def publish_selection(strut_id: int) -> None:
    if st.session_state.get("published_strut_id") == strut_id:
        return
    api_post(
        "/select_struts",
        {"strut_ids": [strut_id], "active_strut_id": strut_id},
    )
    st.session_state.published_strut_id = strut_id


@st.cache_data(ttl=5, show_spinner=False)
def viewer_status() -> tuple[bool, str]:
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
        return None


def severity(df):
    if "severity" in df: return df
    scores = [c for c in ["missing_score","broken_score","thin_score","inflated_score","bend_score"] if c in df]
    if not scores: return df
    values = df[scores].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0); nz = values[values > 0]
    q1, q2 = nz.quantile([.33,.66]).tolist() if len(nz) else (1,2)
    result = df.copy(); result["severity"] = np.select([values <= q1, values <= q2], ["Low","Medium"], default="High"); return result

def overview(summary):
    st.title("Overview"); st.caption("High-level view of the defect-inspection workspace.")
    with st.container(border=True):
        st.subheader("Inspection workspace")
        st.write("FastAPI provides the analysis summary and per-strut station evidence; PyVista provides the linked CT visualization.")
        st.success(f"Analysis data ready — {len(summary):,} struts available from FastAPI.")
        st.write("Inspection Workflow documents the pipeline from source data through dashboard review.")
def flowchart():
    st.title("Inspection Workflow"); st.caption("The inspection pipeline from source data to human review.")
    steps = [("CAD + CT scan","CAD provides nominal geometry; CT provides the measured lattice volume."),("Registration","Aligns CAD and CT coordinates."),("Missing-strut analysis","Finds absent or unexpected material relative to design."),("Station-level morphology analysis","Measures occupancy, diameter/radius, and centerline offset along each strut."),("Defect classification","Combines measurements into defect labels and stage 2 classifications."),("Dashboard review","Supports filtering, strut inspection, local decisions, and contextual questions.")]
    for i,(title,desc) in enumerate(steps):
        a, arrow, b = st.columns([1,.12,2.5]); a.markdown(f"<div class='flow-stage'><div class='flow-number'>{i+1:02d}</div><div class='flow-title'>{title}</div></div>", unsafe_allow_html=True)
        if i < len(steps)-1: arrow.markdown("<div class='flow-arrow'></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='flow-description'>{desc}</div>", unsafe_allow_html=True)

def analysis(summary):
    st.title("Strut Analysis"); st.caption("Review FastAPI-backed analysis and inspect station-level evidence with the linked PyVista viewer.")
    summary = severity(summary.copy())
    for name, default in [("primary_defect", "Unknown"), ("stage2_classification", "Unknown"), ("needs_review", "Not provided"), ("severity", "Unscored")]:
        if name not in summary: summary[name] = default
    try:
        reviews = load_reviews()
    except requests.RequestException as exc:
        reviews = {}
        st.warning(f"Could not load saved review decisions: {exc}")
    render_statistics_banner(summary, reviews)
    filter_keys = ["filter_search", "filter_primary", "filter_stage2", "filter_review", "filter_severity"]
    source_columns = {
        "occupancy": col(summary, SUMMARY["occupancy"]),
        "deviation": col(summary, SUMMARY["deviation"]),
        "ratio": col(summary, SUMMARY["ratio"] + ["diameter_ratio_to_nominal"]),
        "diameter": col(summary, SUMMARY["diameter"]),
        "bent": col(summary, ["curvature", "tortuosity", "tortuosity_ratio", "bend_curvature"]),
        "review": col(summary, SUMMARY["needs_review"]),
    }
    ratio_or_diameter = source_columns["ratio"] or source_columns["diameter"]
    quick_requirements = {
        "Lowest occupancy": source_columns["occupancy"],
        "Largest deviation": source_columns["deviation"],
        "Thinnest": ratio_or_diameter,
        "Thickest/inflated": ratio_or_diameter,
        "Most bent": source_columns["bent"],
        "Needs review": source_columns["review"],
    }
    quick_actions = [("Lowest occupancy", "lowest_occupancy"), ("Largest deviation", "largest_deviation"), ("Thinnest", "thinnest"), ("Thickest/inflated", "thickest_inflated"), ("Most bent", "most_bent"), ("Needs review", "needs_review")]

    def quick_candidates(choice):
        frame = summary.copy()
        if choice == "Needs review":
            frame = frame[frame.needs_review.map(truthy)]
            return frame.sort_values("strut_id"), int(frame.iloc[0].strut_id) if len(frame) else None
        aliases = {
            "Lowest occupancy": SUMMARY["occupancy"],
            "Largest deviation": SUMMARY["deviation"],
            "Thinnest": SUMMARY["ratio"] + ["diameter_ratio_to_nominal"] if source_columns["ratio"] else SUMMARY["diameter"],
            "Thickest/inflated": SUMMARY["ratio"] + ["diameter_ratio_to_nominal"] if source_columns["ratio"] else SUMMARY["diameter"],
            "Most bent": ["curvature", "tortuosity", "tortuosity_ratio", "bend_curvature"],
        }
        source = col(frame, aliases[choice])
        if source is None: return frame.sort_values("strut_id"), None
        ranked = frame.copy(); ranked["__quick_metric"] = pd.to_numeric(ranked[source], errors="coerce")
        ascending = choice in {"Lowest occupancy", "Thinnest"}
        ranked = ranked.dropna(subset=["__quick_metric"]).sort_values(["__quick_metric", "strut_id"], ascending=[ascending, True]).drop(columns="__quick_metric")
        return ranked, int(ranked.iloc[0].strut_id) if len(ranked) else None

    def activate_quick_pick(choice):
        candidates, picked = quick_candidates(choice)
        if picked is None: return
        for filter_key in filter_keys: st.session_state.pop(filter_key, None)
        st.session_state.quick_pick_state = choice
        st.session_state.show_matching_struts = True
        st.session_state.selected_strut_id = picked
        st.session_state.selected_strut_picker = picked

    def reset_filters():
        for filter_key in filter_keys: st.session_state.pop(filter_key, None)
        st.session_state.pop("quick_pick_state", None)
        st.session_state.pop("selected_strut_id", None)
        st.session_state.pop("selected_strut_picker", None)
        st.session_state.show_matching_struts = False

    with st.container(border=True):
        st.subheader("Inspection Start")
        search = st.text_input("Strut ID", placeholder="Search strut ID, e.g. 1728", key="filter_search")
        primary, stage2, review, sev = [], [], [], []
        st.caption(f"Loaded dataset: {len(summary):,} struts")
        quick_picks = st.columns(6)
        for button, (label, action) in zip(quick_picks, quick_actions):
            available = quick_requirements[label] is not None
            help_text = None if available else f"Unavailable: required metric for {label.lower()} is not present in the FastAPI summary."
            button.button(label, key=f"quick_pick_{action}", disabled=not available, help=help_text, width='stretch', on_click=activate_quick_pick, args=(label,))
        st.button("Reset filters", key="reset_filters", use_container_width=False, on_click=reset_filters)

    quick_state = st.session_state.get("quick_pick_state")
    quick_active = bool(quick_state and not search.strip() and not primary and not stage2 and not review and not sev)
    if quick_active:
        filtered, quick_selected = quick_candidates(quick_state)
        st.info(f"Quick pick active: {quick_state} - showing top candidates.")
    else:
        mask = pd.Series(True, index=summary.index)
        if primary: mask &= summary.primary_defect.astype(str).isin(primary)
        if stage2: mask &= summary.stage2_classification.astype(str).isin(stage2)
        if review: mask &= summary.needs_review.astype(str).isin(review)
        if sev: mask &= summary.severity.astype(str).isin(sev)
        filtered = summary[mask].copy()
        if search.strip(): filtered = filtered[filtered.strut_id.astype(str).str.contains(search.strip(), regex=False)]
        filtered = filtered.sort_values("strut_id")
    ids = filtered.strut_id.astype(int).tolist()
    st.caption(f"Showing {len(ids):,} of {len(summary):,} struts.")
    browse = st.button("Browse matching struts", key="browse_matching_struts", use_container_width=False)
    if browse: st.session_state.show_matching_struts = not st.session_state.get("show_matching_struts", False)
    table_event = None
    if st.session_state.get("show_matching_struts", False):
        useful = [x for x in ["strut_id", "primary_defect", "stage2_classification", "severity", "needs_review"] if x in filtered]
        table_event = st.dataframe(filtered[useful], hide_index=True, height=420, width='stretch', on_select="rerun", selection_mode="single-row", key="defect_queue")
    if not ids:
        st.warning("No struts match the current filters."); return

    previous = st.session_state.get("selected_strut_id")
    current = int(previous) if previous in ids else ids[0]
    rows = getattr(getattr(table_event, "selection", None), "rows", [])
    if rows:
        current = int(filtered.iloc[rows[0]].strut_id)
        st.session_state.selected_strut_id = current
    else:
        st.session_state.selected_strut_id = current
    try:
        publish_selection(current)
    except requests.RequestException as exc:
        st.warning(f"Could not publish the selected strut to FastAPI: {exc}")
    st.caption(f"Selected strut: {current}. Browse the full result set above and select a row to inspect it.")
    selected = summary[summary.strut_id == int(current)].iloc[0]
    try:
        stations = load_stations(current)
    except requests.RequestException as exc:
        st.error(f"Could not load station data for strut {current} from FastAPI: {exc}")
        return
    except ValueError as exc:
        st.error(f"Invalid station data for strut {current}: {exc}")
        return
    _, position = series(stations, STATION["position"])
    if position is not None:
        position = pd.to_numeric(position, errors="coerce")
        position = position * 100 if position.dropna().max() <= 1.0 else position
        stations["__position_pct"] = position
        stations = stations.dropna(subset=["__position_pct"]).sort_values("__position_pct")
        position = stations["__position_pct"]

    def value(aliases):
        source = col(summary, aliases); return selected[source] if source else None
    def metric_number(aliases, source_frame=summary, source_row=selected):
        source = col(source_frame, aliases)
        if not source: return None, None
        return pd.to_numeric(pd.Series([source_row[source]]), errors="coerce").iloc[0], source
    def scale_to_um(number, source_name):
        if number is None or pd.isna(number): return number
        name = str(source_name).lower()
        if "_mm" in name or "(mm" in name: return float(number) * 1000
        if "_m" in name or "(m" in name or name.endswith("meter"): return float(number) * 1000000
        return float(number)
    def fmt(number, digits=2, unit=""):
        if number is None or pd.isna(number): return "N/A"
        return f"{float(number):,.{digits}f} {unit}".strip()
    def pct(number, digits=1):
        if number is None or pd.isna(number): return "N/A"
        number = float(number) * 100 if abs(float(number)) <= 1 else float(number)
        return f"{number:.{digits}f}%"

    st.divider(); st.subheader(f"Selected strut {current}")
    radius, radius_source = metric_number(["median_cross_section_radius_um", "median_radius_um"])
    radius = scale_to_um(radius, radius_source)
    length, length_source = metric_number(["length_um", "inventory_length_um", "length_um_from_centerline"])
    length = scale_to_um(length, length_source)
    status, status_detail, tone = material_status(selected)
    secondary = str(selected.get("secondary_defects", "")).replace(";", ", ") or "None reported"
    current_review = reviews.get(int(current), {}).get("decision", "No final decision")
    cards = st.columns(5)
    with cards[0]: status_card("Material status", status, status_detail, tone)
    with cards[1]: status_card("Primary defect", primary_label(selected), "Geometry screening classification.", "warning" if primary_label(selected) != "No detected geometry defect" else "present")
    with cards[2]: status_card("Secondary defects", secondary, "Additional concurrent defect signals.")
    with cards[3]: status_card("Length", fmt(length, 1, "µm"), "Registered centerline length.")
    with cards[4]: status_card("Median diameter", fmt(None if radius is None else radius * 2, 1, "µm"), "Twice the median cross-sectional radius.")

    st.subheader("Linked 3-D inspection")
    viewer_ready, viewer_message = viewer_status()
    if viewer_ready:
        components.iframe(VIEWER_URL, height=720, scrolling=False)
    else:
        st.warning(f"The PyVista viewer is not ready at {VIEWER_URL}: {viewer_message} Numerical analysis remains available.")

    st.subheader("Final review")
    buttons = st.columns(3)
    for button, label, decision in zip(buttons, ["Confirm defect", "Mark needs review", "Mark nominal"], ["confirmed_defect", "needs_review", "nominal"]):
        if button.button(label, key=f"review_{decision}_{current}", width='stretch'):
            try:
                save_review(current, decision)
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Could not save final review decision: {exc}")
    st.caption(f"Saved final decision: {str(current_review).replace('_', ' ')}")

    def station_values(aliases, scale=False):
        source, values = series(stations, aliases)
        if source is None or values is None: return None, None
        values = pd.to_numeric(values, errors="coerce")
        if scale: values = values.map(lambda number: scale_to_um(number, source))
        return source, values
    def make_trend(title, y_title, aliases, scale=False, percent=False, minimum=False, maximum=False, reference=None):
        source, values = station_values(aliases, scale)
        if source is None or position is None or values.dropna().empty: return False
        plot_values = values * 100 if percent and values.dropna().max() <= 1 else values
        valid = pd.DataFrame({"position": position, "value": plot_values}).dropna()
        if valid.empty: return False
        fig = go.Figure(go.Scatter(x=valid.position, y=valid.value, mode="lines+markers", line={"color":"#38bdf8"}, name=y_title))
        if reference is not None and not pd.isna(reference): fig.add_hline(y=reference, line_dash="dash", line_color="#f59e0b", annotation_text="Nominal")
        point = valid.loc[valid.value.idxmin() if minimum else valid.value.idxmax()]
        label = "Lowest" if minimum else "Maximum"
        fig.add_trace(go.Scatter(x=[point.position], y=[point.value], mode="markers+text", text=[label], textposition="top center", marker={"color":"#f97316", "size":10}, name=label))
        fig.update_layout(title=title, xaxis_title="Position along strut (%)", yaxis_title=y_title)
        st.plotly_chart(chart(fig, 300), width='stretch')
        return float(point.value), float(point.position), label

    with st.expander("Additional trends & data", expanded=False):
        render_radius_deviation_analytics(summary)
        trend_specs = [
            ("Material occupancy", "Material occupancy (%)", STATION["occupancy"], False, True, True, False, None),
            ("Equivalent radius", "Equivalent radius (µm)", STATION["diameter"], True, False, True, False, None),
            ("Centerline offset", "Centerline offset (µm)", STATION["offset"], True, False, False, True, None),
            ("Mean CT intensity", "Mean CT intensity (a.u.)", ["mean_intensity", "intensity", "ct_intensity"], False, False, False, True, None),
            ("Outside-nominal material", "Outside-nominal material (%)", ["outside_nominal_material", "outside_nominal_fraction", "outside_nominal", "excess_material_fraction"], False, True, True, False, None),
        ]
        rendered = [spec for spec in trend_specs if (values := station_values(spec[2], spec[3])[1]) is not None and position is not None and not values.dropna().empty]
        for panel, spec in zip(st.columns(max(1, len(rendered))), rendered):
            with panel: make_trend(*spec)
        st.markdown("**Technical data**")
        st.json(selected.to_dict())
        st.dataframe(stations.drop(columns=["__position_pct"], errors="ignore"), hide_index=True, width='stretch')
    st.markdown("**Chat context**"); st.caption(f"Questions use FastAPI-backed values for selected strut {current}.")
    use_gemini = st.toggle("Use Gemini for open-ended questions", value=False, key="use_gemini_chat")
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]): st.markdown(message["content"])
    question = st.chat_input(f"Ask about strut {current} or the defect summary")
    if question:
        messages = st.session_state.setdefault("chat_messages", [])
        messages.append({"role": "user", "content": question})
        try:
            response = api_post(
                "/chat",
                {"message": question, "active_strut_id": current, "chat_history": messages[-4:], "use_gemini": use_gemini},
            )
            messages.append({"role": "assistant", "content": response.get("reply", "Chat returned no response.")})
            selection = response.get("selection") or {}
            active_id = selection.get("active_strut_id")
            if active_id is not None:
                # Adopt the selection published by chat before this rerun.
                # `publish_selection` below then sees this ID as already
                # published instead of replacing the multi-selection with the
                # dashboard's former single selection.
                st.session_state.selected_strut_id = int(active_id)
                st.session_state.published_strut_id = int(active_id)
        except requests.RequestException as exc:
            messages.append({"role": "assistant", "content": f"Chat is unavailable: {exc}"})
        st.rerun()

st.markdown("""
<style>
.stApp{background:#0b1120}[data-testid="stSidebar"]{background:#111827;border-right:1px solid #293750}.metric-card,.status-card{background:#172033;border:1px solid #293750;border-radius:12px;padding:14px 16px;min-height:76px}.metric-label,.status-label{color:#94a3b8;font-size:.76rem;text-transform:uppercase;letter-spacing:.06em}.metric-value,.status-value{color:#f8fafc;font-size:1.15rem;font-weight:650;margin-top:7px}.status-detail{color:#cbd5e1;font-size:.78rem;margin-top:6px}.status-present{border-left:4px solid #22c55e}.status-missing{border-left:4px solid #ef4444}.status-warning{border-left:4px solid #f59e0b}.flow-stage{background:#172033;border:1px solid #334155;border-radius:12px;padding:18px;text-align:center}.flow-number{color:#38bdf8;font-size:.72rem}.flow-title{color:#f8fafc;font-weight:700;margin-top:8px}.flow-description{color:#cbd5e1;padding:18px 4px}.flow-arrow{color:#38bdf8;font-size:2rem;text-align:center;padding-top:18px}
[data-testid="stSidebar"] [data-testid="stButton"] button{justify-content:flex-start;border-radius:7px;font-weight:600;min-height:42px;margin:2px 0}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]{background:transparent;border-color:transparent;color:#e5e7eb}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover{background:#1f2937;border-color:transparent;color:#fff}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]{background:#1f2937;border-color:#1f2937;border-left:3px solid #38bdf8;color:#f8fafc}
</style>
""", unsafe_allow_html=True)
sync_selection()
try:
    summary = load_summary()
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
    st.session_state.setdefault("page", "Overview")
    nav_items = [("Overview", ":material/dashboard:"), ("Inspection Workflow", ":material/account_tree:"), ("Strut Analysis", ":material/analytics:")]
    for label, icon in nav_items:
        if st.button(label, icon=icon, key=f"nav_{label}", type="primary" if st.session_state.page == label else "secondary", width='stretch'):
            st.session_state.page = label
            st.rerun()
page = st.session_state.page
if page=="Overview": overview(summary)
elif page=="Inspection Workflow": flowchart()
else: analysis(summary)
