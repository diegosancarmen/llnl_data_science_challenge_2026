"""FastAPI-backed Streamlit defect-inspection dashboard with embedded PyVista views."""
from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
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
REPO_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_SUMMARY = REPO_ROOT / "part2" / "stage_3_defect_analysis" / "output" / "station_export_20260727T193004Z" / "defect_analysis_summary.json"
INVENTORY_CANDIDATES = (
    REPO_ROOT / "part2" / "stage_2a_developer_output" / "all_struts_inventory.csv",
    REPO_ROOT / "part2" / "registration" / "alignment_check" / "stage2a_candidate_registration_output" / "all_struts_inventory.csv",
)

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


@st.cache_data(ttl=300, show_spinner=False)
def load_pipeline_counts() -> dict[str, Any]:
    """Load workflow counts from the generated Stage 2a/3 artifacts."""
    counts = {"indexed": 0, "compared": 0, "defects": 0, "scan_shape": "volume dimensions unavailable", "stations": 21, "reference_population": 0}
    try:
        payload = json.loads(PIPELINE_SUMMARY.read_text(encoding="utf-8"))
        counts["compared"] = int(payload.get("processed_struts", 0) or 0)
        counts["indexed"] = counts["compared"]
        counts["defects"] = sum(
            int(value or 0)
            for label, value in payload.get("primary_defect_counts", {}).items()
            if str(label).casefold() != "nominal"
        )
        shape = payload.get("scan_shape")
        if shape:
            counts["scan_shape"] = " × ".join(f"{int(value):,}" for value in shape)
        counts["stations"] = int(payload.get("parameters", {}).get("stations", 21) or 21)
        counts["reference_population"] = int(payload.get("nominal_reference_population", 0) or 0)
    except (OSError, ValueError, TypeError):
        pass
    for candidate in INVENTORY_CANDIDATES:
        if candidate.exists():
            try:
                counts["indexed"] = max(counts["indexed"], int(pd.read_csv(candidate, usecols=[0]).shape[0]))
            except (OSError, ValueError, pd.errors.ParserError):
                pass
            break
    return counts


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
    category = strut_category(row)
    details = {
        "Present by Design — No defects": ("Material is expected at this CAD strut; no primary geometry defect detected.", "present"),
        "Present by Design — Defect present": ("Material is expected at this CAD strut; a primary geometry defect was detected.", "warning"),
        "Missing by Design": ("CAD intentionally expects no material at this strut.", "missing"),
        "Missing by Accident": ("CAD expects material, but this strut is missing.", "missing"),
        "Unexpected material": ("Material is present where CAD expects the strut to be absent.", "warning"),
    }
    detail, tone = details[category]
    return category, detail, tone


DISPLAY_CLASSIFICATION_NAMES = {
    "Missing_Intentional": "Missing by Design",
    "Missing_Unintentional": "Missing by Accident",
}


def display_classification(value: Any) -> str:
    return DISPLAY_CLASSIFICATION_NAMES.get(str(value), str(value).replace("_", " "))


def primary_label(row: pd.Series) -> str:
    value = str(row.get("primary_defect", "Nominal"))
    return "No detected geometry defect" if value in {"", "Nominal", "nan"} else value


NO_DEFECT_LABELS = {
    "", "nominal", "nan", "unknown", "none", "null", "no detected geometry defect",
    "missing_intentional", "missing_unintentional",
}
TOP_LEVEL_CATEGORIES = [
    "Present by Design — No defects",
    "Present by Design — Defect present",
    "Missing by Design",
    "Missing by Accident",
    "Unexpected material",
]


def has_primary_defect(row: pd.Series) -> bool:
    return str(row.get("primary_defect", "Nominal")).strip().casefold() not in NO_DEFECT_LABELS


def strut_category(row: pd.Series) -> str:
    classification = str(row.get("stage2_classification", "Unknown")).strip()
    if classification == "Nominal":
        return "Present by Design — Defect present" if has_primary_defect(row) else "Present by Design — No defects"
    if classification == "Missing_Intentional":
        return "Missing by Design"
    if classification == "Missing_Unintentional":
        return "Missing by Accident"
    return "Unexpected material"


NO_DEFECT_LABELS = {
    "", "nominal", "nan", "unknown", "none", "null", "no detected geometry defect",
    "missing_intentional", "missing_unintentional",
}
TOP_LEVEL_CATEGORIES = [
    "Present by Design — No defects",
    "Present by Design — Defect present",
    "Missing by Design",
    "Missing by Accident",
    "Unexpected material",
]


def has_primary_defect(row: pd.Series) -> bool:
    return str(row.get("primary_defect", "Nominal")).strip().casefold() not in NO_DEFECT_LABELS


def strut_category(row: pd.Series) -> str:
    classification = str(row.get("stage2_classification", "Unknown")).strip()
    if classification == "Nominal":
        return "Present by Design — Defect present" if has_primary_defect(row) else "Present by Design — No defects"
    if classification == "Missing_Intentional":
        return "Missing by Design"
    if classification == "Missing_Unintentional":
        return "Missing by Accident"
    return "Unexpected material"


def geometry_classification(row: pd.Series) -> tuple[str, str, str]:
    """Return a plain-language geometry badge, explanation, and visual tone."""
    raw = row.get("primary_defect", "Nominal")
    value = "" if raw is None or pd.isna(raw) else str(raw).strip()
    normalized = value.casefold().replace("_", " ")
    if not normalized or normalized in {"nan", "null", "nominal", "none"}:
        return "NOMINAL", "No defect detected. The measured strut matches the expected geometry.", "present"
    if "missing" in normalized:
        return "MISSING", "The design expects material here, but the measured strut is classified as missing.", "missing"
    if "broken" in normalized:
        return "BROKEN", "The strut shows a break or loss of continuity along its measured path.", "warning"
    if "thin" in normalized:
        return "THIN", "The measured cross-section is smaller than the expected strut geometry.", "warning"
    if "inflated" in normalized or "thick" in normalized:
        return "INFLATED / THICK", "The measured strut is larger than expected or extends beyond its expected profile.", "warning"
    if "bent" in normalized:
        return "BENT", "The measured strut curves away from its expected centerline.", "warning"
    return value.replace("_", " ").upper(), "A geometry difference was detected; see the measurements below for detail.", "warning"


def secondary_defect_text(value: Any) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "None"
    text = str(value).strip()
    if text.casefold() in {"", "nan", "null", "none", "none reported"}:
        return "None"
    return text.replace(";", ", ")

def status_card(label: str, value: str, detail: str, tone: str = "neutral") -> None:
    st.markdown(
        f'<div class="status-card status-{tone}"><div class="status-label">{escape(label)}</div>'
        f'<div class="status-value">{escape(value)}</div><div class="status-detail">{escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def category_count_card(label: str, count: int, percent: float, detail: str, intensity: float, tone: str = "neutral") -> None:
    """Render one category card using the all-strut denominator."""
    lightness = 72 - (27 * intensity)
    color = f"hsl(25, 92%, {lightness:.0f}%)"
    percent_text = "<0.01%" if count and percent < 0.01 else f"{percent:.1f}%"
    st.markdown(
        f'<div class="category-card category-{tone}" style="border-left-color:{color}"><div class="status-label">{escape(label)}</div>'
        f'<div class="status-value">{count:,}</div><div class="status-detail">{escape(percent_text)} of all struts — {escape(detail)}</div></div>',
        unsafe_allow_html=True,
    )


def render_statistics_banner(summary: pd.DataFrame, reviews: dict[int, dict[str, Any]]) -> None:
    """Render mutually exclusive top-level categories plus overlapping defect details."""
    frame = summary.copy()
    frame["__category"] = frame.apply(strut_category, axis=1)
    frame["__primary_label"] = frame.apply(primary_label, axis=1)
    total_struts = len(frame)
    counts = frame["__category"].value_counts().reindex(TOP_LEVEL_CATEGORIES, fill_value=0)

    automated_review = frame["needs_review"].map(truthy)
    unresolved = int(sum(automated_review & ~frame["strut_id"].astype(int).isin(reviews)))
    final_counts = pd.Series([item["decision"] for item in reviews.values()]).value_counts().to_dict()
    reviewed_ids = {int(strut_id) for strut_id in reviews}
    summary_ids = set(frame["strut_id"].astype(int))
    saved_nominal_ids = {
        int(strut_id) for strut_id, item in reviews.items()
        if item.get("decision") == "nominal" and int(strut_id) in summary_ids
    }
    unflagged_nominal_ids = set(frame.loc[
        (frame["stage2_classification"] == "Nominal")
        & ~automated_review
        & ~frame["strut_id"].astype(int).isin(reviewed_ids),
        "strut_id",
    ].astype(int))
    final_nominal = len(saved_nominal_ids | unflagged_nominal_ids)

    content_col, queue_col = st.columns([4.0, 1.0])
    with content_col:
        st.markdown("### Strut categories")
        details = [
            "Expected material with no primary defect.",
            "Expected material with a primary defect.",
            "CAD intentionally has no material.",
            "Expected material is absent.",
            "Material exists outside the expected design.",
        ]
        tones = ["present", "warning", "missing", "missing", "warning"]
        columns = st.columns(3)
        maximum = max(int(counts.max()), 1)
        for index, label in enumerate(TOP_LEVEL_CATEGORIES):
            count = int(counts[label])
            with columns[index % len(columns)]:
                category_count_card(
                    label, count, count / total_struts * 100 if total_struts else 0,
                    details[index], count / maximum, tones[index],
                )

        defect_frame = frame[
            (frame["__category"] == "Present by Design — Defect present")
            & frame["__primary_label"].str.strip().str.casefold().map(lambda value: value not in NO_DEFECT_LABELS)
        ]
        defect_counts = defect_frame["__primary_label"].value_counts().sort_values(ascending=False)
        st.markdown("### Primary defect details")
        if defect_counts.empty:
            st.info("No geometry defects detected.")
        else:
            columns = st.columns(3)
            maximum = max(int(defect_counts.max()), 1)
            for index, (label, count) in enumerate(defect_counts.items()):
                with columns[index % len(columns)]:
                    category_count_card(
                        f"Primary defect: {label}", int(count),
                        int(count) / total_struts * 100 if total_struts else 0,
                        "Intentional detail overlap with Defect present.", int(count) / maximum,
                    )
    with queue_col:
        st.markdown("### Review queue")
        metric("Unresolved", f"{unresolved:,}", "Automated review flags without a final saved decision.")
        st.caption(f"Automated flags: {int(automated_review.sum()):,}")
        st.caption(f"Final nominal: {final_nominal:,} / {total_struts:,} ({final_nominal / total_struts:.1%})" if total_struts else "Final nominal: 0 / 0")
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
    st.title("Lattice inspection workspace")
    st.caption("A connected workspace for comparing designed lattice geometry with measured CT morphology.")
    st.markdown("### Mission")
    st.write("Turn CAD-versus-CT differences into traceable, human-verifiable defect decisions. Geometry, measurements, visual evidence, and review history stay together so every decision can be revisited.")
    st.markdown("### Problem statement")
    st.write("Additively manufactured lattices can deviate from their design through missing, broken, thin, inflated, or bent struts. Those deviations are spatially distributed and difficult to assess from one image or scalar measurement.")
    left, right = st.columns(2)
    with left:
        st.markdown("### Data inputs")
        st.write("• CAD/STL: nominal topology, strut paths, junctions, lengths, and expected material.\n\n• CT TIFF: reconstructed voxel intensities describing manufactured material.\n\n• Derived tables: registered centerlines, station measurements, classifications, and review flags.")
    with right:
        st.markdown("### Research questions")
        st.write("• Where is material missing or unexpectedly present?\n\n• Which struts differ in thickness, continuity, or centerline position?\n\n• Which automated findings need a human decision?\n\n• Can a decision be traced to measurements and 3-D evidence?")
    st.markdown("### System capabilities")
    caps = st.columns(4)
    for panel, title, detail in zip(caps, ["Linked evidence", "Quantitative review", "Human-in-the-loop", "Connected services"], ["PyVista CT/lattice views stay linked to the selected strut.", "Station trends expose occupancy, radius, offset, intensity, and excess material.", "Final decisions are persisted by FastAPI and can be reopened.", "FastAPI, Streamlit, PyVista, Plotly, and optional Gemini work together."]):
        with panel:
            st.markdown(f"**{title}**")
            st.caption(detail)
    st.markdown("### Pipeline summary")
    st.info("CAD/STL geometry -> CT TIFF scan -> registration -> strut comparison -> morphology classification -> human review")
    snapshot = st.columns(3)
    with snapshot[0]: metric("Current struts", f"{len(summary):,}")
    with snapshot[1]: metric("Review flags", f"{int(summary.needs_review.map(truthy).sum()):,}" if "needs_review" in summary else "N/A")
    with snapshot[2]: metric("Defect labels", f"{summary.primary_defect.nunique():,}" if "primary_defect" in summary else "N/A")
    team, tech = st.columns(2)
    with team:
        st.markdown("### Team")
        st.caption("Placeholder for team names, roles, and affiliations.")
    with tech:
        st.markdown("### Technology")
        st.caption("Python - FastAPI - Streamlit - PyVista - Plotly - optional Gemini")
    with st.container(border=True):
        st.subheader("Inspection workspace")
        st.write("FastAPI provides the analysis summary and per-strut station evidence; PyVista provides the linked CT visualization.")
        st.success(f"Analysis data ready — {len(summary):,} struts available from FastAPI.")
        st.write("Inspection Workflow documents the pipeline from source data through dashboard review.")
def flowchart(summary: pd.DataFrame) -> None:
    st.title("Inspection Workflow")
    st.caption("Follow the evidence from the designed lattice to a human decision.")

    try:
        reviews = load_reviews()
    except (requests.RequestException, ValueError):
        reviews = {}
    viewer_ready, _ = viewer_status()
    pipeline = load_pipeline_counts()
    indexed = max(int(pipeline.get("indexed", 0)), len(summary))
    compared = max(int(pipeline.get("compared", 0)), len(summary))
    defect_count = int(pipeline.get("defects", 0)) or int(
        (summary.get("primary_defect", pd.Series(dtype=str)).astype(str) != "Nominal").sum()
    )
    review_flags = int(summary.get("needs_review", pd.Series(dtype=bool)).map(truthy).sum())
    reviewed_count = len(reviews)
    unresolved_reviews = max(0, review_flags - reviewed_count)
    scan_shape = pipeline.get("scan_shape", "volume dimensions unavailable")
    stations = int(pipeline.get("stations", 21))
    reference_population = int(pipeline.get("reference_population", 0))

    stages = [
        {
            "title": "CAD geometry load",
            "one_liner": "The design tells us where each strut should be.",
            "status": f"{indexed:,} struts indexed",
            "tone": "complete" if indexed else "attention",
            "marker": "✓" if indexed else "01",
            "explanation": "Think of this as the blueprint for the lattice: it lists every expected strut and how the pieces should connect.",
            "technical": f"Source: all_struts_inventory.csv · {indexed:,} rows loaded.",
        },
        {
            "title": "CT scan ingest",
            "one_liner": "The scan shows what was actually made.",
            "status": "Viewer connected" if viewer_ready else "Viewer offline",
            "tone": "complete" if viewer_ready else "attention",
            "marker": "✓" if viewer_ready else "⚠",
            "explanation": "Like taking a detailed photograph of the finished lattice, this step brings the measured material into the workspace.",
            "technical": f"Scan volume: {scan_shape} voxels · viewer {'available' if viewer_ready else 'not reachable'}.",
        },
        {
            "title": "Registration",
            "one_liner": "The design and scan are lined up in the same place.",
            "status": "Registration complete" if compared else "Needs attention",
            "tone": "complete" if compared else "attention",
            "marker": "✓" if compared else "03",
            "explanation": "This is like placing tracing paper over a blueprint so every designed strut can be compared with the matching piece in the scan.",
            "technical": f"Reference population: {reference_population:,} designed positions." if reference_population else "Registered comparison coordinates are not available.",
        },
        {
            "title": "Strut comparison",
            "one_liner": "Each expected strut is checked against its measured shape.",
            "status": f"{compared:,} struts compared",
            "tone": "complete" if compared else "progress",
            "marker": "✓" if compared else "04",
            "explanation": "Like checking every item on a packing list, the system looks for missing material, breaks, thin spots, bulges, and shifts.",
            "technical": f"{compared:,} struts · {stations} measurement points sampled along each strut.",
            "target": "Explore Defects",
        },
        {
            "title": "Morphology classification",
            "one_liner": "Measured differences are grouped into clear finding types.",
            "status": f"{defect_count:,} defects found" if defect_count else "No defects found",
            "tone": "defect" if defect_count else "complete",
            "marker": "⚠" if defect_count else "✓",
            "explanation": "This is the sorting step: each unusual strut is placed into a simple category so a reviewer can understand what needs attention.",
            "technical": f"{defect_count:,} non-nominal findings across {compared:,} compared struts.",
            "target": "Defect Atlas",
        },
        {
            "title": "Human review",
            "one_liner": "A person checks the evidence and records the final call.",
            "status": f"{unresolved_reviews:,} flags to review" if unresolved_reviews else f"{reviewed_count:,} decisions saved",
            "tone": "defect" if unresolved_reviews else "progress",
            "marker": "⚠" if unresolved_reviews else "06",
            "explanation": "Like a second pair of eyes on a quality checklist, a reviewer confirms the finding using the linked images, measurements, and notes.",
            "technical": f"{reviewed_count:,} saved decisions · {review_flags:,} review flags from the analysis.",
            "target": "Review History",
        },
    ]

    if "selected_stage" not in st.session_state:
        st.session_state.selected_stage = int(st.session_state.get("workflow_step", 0)) + 1
    selected_stage = max(1, min(len(stages), int(st.session_state.selected_stage)))
    active = selected_stage - 1
    st.session_state.selected_stage = selected_stage
    st.session_state.workflow_step = active
    if "workflow_expanded" not in st.session_state:
        st.session_state.workflow_expanded = active
    expanded = st.session_state.workflow_expanded

    light_mode = st.toggle("Use light mode", value=st.session_state.get("workflow_light_mode", False), key="workflow_light_mode")
    st.markdown(f"<div class='workflow-summary-heading'>Stage {selected_stage} of 6</div>", unsafe_allow_html=True)
    components.html(f"""
    <style>
      :root {{ --stat-bg: {'#ffffff' if light_mode else '#111827'}; --stat-border: {'#cbd5e1' if light_mode else '#334155'}; --stat-text: {'#0f172a' if light_mode else '#f8fafc'}; --stat-muted: {'#475569' if light_mode else '#94a3b8'}; }}
      .workflow-stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; font-family:Arial,sans-serif; }}
      .workflow-stat {{ background:var(--stat-bg); border:1px solid var(--stat-border); border-radius:10px; padding:12px 14px; }}
      .workflow-stat-label {{ color:var(--stat-muted); font-size:12px; font-weight:500; }}
      .workflow-stat-value {{ color:var(--stat-text); font-size:24px; font-weight:500; margin-top:5px; }}
    </style>
    <div class="workflow-stats">
      <div class="workflow-stat"><div class="workflow-stat-label">Struts indexed</div><div class="workflow-stat-value" data-count="{indexed}">0</div></div>
      <div class="workflow-stat"><div class="workflow-stat-label">Struts compared</div><div class="workflow-stat-value" data-count="{compared}">0</div></div>
      <div class="workflow-stat"><div class="workflow-stat-label">Defects found</div><div class="workflow-stat-value" data-count="{defect_count}">0</div></div>
    </div>
    <script>
      const ease = t => 1 - Math.pow(1 - t, 4);
      document.querySelectorAll('[data-count]').forEach(node => {{
        const target = Number(node.dataset.count), start = performance.now(), duration = 900;
        const tick = now => {{
          const progress = Math.min(1, (now - start) / duration);
          node.textContent = Math.round(target * ease(progress)).toLocaleString();
          if (progress < 1) requestAnimationFrame(tick);
        }};
        requestAnimationFrame(tick);
      }});
    </script>
    """, height=104, scrolling=False)

    tone_vars = {"complete": "green", "attention": "amber", "defect": "red", "progress": "blue"}
    stage_css = "\n".join(
        f"""
    .st-key-workflow-card-{index} {{ position:relative; padding-left:3.15rem; margin:0; }}
    .st-key-workflow-card-{index} .stButton {{ position:relative; z-index:1; transition:transform 300ms ease; }}
    .st-key-workflow-card-{index} .stButton:hover {{ transform:translateX(4px); }}
    .st-key-workflow-card-{index} button {{ position:relative; min-height:86px; padding:1rem 3rem 1rem 1.2rem; border:1px solid var(--workflow-border); border-radius:10px; background:var(--workflow-surface); color:var(--workflow-text); text-align:left; white-space:pre-line; font-size:1rem; font-weight:500; line-height:1.45; transition:background-color 300ms ease, border-color 300ms ease; }}
    .st-key-workflow-card-{index} button:hover {{ background:var(--workflow-surface-hover); border-color:var(--workflow-blue); }}
    .st-key-workflow-card-{index} button::before {{ content:'{stage["marker"]}'; position:absolute; left:-2.7rem; top:1.1rem; width:2rem; height:2rem; display:grid; place-items:center; border:2px solid var(--workflow-{tone_vars[stage["tone"]]}); border-radius:50%; background:var(--workflow-bg); color:var(--workflow-{tone_vars[stage["tone"]]}); font-size:.8rem; font-weight:500; transition:transform 300ms cubic-bezier(.34,1.56,.64,1), border-color 300ms ease, color 300ms ease; }}
    .st-key-workflow-card-{index} button:hover::before {{ transform:scale(1.1); border-color:var(--workflow-blue); color:var(--workflow-blue); }}
    .st-key-workflow-card-{index} button::after {{ content:'›'; position:absolute; right:1.2rem; top:1.3rem; color:var(--workflow-muted); font-size:1.5rem; transition:transform 300ms ease, color 300ms ease; }}
    .st-key-workflow-card-{index} .workflow-card-status {{ color:var(--workflow-{tone_vars[stage["tone"]]}); }}
    .st-key-workflow-card-{index} .workflow-detail-inner {{ max-height:0; overflow:hidden; opacity:0; margin:0; padding:0 .2rem; transition:max-height 300ms ease, opacity 300ms ease, margin 300ms ease; }}
    .st-key-workflow-card-{index} .workflow-detail-inner p {{ color:var(--workflow-text); margin:.25rem 0 .55rem; line-height:1.55; font-weight:500; }}
    """
        for index, stage in enumerate(stages)
    )
    expanded_css = ""
    if expanded is not None:
        expanded_css = f"""
    .st-key-workflow-card-{expanded} button::after {{ transform:rotate(90deg); color:var(--workflow-blue); }}
    .st-key-workflow-card-{expanded} .workflow-detail-inner {{ max-height:190px; opacity:1; margin:.2rem 0 1rem; }}
    """
    pulse_css = ""
    if stages[active]["tone"] in {"progress", "attention"}:
        pulse_css = f".st-key-workflow-card-{active} button::before {{ animation:workflow-ring 1.8s ease-in-out infinite; }}"
    theme_vars = (
        "--workflow-bg:#f8fafc; --workflow-surface:#ffffff; --workflow-surface-hover:#f1f5f9; --workflow-border:#cbd5e1; --workflow-text:#0f172a; --workflow-muted:#475569; --workflow-blue:#0284c7; --workflow-green:#15803d; --workflow-amber:#a16207; --workflow-red:#b91c1c; --workflow-line:#94a3b8;"
        if light_mode else
        "--workflow-bg:#0f172a; --workflow-surface:#111827; --workflow-surface-hover:#1e293b; --workflow-border:#334155; --workflow-text:#f8fafc; --workflow-muted:#94a3b8; --workflow-blue:#38bdf8; --workflow-green:#4ade80; --workflow-amber:#fbbf24; --workflow-red:#f87171; --workflow-line:#475569;"
    )
    st.markdown(f"""<style>
    :root {{ {theme_vars} }}
    .workflow-summary-heading {{ color:var(--workflow-muted); font-size:.82rem; font-weight:500; margin:.8rem 0 .2rem; }}
    .workflow-timeline {{ max-width:860px; margin:.5rem auto 0; padding:0 1rem; }}
    .workflow-card-status {{ display:inline-block; margin:.55rem 0 .35rem; border:1px solid currentColor; border-radius:999px; padding:.22rem .65rem; font-size:.74rem; font-weight:500; }}
    .workflow-connector {{ height:1.2rem; margin-left:1.25rem; border-left:2px dashed var(--workflow-line); position:relative; }}
    .workflow-connector.workflow-flowing::after {{ content:''; position:absolute; left:-3px; top:0; width:4px; height:10px; background:var(--workflow-blue); animation:workflow-dash 1s linear infinite; }}
    @keyframes workflow-dash {{ to {{ transform:translateY(1.2rem); }} }}
    @keyframes workflow-ring {{ 0%,100% {{ outline:0 solid transparent; }} 50% {{ outline:4px solid var(--workflow-blue); outline-offset:2px; }} }}
    @media (prefers-reduced-motion: reduce) {{ .workflow-connector.workflow-flowing::after, * {{ animation:none !important; }} }}
    {{stage_css}}
    {{expanded_css}}
    {{pulse_css}}
    </style>""", unsafe_allow_html=True)

    st.markdown(f"<div class='workflow-shell {'workflow-light' if light_mode else ''}'><div class='workflow-timeline'>", unsafe_allow_html=True)
    for index, stage in enumerate(stages):
        is_expanded = expanded == index
        with st.container(key=f"workflow-card-{index}"):

            if st.button(f"{index + 1:02d}  {stage['title']}\n{stage['one_liner']}", key=f"workflow_step_{index}", width="stretch"):
                st.session_state.selected_stage = index + 1
                st.session_state.workflow_step = index
                st.session_state.workflow_expanded = None if expanded == index else index
                st.rerun()
            st.markdown(f"<div class='workflow-card-status'>{stage['status']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='workflow-detail-inner'><p>{stage['explanation']}</p><div class='workflow-technical'>{stage['technical']}</div></div>", unsafe_allow_html=True)
        if index < len(stages) - 1:
            connector_class = "workflow-connector workflow-flowing" if index < active else "workflow-connector"
            st.markdown(f"<div class='{connector_class}'></div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.divider()
    if st.button("Start reviewing defects", key="workflow_start_review", type="primary", icon=":material/play_arrow:"):
        st.session_state.page = "Explore Defects"
        st.rerun()
ATLAS_DEFS = {
    "Missing": ("Missing", "Missing_Unintentional", "Missing_Intentional"),
    "Broken": ("Broken",), "Thin": ("Thin",), "Inflated": ("Inflated",), "Bent": ("Bent",)
}
ATLAS_TEXT = {
    "Missing": ("A strut expected by the design is absent, or material is classified as missing.", "Stage 2 missing-material classification; compare expected CAD occupancy with CT material mask.", "occupancy fraction / voxel-derived material mask", "Can remove a load path and reduce stiffness or connectivity."),
    "Broken": ("A strut contains a discontinuity or a sustained low-material run.", "Broken score and longest low-material run across stations.", "station count / occupancy fraction", "May behave like a crack or complete loss of load transfer."),
    "Thin": ("Measured cross-section is smaller than the nominal strut geometry.", "Low diameter or radius relative to nominal; thin score.", "um and diameter-to-nominal ratio", "Reduces local load capacity and can increase buckling risk."),
    "Inflated": ("Measured material is larger than nominal or extends outside the expected profile.", "Outside-nominal material fraction and inflated score.", "um and outside-material fraction", "Can change local stiffness, pore size, and downstream flow or fit."),
    "Bent": ("The measured centerline departs from the designed path.", "Maximum centerline offset and bend curvature score.", "um and curvature-derived score", "Changes load direction and may create eccentric or concentrated stresses."),
}

def defect_atlas(summary: pd.DataFrame) -> None:
    st.title("Defect Atlas")
    st.caption("A quick reference for the morphology labels used in the available analysis tables.")
    st.info("Counts are derived from the current FastAPI summary. A zero count means no matching label is available, not necessarily that the defect is impossible.")
    for name, aliases in ATLAS_DEFS.items():
        count = int(summary.primary_defect.astype(str).isin(aliases).sum()) if "primary_defect" in summary else 0
        matches = summary[summary.primary_defect.astype(str).isin(aliases)] if "primary_defect" in summary else pd.DataFrame()
        example = int(matches.iloc[0].strut_id) if not matches.empty else None
        meaning, rule, units, impact = ATLAS_TEXT[name]
        with st.container(border=True):
            title, count_col, action = st.columns([2.2, 1, 1.4])
            with title: st.subheader(name)
            with count_col: metric("Available count", f"{count:,}")
            with action:
                if st.button("Inspect example strut", key=f"atlas_{name}", disabled=example is None, width="stretch"):
                    st.session_state.selected_strut_id = example
                    st.session_state.selected_strut_picker = example
                    st.session_state.page = "Explore Defects"
                    st.rerun()
            detail_a, detail_b = st.columns(2)
            with detail_a: st.markdown(f"**What it means**  \n{meaning}\n\n**Detection rule**  \n{rule}")
            with detail_b: st.markdown(f"**Measurement units**  \n{units}\n\n**Likely structural impact**  \n{impact}")

def review_history(summary: pd.DataFrame) -> None:
    st.title("Review History")
    st.caption("Persisted human decisions returned by the FastAPI broker.")
    try:
        reviews = load_reviews()
    except (requests.RequestException, ValueError) as exc:
        st.warning(f"Review history is unavailable: {exc}")
        return
    decisions = pd.DataFrame([{"strut_id": sid, **record} for sid, record in reviews.items()])
    counts = decisions["decision"].value_counts().to_dict() if not decisions.empty else {}
    cards = st.columns(4)
    for panel, label, value in zip(cards, ["Total decisions", "Confirmed defects", "Marked nominal", "Needs review"], [len(decisions), counts.get("confirmed_defect", 0), counts.get("nominal", 0), counts.get("needs_review", 0)]):
        with panel: metric(label, f"{value:,}")
    if decisions.empty:
        st.info("No persisted review decisions yet. Save a decision from Explore Defects to begin the history.")
        return
    st.progress(min(1.0, len(decisions) / max(1, len(summary))), text=f"{len(decisions):,} of {len(summary):,} struts have a saved decision")
    history = pd.DataFrame([{"strut_id": int(row.strut_id), "decision": str(row.get("decision", "")).replace("_", " ").title(), "timestamp": row.get("updated_utc", row.get("timestamp", "N/A")), "notes": row.get("notes", row.get("note", "")) or ""} for _, row in decisions.sort_values("strut_id").iterrows()])
    event = st.dataframe(history, hide_index=True, width="stretch", height=360, on_select="rerun", selection_mode="single-row", key="review_history_table")
    selected_rows = getattr(getattr(event, "selection", None), "rows", [])
    if selected_rows:
        selected_id = int(history.iloc[selected_rows[0]]["strut_id"])
        if st.button(f"Reopen strut {selected_id}", key="reopen_selected_review"):
            st.session_state.selected_strut_id = selected_id
            st.session_state.selected_strut_picker = selected_id
            st.session_state.page = "Explore Defects"
            st.rerun()

def analysis(summary):
    st.title("Explore Defects"); st.caption("Search, measure, visualize, chat about, and review one strut at a time.")
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
        "ratio": col(summary, SUMMARY["ratio"] + ["diameter_ratio_to_nominal", "diameter_to_nominal_ratio"]),
        "diameter": col(summary, SUMMARY["diameter"]),
        "bent": col(summary, SUMMARY["curvature"]),
        "review": col(summary, SUMMARY["needs_review"]),
    }
    metric_actions = []
    if source_columns["occupancy"]:
        metric_actions.append(("Lowest occupancy", "lowest_occupancy"))
    if source_columns["deviation"]:
        metric_actions.append(("Largest deviation", "largest_deviation"))
    if source_columns["ratio"] or source_columns["diameter"]:
        metric_actions.extend([("Thinnest", "thinnest"), ("Thickest/inflated", "thickest_inflated")])
    if source_columns["bent"]:
        metric_actions.append(("Most bent", "most_bent"))

    available_analysis = [
        ("Material occupancy", source_columns["occupancy"]),
        ("Centerline deviation", source_columns["deviation"]),
        ("Effective diameter or radius", source_columns["ratio"] or source_columns["diameter"]),
        ("Curvature or tortuosity", source_columns["bent"]),
        ("Primary defect type", col(summary, SUMMARY["primary_defect"])),
        ("Review flags", source_columns["review"]),
        ("Material status", col(summary, SUMMARY["stage2_classification"])),
    ]
    def quick_candidates(choice: str) -> tuple[pd.DataFrame, int | None, str | None]:
        """Rank the complete loaded summary, independent of the search box."""
        frame = summary.copy()
        if choice == "Random flagged strut":
            picked = st.session_state.get("selected_strut_id")
            return frame, int(picked) if picked is not None else None, "Random flagged strut."

        if "strut_id" not in frame:
            return frame, None, "The loaded data does not include Strut IDs."

        if choice == "Needs review":
            review_source = source_columns["review"]
            if review_source is None:
                return frame, None, "Needs review is unavailable because review flags are not in the loaded data."
            flagged = frame[frame[review_source].map(truthy)].copy()
            resolved_ids = {
                int(strut_id)
                for strut_id, record in reviews.items()
                if str(record.get("decision", "")).casefold() != "needs_review"
            }
            flagged = flagged[~flagged["strut_id"].astype(int).isin(resolved_ids)]
            if flagged.empty:
                return flagged, None, "There are no unresolved review flags in the loaded data."
            severity_source = col(flagged, SUMMARY["severity"])
            if severity_source:
                numeric_severity = pd.to_numeric(flagged[severity_source], errors="coerce")
                if numeric_severity.notna().any():
                    flagged["__severity_rank"] = numeric_severity
                else:
                    severity_names = {"critical": 4, "high": 3, "medium": 2, "low": 1}
                    flagged["__severity_rank"] = flagged[severity_source].astype(str).str.casefold().map(severity_names).fillna(0)
            else:
                flagged["__severity_rank"] = 0
            ranked = flagged.sort_values(["__severity_rank", "strut_id"], ascending=[False, True]).drop(columns="__severity_rank")
            picked = int(ranked.iloc[0]["strut_id"])
            return ranked, picked, f"Selected strut {picked}: needs review."

        metric_sources = {
            "Lowest occupancy": source_columns["occupancy"],
            "Largest deviation": source_columns["deviation"],
            "Thinnest": source_columns["ratio"] or source_columns["diameter"],
            "Thickest/inflated": source_columns["ratio"] or source_columns["diameter"],
            "Most bent": source_columns["bent"],
        }
        source = metric_sources[choice]
        if source is None:
            return frame, None, f"{choice} is unavailable because its metric is not in the loaded data."

        if choice == "Thickest/inflated" and "primary_defect" in frame:
            inflated = frame[frame["primary_defect"].astype(str).str.casefold() == "inflated"]
            if not inflated.empty:
                frame = inflated.copy()

        ranked = frame.copy()
        ranked["__quick_metric"] = pd.to_numeric(ranked[source], errors="coerce")
        ranked = ranked.dropna(subset=["__quick_metric"])
        if ranked.empty:
            return ranked, None, f"{choice} is unavailable because the metric has no usable values."
        ascending = choice in {"Lowest occupancy", "Thinnest"}
        ranked = ranked.sort_values(["__quick_metric", "strut_id"], ascending=[ascending, True]).drop(columns="__quick_metric")
        picked = int(ranked.iloc[0]["strut_id"])
        metric_label = {
            "Lowest occupancy": "lowest occupancy",
            "Largest deviation": "largest centerline deviation",
            "Thinnest": "thinnest strut",
            "Thickest/inflated": "thickest/inflated strut",
            "Most bent": "most bent strut",
        }[choice]
        return ranked, picked, f"Selected strut {picked}: {metric_label}."

    def activate_quick_pick(choice: str) -> None:
        # A shortcut always ranks the complete loaded dataset, never the text
        # search or filter subset currently visible on the page.
        for filter_key in filter_keys:
            st.session_state.pop(filter_key, None)
        _candidates, picked, message = quick_candidates(choice)
        st.session_state.quick_pick_message = message
        st.session_state.quick_pick_message_tone = "success" if picked is not None else "warning"
        if picked is not None:
            st.session_state.quick_pick_state = choice
            st.session_state.quick_pick_sort = choice
            st.session_state.show_matching_struts = True
            st.session_state.selected_strut_id = picked
            st.session_state.selected_strut_picker = picked
            st.session_state.page = "Explore Defects"
        else:
            st.session_state.pop("quick_pick_state", None)
            st.session_state.pop("quick_pick_sort", None)
            st.session_state.show_matching_struts = False

    def clear_launcher_filters() -> None:
        for filter_key in filter_keys:
            st.session_state.pop(filter_key, None)

    def activate_random_flagged() -> None:
        clear_launcher_filters()
        review_source = source_columns["review"]
        if review_source is None:
            st.session_state.quick_pick_message = "Random flagged strut is unavailable because review flags are not in the loaded data."
            st.session_state.quick_pick_message_tone = "warning"
            return
        flagged = summary[summary[review_source].map(truthy)]
        if flagged.empty:
            st.session_state.quick_pick_message = "No flagged struts are available in the loaded data."
            st.session_state.quick_pick_message_tone = "warning"
            return
        picked = int(flagged.sample(n=1).iloc[0].strut_id)
        st.session_state.quick_pick_state = "Random flagged strut"
        st.session_state.quick_pick_sort = "Random flagged strut"
        st.session_state.show_matching_struts = True
        st.session_state.selected_strut_id = picked
        st.session_state.selected_strut_picker = picked
        st.session_state.page = "Explore Defects"
        st.session_state.quick_pick_message = f"Selected strut {picked}: random flagged strut."
        st.session_state.quick_pick_message_tone = "success"

    def activate_next_unresolved() -> None:
        activate_quick_pick("Needs review")

    def browse_all_struts() -> None:
        clear_launcher_filters()
        st.session_state.pop("quick_pick_state", None)
        st.session_state.pop("quick_pick_sort", None)
        st.session_state.show_matching_struts = True
        st.session_state.page = "Explore Defects"
        st.session_state.quick_pick_message = "Showing all struts."
        st.session_state.quick_pick_message_tone = "success"
    def reset_filters() -> None:
        for filter_key in filter_keys:
            st.session_state.pop(filter_key, None)
        st.session_state.pop("quick_pick_state", None)
        st.session_state.pop("quick_pick_sort", None)
        st.session_state.pop("selected_strut_id", None)
        st.session_state.pop("selected_strut_picker", None)
        st.session_state.show_matching_struts = False
        st.session_state.quick_pick_message = "Filters reset; showing the full dataset."
        st.session_state.quick_pick_message_tone = "success"

    with st.container(border=True):
        st.subheader("Inspection launcher")
        search = st.text_input("Strut ID", placeholder="Search strut ID, e.g. 1728", key="filter_search")
        primary, stage2, review, sev = [], [], [], []
        st.caption(f"Loaded dataset: {len(summary):,} struts")
        launcher = st.columns(3)
        with launcher[0]:
            st.button("Open random flagged strut", key="open_random_flagged", width="stretch", on_click=activate_random_flagged)
        with launcher[1]:
            st.button("Open next unresolved review", key="open_next_unresolved", width="stretch", on_click=activate_next_unresolved)
        with launcher[2]:
            st.button("Browse all struts", key="browse_all_struts", width="stretch", on_click=browse_all_struts)

        if metric_actions:
            st.caption("Metric ranking")
            metric_buttons = st.columns(min(3, len(metric_actions)))
            for index, (label, action) in enumerate(metric_actions):
                metric_buttons[index % len(metric_buttons)].button(
                    label,
                    key=f"quick_pick_{action}",
                    width="stretch",
                    on_click=activate_quick_pick,
                    args=(label,),
                )

        with st.expander("Available analysis", expanded=False):
            available = [f"{label} ({column})" for label, column in available_analysis if column]
            unavailable = [label for label, column in available_analysis if not column]
            panel_left, panel_right = st.columns(2)
            with panel_left:
                st.caption("Available")
                st.write(" · ".join(available) if available else "No optional analysis fields detected.")
            with panel_right:
                st.caption("Unavailable")
                st.write(" · ".join(unavailable) if unavailable else "All listed analysis fields are available.")

        st.button("Reset filters", key="reset_filters", use_container_width=False, on_click=reset_filters)

        quick_message = st.session_state.pop("quick_pick_message", None)
        if quick_message:
            if st.session_state.pop("quick_pick_message_tone", "success") == "warning":
                st.warning(quick_message)
            else:
                st.success(quick_message)
    quick_state = st.session_state.get("quick_pick_state")
    quick_active = bool(quick_state and not search.strip() and not primary and not stage2 and not review and not sev)
    if quick_active:
        filtered, quick_selected, _ = quick_candidates(quick_state)
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
    st.subheader("Strut ID selector")
    filtered = summary.copy()
    if search.strip():
        filtered = filtered[filtered.strut_id.astype(str).str.contains(search.strip(), regex=False)]
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
        st.warning("No struts match this Strut ID search."); return

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
    def nominal_median(aliases, convert_radius_to_diameter=False):
        source = col(summary, aliases)
        if source is None:
            return None
        values = pd.to_numeric(
            summary.loc[summary["stage2_classification"] == "Nominal", source], errors="coerce"
        ).dropna()
        if values.empty:
            return None
        baseline = float(values.map(lambda number: scale_to_um(number, source)).median())
        return baseline * 2 if convert_radius_to_diameter else baseline

    st.divider(); st.subheader(f"Selected strut {current}")
    radius, radius_source = metric_number(["median_cross_section_radius_um", "median_radius_um"])
    radius = scale_to_um(radius, radius_source)
    length, length_source = metric_number(["length_um", "inventory_length_um", "length_um_from_centerline"])
    length = scale_to_um(length, length_source)
    status, status_detail, tone = material_status(selected)
    secondary_raw = selected.get("secondary_defects", "")
    secondary = (
        "Not Listed"
        if pd.isna(secondary_raw) or str(secondary_raw).strip().casefold() in {"", "nan", "none", "null", "<na>"}
        else str(secondary_raw).replace(";", ", ")
    )
    nominal_length = nominal_median(["length_um", "inventory_length_um", "length_um_from_centerline"])
    nominal_diameter = nominal_median(
        ["median_cross_section_radius_um", "median_radius_um"], convert_radius_to_diameter=True
    )
    classification, classification_detail, classification_tone = geometry_classification(selected)
    secondary = secondary_defect_text(selected.get("secondary_defects"))
    current_review = reviews.get(int(current), {}).get("decision", "No final decision")
    st.markdown(
        f'<div class="classification-card classification-{classification_tone}">'
        f'<div class="classification-label">Overall classification</div>'
        f'<div class="classification-badge">{escape(classification)}</div>'
        f'<div class="classification-detail">{escape(classification_detail)}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("Material status describes whether material was expected and found. Overall classification describes the measured shape of the strut.")
    cards = st.columns(4)
    with cards[0]: status_card("Material status", status, status_detail, tone)
    with cards[1]: status_card("Secondary defects", secondary, "Additional geometry signals beyond the overall classification.")
    with cards[2]: status_card("Length", fmt(length, 2, "µm"), f"All nominal struts median: {fmt(nominal_length, 2, 'µm')}")
    with cards[3]: status_card("Median diameter", fmt(None if radius is None else radius * 2, 2, "µm"), f"All nominal struts median: {fmt(nominal_diameter, 2, 'µm')}")
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
    st.markdown("**Chat context**"); st.caption(f"Questions use FastAPI-backed values for selected strut {current}. Suggested prompts fill the editable message box.")
    provider = st.selectbox("Chat provider", options=["fastapi", "gemini"], format_func=lambda value: "FastAPI (local analysis)" if value == "fastapi" else "Gemini", key="chat_provider", help="FastAPI answers with local deterministic analysis. Gemini answers using read-only dashboard data tools.")
    for message in st.session_state.get("chat_messages", []):
        with st.chat_message(message["role"]): st.markdown(message["content"])
    defect_categories = sorted(
        {
            str(category) for category in summary["primary_defect"].dropna()
            if str(category).strip().casefold() not in {"", "nominal", "nan", "unknown"}
        },
        key=str.casefold,
    )
    suggested_prompts = {
        "Current strut": [
            ("Inspect this strut", f"Inspect strut {current}"),
            ("Largest deviation", f"Where is the largest deviation on strut {current}?"),
            ("Lowest occupancy", f"What is the lowest occupancy on strut {current}?"),
        ],
        "Compare / select": [
            ("Compare with nominal", f"Compare strut {current} with all nominal struts"),
            ("Review candidates", "List struts needing review"),
            ("Rank deviations", "Show the top 10 struts by maximum deviation"),
        ],
        "Dataset overview": [
            ("Defect summary", "Give me a defect summary"),
            ("Explain occupancy", "What does material occupancy mean?"),
            ("Explain Stage 2", "What is Stage 2 classification?"),
        ],
        "Inspect by defect": [
            (f"All {display_classification(category)}", f"Inspect all struts with primary defect {display_classification(category)}")
            for category in defect_categories
        ],
    }
    def set_chat_draft(prompt: str) -> None:
        st.session_state.chat_draft = prompt
    st.caption("Suggested prompts")
    prompt_groups = st.columns(len(suggested_prompts))
    for panel, (group, prompts) in zip(prompt_groups, suggested_prompts.items()):
        with panel:
            st.markdown(f"*{group}*")
            for label, prompt in prompts:
                st.button(label, key=f"chat_prompt_{group}_{label}_{current}", width="stretch", on_click=set_chat_draft, args=(prompt,))
    question = st.chat_input(f"Ask about strut {current}, selected struts, or the dataset", key="chat_draft")
    if question:
        messages = st.session_state.setdefault("chat_messages", [])
        messages.append({"role": "user", "content": question})
        try:
            response = api_post(
                "/chat",
                {"message": question, "active_strut_id": current, "chat_history": messages[-4:], "provider": provider},
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
.stApp{background:#0b1120}[data-testid="stSidebar"]{background:#111827;border-right:1px solid #293750}.metric-card,.status-card,.category-card{background:#172033;border:1px solid #293750;border-radius:12px;padding:14px 16px;height:150px;min-height:150px;box-sizing:border-box}.status-card{height:180px;min-height:180px}.metric-card,.status-card,.category-card{display:flex;flex-direction:column}.category-card{border-left:4px solid #f97316;margin-bottom:10px}.metric-label,.status-label{color:#94a3b8;font-size:.76rem;text-transform:uppercase;letter-spacing:.06em}.metric-value,.status-value{color:#f8fafc;font-size:1.15rem;font-weight:650;margin-top:7px}.status-detail{color:#cbd5e1;font-size:.78rem;margin-top:6px}.status-present,.category-present{border-left-color:#22c55e}.status-missing,.category-missing{border-left-color:#ef4444}.status-warning,.category-warning{border-left-color:#f59e0b}.flow-stage{background:#172033;border:1px solid #334155;border-radius:12px;padding:18px;text-align:center}.flow-number{color:#38bdf8;font-size:.72rem}.flow-title{color:#f8fafc;font-weight:700;margin-top:8px}.flow-description{color:#cbd5e1;padding:18px 4px}.flow-arrow{color:#38bdf8;font-size:2rem;text-align:center;padding-top:18px}
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
    st.caption("Stage 4 inspection")
    st.session_state.setdefault("page", "Overview")
    nav_items = [("Overview", ":material/dashboard:"), ("Inspection Workflow", ":material/account_tree:"), ("Explore Defects", ":material/analytics:"), ("Defect Atlas", ":material/menu_book:")]
    for label, icon in nav_items:
        if st.button(label, icon=icon, key=f"nav_{label}", type="primary" if st.session_state.page == label else "secondary", width='stretch'):
            st.session_state.page = label
            st.rerun()
page = st.session_state.page
if page=="Overview": overview(summary)
elif page=="Inspection Workflow": flowchart(summary)
elif page=="Explore Defects": analysis(summary)
elif page=="Defect Atlas": defect_atlas(summary)
elif page=="Review History": review_history(summary)
