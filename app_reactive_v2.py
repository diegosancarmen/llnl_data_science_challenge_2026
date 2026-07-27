from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tifffile
try:
    import pyvista as pv
    from stpyvista import stpyvista
    HAS_PYVISTA = True
except ImportError:
    pv = None
    stpyvista = None
    HAS_PYVISTA = False

ROOT = Path(__file__).parent
INVENTORY = ROOT / "part2/stage_2a_developer_output/all_struts_inventory.csv"
CENTERLINES = ROOT / "part2/napari_visualizer/napari_centerlines.csv"
SCAN = next((p for p in [
    ROOT / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif",
    ROOT / "part2/data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif",
] if p.exists()), None)
UPLOADS = ROOT / "outputs/streamlit_uploads"

st.set_page_config(page_title="Lattice NDE Dashboard", layout="wide")

REQUIRED = {
    "strut_id", "inventory_unit_cell_ids", "inventory_classification",
    "start_z_vox", "start_y_vox", "start_x_vox", "direction_z_vox",
    "direction_y_vox", "direction_x_vox", "center_z_vox", "center_y_vox",
    "center_x_vox", "bbox_z_min_inclusive", "bbox_y_min_inclusive",
    "bbox_x_min_inclusive", "bbox_z_max_exclusive", "bbox_y_max_exclusive",
    "bbox_x_max_exclusive",
}


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def read_scan(path):
    volume = tifffile.memmap(path)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF, got shape {volume.shape}")
    return volume


def save_upload(upload):
    if upload is None:
        return None
    UPLOADS.mkdir(parents=True, exist_ok=True)
    # Streamlit reruns the script whenever a widget changes. An uploaded CT
    # scan may already be memory-mapped by a previous run, so avoid replacing
    # an identical file on Windows (which can raise Errno 22).
    filename = Path(upload.name).name.rstrip(" .") or "uploaded_scan.tif"
    path = UPLOADS / filename
    signature = (filename, upload.size)
    key = f"upload_{path.name}"
    if path.exists() and path.stat().st_size == upload.size:
        st.session_state[key] = signature
        return path
    if st.session_state.get(key) != signature or not path.exists():
        path.write_bytes(upload.getbuffer())
        st.session_state[key] = signature
    return path


def pv_viewer(table, selected_id, indices, selected_index, lower, crop, threshold, mode, camera):
    """Render the lattice with batched meshes to keep Streamlit responsive."""
    plotter = pv.Plotter(window_size=(1400, 820))
    plotter.set_background("#111827")
    visible = table.index.tolist() if mode == "Whole Lattice" else indices
    origin = np.zeros(3) if lower is None else np.asarray(lower, float)
    points, cells, colors = [], [], []
    selected_segment = None

    for index in visible:
        row = table.loc[index]
        start = row[["start_x_vox", "start_y_vox", "start_z_vox"]].to_numpy(float) - origin[[2, 1, 0]]
        end = start + row[["direction_x_vox", "direction_y_vox", "direction_z_vox"]].to_numpy(float)
        if index == selected_index:
            selected_segment = (start, end)
            continue
        base = len(points)
        points.extend([start, end])
        cells.extend([2, base, base + 1])
        rgb = tuple(int(color(row["inventory_classification"])[i:i + 2], 16) for i in (1, 3, 5))
        colors.append(rgb)

    if points:
        mesh = pv.PolyData(np.asarray(points), lines=np.asarray(cells, dtype=np.int64))
        mesh.cell_data["strut_color"] = np.asarray(colors, dtype=np.uint8)
        plotter.add_mesh(mesh, scalars="strut_color", rgb=True, line_width=3)

    if selected_segment is not None:
        start, end = selected_segment
        plotter.add_mesh(pv.Line(start, end), color="#facc15", line_width=10)
        radius = max(float(np.linalg.norm(end - start)) * 0.035, 0.5)
        plotter.add_mesh(pv.Sphere(radius=radius, center=start), color="white")
        plotter.add_mesh(pv.Sphere(radius=radius, center=end), color="white")

    if crop is not None:
        step = max(1, int(max(crop.shape) / 70))
        points = np.argwhere(crop[::step, ::step, ::step] >= threshold) * step
        # Keep browser payloads bounded for dense CT regions.
        if len(points) > 20000:
            points = points[np.linspace(0, len(points) - 1, 20000, dtype=int)]
        if len(points):
            cloud = pv.PolyData(points[:, [2, 1, 0]].astype(float))
            plotter.add_mesh(cloud, color="#d1d5db", point_size=3, render_points_as_spheres=True)

    # Fit geometry first, then orient the camera so no side is clipped.
    plotter.reset_camera()
    if camera == "Isometric": plotter.view_isometric()
    elif camera == "Front": plotter.view_yz()
    elif camera == "Side": plotter.view_xz()
    elif camera == "Top": plotter.view_xy()
    plotter.reset_camera_clipping_range()
    return plotter

def color(classification):
    text = str(classification).strip().lower()
    if "expected_missing_but_material_present" in text:
        return "#a855f7"
    if "missing_unintentional" in text or "broken" in text:
        return "#ef4444"
    if "missing_intentional" in text:
        return "#f97316"
    if "nominal" in text:
        return "#67e8f9"
    return "#d1d5db"

def macro_figure(table, selected_id):
    fig = go.Figure()
    for classification, group in table.groupby("inventory_classification", dropna=False):
        xs, ys, zs, custom, labels = [], [], [], [], []
        for _, row in group.iterrows():
            start = row[["start_x_vox", "start_y_vox", "start_z_vox"]].to_numpy(float)
            end = start + row[["direction_x_vox", "direction_y_vox", "direction_z_vox"]].to_numpy(float)
            sid = int(row["strut_id"])
            xs += [start[0], end[0], None]
            ys += [start[1], end[1], None]
            zs += [start[2], end[2], None]
            custom += [sid, sid, None]
            labels += [f"Strut {sid} — {classification}", f"Strut {sid} — {classification}", None]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines", name=str(classification),
            line={"color": color(classification), "width": 3}, customdata=custom,
            text=labels, hovertemplate="%{text}<extra></extra>",
        ))
    selected = table[table["strut_id"].astype(int) == int(selected_id)].iloc[0]
    start = selected[["start_x_vox", "start_y_vox", "start_z_vox"]].to_numpy(float)
    end = start + selected[["direction_x_vox", "direction_y_vox", "direction_z_vox"]].to_numpy(float)
    center = selected[["center_x_vox", "center_y_vox", "center_z_vox"]].to_numpy(float)
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode="lines+markers", name="Selected strut",
        line={"color": "#facc15", "width": 10}, marker={"color": "#facc15", "size": 4},
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
        mode="markers", name="Selected endpoints",
        marker={"color": "white", "size": 7}, hoverinfo="skip",
    ))
    fig.update_layout(
        title="Whole Lattice — click a centerline or use the selector", height=780,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        paper_bgcolor="#111827", plot_bgcolor="#111827", font={"color": "#f9fafb"},
        scene={"xaxis_title": "X", "yaxis_title": "Y", "zaxis_title": "Z", "aspectmode": "data", "bgcolor": "#111827"},
        scene_camera={"center": {"x": center[0], "y": center[1], "z": center[2]}},
        legend={"orientation": "h"},
    )
    return fig


def get_context(table, selected_index):
    unit_cell = table.iloc[selected_index]["inventory_unit_cell_ids"]
    if pd.isna(unit_cell) or not str(unit_cell).strip():
        return [selected_index], None
    value = str(unit_cell)
    indices = table.index[table["inventory_unit_cell_ids"].fillna("").astype(str) == value].tolist()
    return indices or [selected_index], value


def get_bounds(table, indices, padding, shape):
    low_cols = ["bbox_z_min_inclusive", "bbox_y_min_inclusive", "bbox_x_min_inclusive"]
    high_cols = ["bbox_z_max_exclusive", "bbox_y_max_exclusive", "bbox_x_max_exclusive"]
    rows = table.loc[indices]
    lower = rows[low_cols].to_numpy(int).min(axis=0) - padding
    upper = rows[high_cols].to_numpy(int).max(axis=0) + padding
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.asarray(shape))
    if np.any(lower >= upper):
        raise ValueError("Selected unit-cell crop does not intersect the scan")
    return lower, upper


def micro_figure(crop, table, indices, selected_index, lower, threshold, title="Selected Unit Cell"):
    fig = go.Figure()
    step = max(1, int(max(crop.shape) / 70))
    points = np.argwhere(crop[::step, ::step, ::step] >= threshold) * step
    if len(points):
        fig.add_trace(go.Scatter3d(
            x=points[:, 2], y=points[:, 1], z=points[:, 0], mode="markers",
            name="CT material", marker={"size": 2, "color": "#d1d5db", "opacity": 0.35},
            hoverinfo="skip",
        ))
    origin = np.asarray(lower, float)
    for index in indices:
        row = table.iloc[index]
        start = row[["start_z_vox", "start_y_vox", "start_x_vox"]].to_numpy(float) - origin
        end = start + row[["direction_z_vox", "direction_y_vox", "direction_x_vox"]].to_numpy(float)
        active = index == selected_index
        fig.add_trace(go.Scatter3d(
            x=[start[2], end[2]], y=[start[1], end[1]], z=[start[0], end[0]], mode="lines",
            name=f"Strut {int(row['strut_id'])}",
            line={"color": "#facc15" if active else color(row["inventory_classification"]), "width": 8 if active else 3},
            hoverinfo="name",
        ))
        if active:
            fig.add_trace(go.Scatter3d(
                x=[start[2], end[2]], y=[start[1], end[1]], z=[start[0], end[0]],
                mode="markers", name="Selected endpoints",
                marker={"color": "white", "size": 8}, hoverinfo="skip",
            ))
    fig.update_layout(
        title=title, height=780, margin={"l": 0, "r": 0, "t": 45, "b": 0},
        paper_bgcolor="#111827", plot_bgcolor="#111827", font={"color": "#f9fafb"},
        scene={"xaxis_title": "X crop", "yaxis_title": "Y crop", "zaxis_title": "Z crop", "aspectmode": "data", "bgcolor": "#111827"},
        showlegend=False,
    )
    return fig



def _normalized_column_name(name):
    return ''.join(ch for ch in str(name).lower() if ch.isalnum())


def _find_column(frame, aliases):
    normalized = {_normalized_column_name(column): column for column in frame.columns}
    for alias in aliases:
        found = normalized.get(_normalized_column_name(alias))
        if found is not None:
            return found
    return None


def _require_strut_id(frame, label):
    column = _find_column(frame, ["strut_id", "strut id", "strut", "id"])
    if column is None:
        raise ValueError(f"{label} must contain a strut ID column (for example strut_id).")
    result = frame.copy()
    result["strut_id"] = pd.to_numeric(result[column], errors="coerce")
    if result["strut_id"].isna().any():
        raise ValueError(f"{label} contains non-numeric or missing strut IDs.")
    result["strut_id"] = result["strut_id"].astype(int)
    return result


def _canonicalize(frame, aliases):
    result = frame.copy()
    for canonical, choices in aliases.items():
        column = _find_column(result, choices)
        if column is not None and canonical not in result.columns:
            result[canonical] = result[column]
    return result


SUMMARY_ALIASES = {
    "primary_defect": ["primary_defect", "primary defect", "defect"],
    "secondary_defects": ["secondary_defects", "secondary defects", "secondary_defect"],
    "stage2_classification": ["stage2_classification", "stage2 classification", "stage2a_classification", "classification"],
    "missing_score": ["missing_score", "missing score", "missing"],
    "broken_score": ["broken_score", "broken score", "broken"],
    "thin_score": ["thin_score", "thin score", "thin"],
    "inflated_score": ["inflated_score", "inflated score", "thick_score", "thick score", "inflated", "thick"],
    "bend_score": ["bend_score", "bend score", "bent_score", "bent score", "bend", "bent"],
    "occupancy": ["occupancy", "material_occupancy", "material fraction", "ct_material_occupancy", "station_material_fraction"],
    "diameter": ["diameter", "equivalent_diameter", "effective_diameter_um", "median_effective_diameter_um"],
    "radius": ["radius", "equivalent_radius", "effective_radius_um", "median_effective_radius_um"],
    "centerline_deviation": ["centerline_deviation", "max_centerline_deviation_um", "centerline deviation"],
    "curvature": ["curvature", "tortuosity", "bend_curvature"],
    "confidence": ["confidence", "confidence_score"],
    "needs_review": ["needs_review", "needs review", "review_required"],
}

STATION_ALIASES = {
    "position_fraction": ["position_fraction", "station_fraction", "position", "fraction"],
    "material_occupancy": ["material_occupancy", "occupancy", "material_fraction", "station_material_fraction"],
    "equivalent_radius": ["equivalent_radius", "radius", "effective_radius_um"],
    "equivalent_diameter": ["equivalent_diameter", "diameter", "effective_diameter_um"],
    "centroid_offset": ["centroid_offset", "centroid_offset_um", "centerline_deviation", "centerline_deviation_um"],
}


st.title("Lattice Structure NDE Dashboard")
st.caption("Uploaded morphology-defect metrics with station-level inspection")

defect_strut_upload = st.file_uploader("Strut summary CSV", type=["csv"], key="defect_by_strut", help="Upload the one-row-per-strut defect summary CSV.")
defect_station_upload = st.file_uploader("Station measurements CSV", type=["csv"], key="defect_by_station", help="Upload the station-level measurements CSV.")

# Persist each upload independently because selecting one Streamlit uploader
# can rerun the script before the other uploader has been selected.
if defect_strut_upload is not None:
    try:
        st.session_state["defect_strut_path"] = str(save_upload(defect_strut_upload))
        st.session_state["defect_upload_error"] = None
    except Exception as exc:
        st.session_state["defect_upload_error"] = str(exc)
if defect_station_upload is not None:
    try:
        st.session_state["defect_station_path"] = str(save_upload(defect_station_upload))
        st.session_state["defect_upload_error"] = None
    except Exception as exc:
        st.session_state["defect_upload_error"] = str(exc)

load_defect_files = st.button("Load defect CSVs")
if st.session_state.get("defect_upload_error"):
    st.error(f"Could not save uploaded files: {st.session_state['defect_upload_error']}")

defect_strut_path = st.session_state.get("defect_strut_path")
defect_station_path = st.session_state.get("defect_station_path")
st.subheader("Defect CSV upload status")
upload_status_cols = st.columns(2)
with upload_status_cols[0]:
    if defect_strut_path and Path(defect_strut_path).exists():
        st.success(f"Strut summary saved: {Path(defect_strut_path).name}")
    else:
        st.warning("Strut summary: not selected")
with upload_status_cols[1]:
    if defect_station_path and Path(defect_station_path).exists():
        st.success(f"Station data saved: {Path(defect_station_path).name}")
    else:
        st.warning("Station data: not selected")
if load_defect_files and (not defect_strut_path or not defect_station_path):
    st.error("Select both defect CSV files before clicking Load defect CSVs.")
if not defect_strut_path or not defect_station_path:
    st.info("Select both CSVs above. Each file is saved when selected; then click Load defect CSVs.")
    st.stop()
if not Path(defect_strut_path).exists() or not Path(defect_station_path).exists():
    st.error("The saved upload files are no longer available. Please select both files again.")
    st.stop()

try:
    defect_by_strut = _canonicalize(_require_strut_id(pd.read_csv(defect_strut_path), "defect_analysis_by_strut.csv"), SUMMARY_ALIASES)
    defect_by_station = _canonicalize(_require_strut_id(pd.read_csv(defect_station_path), "defect_analysis_by_station.csv"), STATION_ALIASES)
    st.success("Both defect CSVs loaded and validated successfully.")
except Exception as exc:
    st.error(f"Could not read the uploaded defect CSVs: {exc}")
    st.stop()

if defect_by_strut["strut_id"].duplicated().any():
    st.error("defect_analysis_by_strut.csv must contain exactly one row per strut ID.")
    st.stop()
if "position_fraction" not in defect_by_station.columns:
    st.error("defect_analysis_by_station.csv needs a position_fraction or station_fraction column.")
    st.stop()
if not set(defect_by_strut["strut_id"]).intersection(set(defect_by_station["strut_id"])):
    st.error("The two uploaded CSVs do not share any strut IDs.")
    st.stop()
st.sidebar.header("Inspection")

# The uploaded summary is the source of truth for defect filtering and the
# selected-strut panel. Centerlines remain available only for optional legacy
# metadata fallbacks; no 3-D renderer is invoked in this UI.
centerlines = read_csv(str(CENTERLINES)).copy()
centerlines["strut_id"] = pd.to_numeric(centerlines.get("strut_id"), errors="coerce")
centerlines = centerlines[centerlines["strut_id"].notna()].copy()
centerlines["strut_id"] = centerlines["strut_id"].astype(int)

summary = defect_by_strut.copy()
summary_lookup = summary.set_index("strut_id")

# Fill optional filter fields from uploaded data first, then from the existing
# centerline table when that field is available there.
def _ensure_summary_field(frame, name, aliases, fallback=None):
    if name in frame.columns:
        return
    column = _find_column(frame, aliases)
    if column is not None:
        frame[name] = frame[column]
        return
    if fallback is not None:
        frame[name] = fallback


_ensure_summary_field(summary, "primary_defect", ["primary_defect", "primary defect", "defect"], "Unknown")
_ensure_summary_field(summary, "stage2_classification", ["stage2_classification", "stage2 classification", "stage2a_classification", "classification"], "Unknown")
_ensure_summary_field(summary, "needs_review", ["needs_review", "needs review", "review_required"], "Not provided")
center_lookup = centerlines.set_index("strut_id")
if "unit_cell_id" not in summary.columns:
    center_unit = _find_column(centerlines, ["unit_cell_id", "unit_cell_ids", "inventory_unit_cell_ids", "unit_cell"])
    if center_unit is not None:
        summary["unit_cell_id"] = summary["strut_id"].map(centerlines.set_index("strut_id")[center_unit])

score_columns = []
for score_name, aliases in {
    "missing_score": ["missing_score", "missing score", "missing"],
    "broken_score": ["broken_score", "broken score", "broken"],
    "thin_score": ["thin_score", "thin score", "thin"],
    "inflated_score": ["inflated_score", "inflated score", "thick_score", "thick score", "inflated", "thick"],
    "bend_score": ["bend_score", "bend score", "bent_score", "bent score", "bend", "bent"],
}.items():
    column = _find_column(summary, aliases)
    if column is not None:
        summary[score_name] = summary[column]
        score_columns.append(score_name)

if score_columns:
    score_values = summary[score_columns].apply(pd.to_numeric, errors="coerce").max(axis=1).fillna(0)
    nonzero = score_values[score_values > 0]
    q1, q2 = nonzero.quantile([0.33, 0.66]).tolist() if len(nonzero) else (1, 2)
    summary["severity"] = np.select([score_values <= q1, score_values <= q2], ["Low", "Medium"], default="High")
else:
    summary["severity"] = "Unscored"

# Preserve the existing defect/severity controls and add the requested
# uploaded-summary filters.
primary_values = sorted(summary["primary_defect"].fillna("Unknown").astype(str).unique())
stage2_values = sorted(summary["stage2_classification"].fillna("Unknown").astype(str).unique())
review_values = sorted(summary["needs_review"].fillna("Not provided").astype(str).unique())
unit_cell_column = _find_column(summary, ["unit_cell_id", "unit_cell_ids", "unit_cell", "inventory_unit_cell_ids"])
unit_cell_values = sorted(summary[unit_cell_column].dropna().astype(str).unique()) if unit_cell_column else []

filter_cols = st.sidebar.columns(2)
strut_search = st.sidebar.text_input("Search strut_id", placeholder="e.g. 1234")
selected_primary = st.sidebar.multiselect("Primary defect", primary_values, default=primary_values)
selected_stage2 = st.sidebar.multiselect("Stage 2 classification", stage2_values, default=stage2_values)
selected_review = st.sidebar.multiselect("Needs review", review_values, default=review_values)
selected_unit_cells = st.sidebar.multiselect("Unit cell ID", unit_cell_values, default=unit_cell_values) if unit_cell_values else []
selected_severity = st.sidebar.multiselect("Severity", sorted(summary["severity"].unique()), default=sorted(summary["severity"].unique()))

filtered = summary[
    summary["primary_defect"].astype(str).isin(selected_primary)
    & summary["stage2_classification"].astype(str).isin(selected_stage2)
    & summary["needs_review"].astype(str).isin(selected_review)
    & summary["severity"].isin(selected_severity)
].copy()
if unit_cell_column and selected_unit_cells:
    filtered = filtered[filtered[unit_cell_column].astype(str).isin(selected_unit_cells)]
if strut_search.strip():
    filtered = filtered[filtered["strut_id"].astype(str).str.contains(strut_search.strip(), regex=False)]
if filtered.empty:
    st.warning("No uploaded struts match the current filters.")
    st.stop()

filtered = filtered.sort_values("strut_id").reset_index(drop=True)
st.subheader("Defect summary")
st.caption("Select a row to make that strut active. The table uses the uploaded one-row-per-strut dataset.")
table_event = st.dataframe(
    filtered.drop(columns=["severity"], errors="ignore"),
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="uploaded_strut_summary",
)
selected_rows = getattr(getattr(table_event, "selection", None), "rows", [])
if selected_rows:
    st.session_state.selected_strut_id = int(filtered.iloc[selected_rows[0]]["strut_id"])

ids = filtered["strut_id"].astype(int).tolist()
current = st.session_state.get("selected_strut_id", ids[0])
if current not in ids:
    current = ids[0]
selected_id = st.sidebar.selectbox("Active strut", ids, index=ids.index(current))
st.session_state.selected_strut_id = int(selected_id)
selected_summary = summary[summary["strut_id"] == int(selected_id)].iloc[0]
selected_stations = defect_by_station[defect_by_station["strut_id"] == int(selected_id)].copy()
station_position_column = _find_column(selected_stations, ["position_fraction", "station_fraction", "position"])
if station_position_column is not None:
    selected_stations["__position_fraction"] = pd.to_numeric(selected_stations[station_position_column], errors="coerce")
    selected_stations = selected_stations.dropna(subset=["__position_fraction"]).sort_values("__position_fraction")


def _value_from_alias(row, aliases):
    column = _find_column(pd.DataFrame([row]), aliases)
    return (column, row.get(column)) if column is not None else (None, np.nan)


def _format_value(value):
    if value is None or pd.isna(value) or value == "":
        return "N/A"
    return str(value)


def _score_value(row, name):
    column, value = _value_from_alias(row, [name, name.replace("_", " ")])
    return _format_value(value) if column else "N/A"


primary_col, primary_value = _value_from_alias(selected_summary, ["primary_defect", "primary defect", "defect"])
secondary_col, secondary_value = _value_from_alias(selected_summary, ["secondary_defects", "secondary defects", "secondary_defect"])
stage2_col, stage2_value = _value_from_alias(selected_summary, ["stage2_classification", "stage2 classification", "stage2a_classification", "classification"])
badge = _format_value(primary_value)
st.subheader(f"Selected strut: {selected_id}")
st.markdown(f"**Defect badge:** `{badge}`")
summary_left, summary_middle, summary_right = st.columns(3)
with summary_left:
    st.write({
        "primary_defect": _format_value(primary_value),
        "secondary_defects": _format_value(secondary_value),
        "stage2_classification": _format_value(stage2_value),
        "confidence": _format_value(_value_from_alias(selected_summary, ["confidence", "confidence_score"])[1]),
        "needs_review": _format_value(_value_from_alias(selected_summary, ["needs_review", "needs review", "review_required"])[1]),
    })
with summary_middle:
    st.write({
        "missing_score": _score_value(selected_summary, "missing_score"),
        "broken_score": _score_value(selected_summary, "broken_score"),
        "thin_score": _score_value(selected_summary, "thin_score"),
        "inflated_score": _score_value(selected_summary, "inflated_score"),
        "bend_score": _score_value(selected_summary, "bend_score"),
    })
with summary_right:
    physical_fields = [
        ("sampled_occupancy", ["sampled_occupancy", "occupancy", "ct_material_occupancy"]),
        ("sampled_mean_intensity", ["sampled_mean_intensity", "mean_intensity"]),
        ("cross-section field", ["median_cross_section_radius_um", "median_cross_section_diameter_um", "equivalent_radius_um", "equivalent_diameter_um", "radius_um", "diameter_um", "radius", "diameter"]),
        ("max_centerline_offset_um", ["max_centerline_offset_um", "max_centerline_deviation_um", "centerline_offset_um"]),
        ("rms_centerline_offset_um", ["rms_centerline_offset_um", "rms_centerline_deviation_um"]),
        ("bend_curvature_um", ["bend_curvature_um", "curvature_um", "curvature", "tortuosity"]),
    ]
    physical_display = {}
    for requested_label, aliases in physical_fields:
        column, value = _value_from_alias(selected_summary, aliases)
        physical_display[column or requested_label] = _format_value(value)
    st.write(physical_display)

configured_low_threshold = None
config_path = ROOT / "part2/morphology/morphology_defect_thresholds.json"
try:
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    configured_low_threshold = config_payload.get("thresholds", {}).get("station_material_fraction")
except Exception:
    configured_low_threshold = None
st.subheader("Station-level defect analysis")
offset_col = None
offsets = pd.Series(dtype=float)
if selected_stations.empty or station_position_column is None:
    st.warning("No station rows with position_fraction were found for the active strut.")
else:
    position = selected_stations["__position_fraction"]
    station_chart_cols = st.columns(2)

    def _explicit_series(frame, aliases):
        column = _find_column(frame, aliases)
        if column is None:
            return None, None
        return column, pd.to_numeric(frame[column], errors="coerce")

    def _add_explicit_lines(fig, frame, metric_aliases):
        threshold_aliases = []
        for alias in metric_aliases:
            threshold_aliases.extend([f"{alias}_threshold", f"{alias}_tolerance", f"{alias}_lower_threshold", f"{alias}_upper_threshold", f"threshold_{alias}"])
        threshold_col = _find_column(frame, threshold_aliases)
        if threshold_col is not None:
            values = pd.to_numeric(frame[threshold_col], errors="coerce").dropna()
            for value in sorted(values.unique()):
                fig.add_hline(y=float(value), line_dash="dash", line_color="orange", annotation_text=f"{threshold_col}={value:g}")

    offset_col, offsets = _explicit_series(selected_stations, ["centroid_offset_um", "centroid_offset", "centerline_offset_um", "centerline_deviation_um"])
    radius_col, radii = _explicit_series(selected_stations, ["equivalent_radius_um", "equivalent_radius", "radius_um"])
    occupancy_col, occupancies = _explicit_series(selected_stations, ["material_occupancy", "occupancy", "material_fraction"])
    intensity_col, intensities = _explicit_series(selected_stations, ["mean_intensity", "sampled_mean_intensity"])

    with station_chart_cols[0]:
        if offset_col:
            fig = go.Figure(go.Scatter(x=position, y=offsets, mode="lines+markers", name=offset_col))
            valid_offsets = offsets.dropna()
            if not valid_offsets.empty:
                max_idx = valid_offsets.idxmax()
                max_row = selected_stations.loc[max_idx]
                fig.add_trace(go.Scatter(x=[max_row["__position_fraction"]], y=[offsets.loc[max_idx]], mode="markers+text", text=["maximum offset"], textposition="top center", marker={"color": "red", "size": 11}, name="Maximum offset"))
            _add_explicit_lines(fig, selected_stations, ["centroid_offset_um", "centerline_offset_um"])
            fig.update_layout(title="Centroid offset vs position_fraction", xaxis_title="position_fraction", yaxis_title=offset_col)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("The station file does not provide centroid_offset_um.")

        if occupancy_col:
            fig = go.Figure(go.Scatter(x=position, y=occupancies, mode="lines+markers", name=occupancy_col))
            low_mask = pd.Series(False, index=selected_stations.index)
            low_threshold_col = _find_column(selected_stations, ["material_occupancy_threshold", "occupancy_threshold", "low_occupancy_threshold"])
            if low_threshold_col:
                low_mask = occupancies < pd.to_numeric(selected_stations[low_threshold_col], errors="coerce")
                _add_explicit_lines(fig, selected_stations, ["material_occupancy", "occupancy"])
            elif configured_low_threshold is not None:
                low_mask = occupancies < float(configured_low_threshold)
                fig.add_hline(y=float(configured_low_threshold), line_dash="dash", line_color="orange", annotation_text=f"configured threshold={float(configured_low_threshold):g}")
            low_station_col = _find_column(selected_stations, ["low_occupancy", "is_low_occupancy", "low_occupancy_flag"])
            if low_station_col:
                low_mask = selected_stations[low_station_col].astype(str).str.lower().isin(["1", "true", "yes", "low"])
            if low_mask.any():
                fig.add_trace(go.Scatter(x=position[low_mask], y=occupancies[low_mask], mode="markers", marker={"color": "red", "size": 9}, name="Low occupancy"))
            fig.update_layout(title="Material occupancy vs position_fraction", xaxis_title="position_fraction", yaxis_title=occupancy_col)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("The station file does not provide material_occupancy.")

    with station_chart_cols[1]:
        if radius_col:
            fig = go.Figure(go.Scatter(x=position, y=radii, mode="lines+markers", name=radius_col))
            _add_explicit_lines(fig, selected_stations, ["equivalent_radius_um", "radius_um"])
            fig.update_layout(title="equivalent_radius_um vs position_fraction", xaxis_title="position_fraction", yaxis_title=radius_col)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("The station file does not provide equivalent_radius_um.")
        if intensity_col:
            fig = go.Figure(go.Scatter(x=position, y=intensities, mode="lines+markers", name=intensity_col))
            _add_explicit_lines(fig, selected_stations, ["mean_intensity"])
            fig.update_layout(title="Mean intensity vs position_fraction", xaxis_title="position_fraction", yaxis_title=intensity_col)
            st.plotly_chart(fig, width="stretch")

with st.expander("Selected station rows"):
    st.dataframe(selected_stations.drop(columns=["__position_fraction"], errors="ignore"), width="stretch", hide_index=True)

st.subheader("Chat")
chat_context = selected_summary.to_dict()
chat_context["station_rows"] = len(selected_stations)
if offset_col if not selected_stations.empty else False:
    valid_offsets = offsets.dropna()
    if not valid_offsets.empty:
        max_idx = valid_offsets.idxmax()
        chat_context["max_offset_position_fraction"] = float(selected_stations.loc[max_idx, "__position_fraction"])
        chat_context["max_offset_value"] = float(valid_offsets.loc[max_idx])
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
question = st.chat_input("Ask about the active strut or uploaded defect summary")
if question:
    question_lower = question.lower()
    st.session_state.chat_messages.append({"role": "user", "content": question})
    if "inflated" in question_lower:
        inflated_mask = pd.Series(False, index=summary.index)
        for column in ["primary_defect", "secondary_defects"]:
            if column in summary.columns:
                inflated_mask = inflated_mask | summary[column].astype(str).str.contains("inflated", case=False, na=False)
        if "inflated_score" in summary.columns:
            inflated_mask = inflated_mask | (pd.to_numeric(summary["inflated_score"], errors="coerce").fillna(0) > 0)
        inflated_ids = summary.loc[inflated_mask, "strut_id"].astype(int).tolist()
        answer = "Suspected inflated struts in the uploaded data: " + (", ".join(map(str, inflated_ids)) if inflated_ids else "none reported.")
    elif "review" in question_lower:
        review_mask = summary["needs_review"].astype(str).str.lower().isin(["true", "1", "yes", "required", "needs_review"])
        review_ids = summary.loc[review_mask, "strut_id"].astype(int).tolist()
        answer = "Struts marked for review by the uploaded data: " + (", ".join(map(str, review_ids)) if review_ids else "none reported.")
    elif "largest" in question_lower or "deviation" in question_lower or "where" in question_lower:
        if chat_context.get("max_offset_position_fraction") is not None:
            answer = f"For strut {selected_id}, the largest uploaded centerline offset is {chat_context['max_offset_value']:g} at position_fraction {chat_context['max_offset_position_fraction']:.4g}."
        else:
            answer = f"The station file does not provide a usable centroid_offset_um series for strut {selected_id}."
    elif "why" in question_lower or "flag" in question_lower:
        answer = f"The uploaded data reports primary_defect={_format_value(primary_value)}, secondary_defects={_format_value(secondary_value)}, stage2_classification={_format_value(stage2_value)}, and needs_review={_format_value(_value_from_alias(selected_summary, ["needs_review", "needs review", "review_required"])[1])}. These are reported uploaded results; no defect is called confirmed unless the uploaded field explicitly says confirmed."
    else:
        answer = f"Active strut {selected_id}: primary_defect={_format_value(primary_value)}, stage2_classification={_format_value(stage2_value)}, station_rows={len(selected_stations)}. Ask why it is flagged, where the largest deviation occurs, which struts are inflated, or which need review."
    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.rerun()