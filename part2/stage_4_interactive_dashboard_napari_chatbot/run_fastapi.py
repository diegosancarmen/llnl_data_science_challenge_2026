"""FastAPI broker for the Napari viewer and the future Next.js dashboard.

The API deliberately keeps the CT volume out of the web process.  Napari owns
the TIFF and this service exposes the precomputed geometry and Stage 3
metadata, which keeps dashboard requests small and predictable.
"""

from __future__ import annotations

import os
import re
import logging
import asyncio
from time import monotonic
import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENTERLINES = REPO_ROOT / "part2/napari_visualizer/napari_centerlines.csv"
DEFAULT_EXPORT = REPO_ROOT / "part2/stage_3_defect_analysis/output/station_export_20260727T193004Z"
DEFAULT_DEFECT_BY_STRUT = DEFAULT_EXPORT / "defect_analysis_by_strut.csv"
DEFAULT_DEFECT_BY_STATION = DEFAULT_EXPORT / "defect_analysis_by_station.csv"
DEFAULT_REVIEW_DECISIONS = REPO_ROOT / "part2/stage_4_interactive_dashboard_napari_chatbot/output/review_decisions.json"
logger = logging.getLogger(__name__)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


def _json_value(value: Any) -> Any:
    """Convert Pandas/NumPy scalar values to JSON-safe Python values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if not isinstance(value, (list, tuple, dict)) and pd.isna(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def _records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [_json_value(row) for row in frame.to_dict(orient="records")]


class DataStore:
    def __init__(self, centerlines: Path, defect_by_strut: Path, defect_by_station: Path):
        self.paths = {
            "centerlines": centerlines,
            "defect_by_strut": defect_by_strut,
            "defect_by_station": defect_by_station,
        }
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing dashboard data file(s): " + ", ".join(missing))

        self.centerlines = pd.read_csv(centerlines)
        self.defect_by_strut = pd.read_csv(defect_by_strut)
        self.defect_by_station = pd.read_csv(defect_by_station)
        for name, frame in (
            ("centerlines", self.centerlines),
            ("defect_by_strut", self.defect_by_strut),
            ("defect_by_station", self.defect_by_station),
        ):
            if "strut_id" not in frame.columns:
                raise ValueError(f"{name} must contain a strut_id column")

        self.centerlines["_strut_key"] = self.centerlines["strut_id"].astype(str)
        self.defect_by_strut["_strut_key"] = self.defect_by_strut["strut_id"].astype(str)
        self.defect_by_station["_strut_key"] = self.defect_by_station["strut_id"].astype(str)
        if self.centerlines["_strut_key"].duplicated().any():
            raise ValueError("centerlines must contain one row per strut_id")
        if self.defect_by_strut["_strut_key"].duplicated().any():
            raise ValueError("defect_by_strut must contain one row per strut_id")
        self.strut_keys = set(self.centerlines["_strut_key"])
        self.centerline_by_key = self.centerlines.set_index("_strut_key", drop=False)
        self.defect_by_key = self.defect_by_strut.set_index("_strut_key", drop=False)
        self.station_indices_by_key = self.defect_by_station.groupby("_strut_key").indices

        # Streamlit needs a complete, filterable defect table but not Napari
        # geometry. Serialize this once at startup rather than rebuilding
        # 18k records for every dashboard request.
        self.dashboard_summary = self.defect_by_strut.drop(columns="_strut_key").copy()
        length_column = next(
            (column for column in ("inventory_length_um", "length_um_from_centerline", "length_um") if column in self.centerlines),
            None,
        )
        if length_column is not None:
            self.dashboard_summary["length_um"] = self.dashboard_summary["strut_id"].astype(str).map(
                self.centerline_by_key[length_column]
            )
        self.dashboard_records = _records(self.dashboard_summary)

    @staticmethod
    def _value(row: pd.Series, *columns: str) -> Any:
        """Return the first non-null value available under the supplied aliases."""
        for column in columns:
            if column in row.index and pd.notna(row[column]):
                return _json_value(row[column])
        return None

    def _compact_strut_record(self, key: str) -> Dict[str, Any]:
        """Merge the useful inventory and defect fields for dashboard clients."""
        geometry = self.centerline_by_key.loc[key]
        defect = self.defect_by_key.loc[key] if key in self.defect_by_key.index else pd.Series(dtype=object)

        def value(*columns: str) -> Any:
            defect_value = self._value(defect, *columns)
            return defect_value if defect_value is not None else self._value(geometry, *columns)

        record = {
            "strut_id": self._value(geometry, "strut_id"),
            "classification": value("stage2_classification", "inventory_classification"),
            "primary_defect": value("primary_defect"),
            "needs_review": value("needs_review"),
            "expected_by_0point5_cad": value("expected_by_0point5_cad", "inventory_expected_by_0point5_cad"),
            "unit_cell_ids": value("unit_cell_ids", "inventory_unit_cell_ids"),
            "junction0_id": value("junction0_id", "inventory_junction0_id"),
            "junction1_id": value("junction1_id", "inventory_junction1_id"),
            "junction0_degree": value("junction0_degree", "inventory_junction0_degree"),
            "junction1_degree": value("junction1_degree", "inventory_junction1_degree"),
            "length_um": value("inventory_length_um", "length_um_from_centerline", "length_um"),
            "center_z_vox": self._value(geometry, "center_z_vox"),
            "center_y_vox": self._value(geometry, "center_y_vox"),
            "center_x_vox": self._value(geometry, "center_x_vox"),
            "start_z_vox": self._value(geometry, "start_z_vox"),
            "start_y_vox": self._value(geometry, "start_y_vox"),
            "start_x_vox": self._value(geometry, "start_x_vox"),
            "end_z_vox": self._value(geometry, "end_z_vox"),
            "end_y_vox": self._value(geometry, "end_y_vox"),
            "end_x_vox": self._value(geometry, "end_x_vox"),
        }
        record["defect_summary"] = (
            {column: _json_value(value) for column, value in defect.items() if column != "_strut_key"}
            if not defect.empty
            else {}
        )
        return record

    def summary(self) -> Dict[str, Any]:
        classification_column = (
            "stage2_classification"
            if "stage2_classification" in self.defect_by_strut.columns
            else "inventory_classification"
        )
        classifications = self.defect_by_strut[classification_column].value_counts().to_dict()
        defects = (
            self.defect_by_strut["primary_defect"].value_counts().to_dict()
            if "primary_defect" in self.defect_by_strut.columns
            else {}
        )
        unit_cell_column = (
            "unit_cell_ids" if "unit_cell_ids" in self.defect_by_strut.columns else "inventory_unit_cell_ids"
        )
        return {
            "strut_count": len(self.centerlines),
            "station_record_count": len(self.defect_by_station),
            "station_count_per_strut": int(self.defect_by_station.groupby("_strut_key").size().mode().iloc[0]),
            "stage2_classification_counts": _json_value(classifications),
            "primary_defect_counts": _json_value(defects),
            "unit_cell_count": int(self.defect_by_strut[unit_cell_column].astype(str).nunique()),
            "data_files": {key: str(path) for key, path in self.paths.items()},
        }

    def require_key(self, strut_id: int | str) -> str:
        key = str(strut_id)
        if key not in self.strut_keys:
            raise HTTPException(status_code=404, detail=f"Unknown strut_id: {strut_id}")
        return key

    def strut_detail(self, strut_id: int | str) -> Dict[str, Any]:
        key = self.require_key(strut_id)
        stations = self.station_frame(key)
        return {
            "strut_id": int(strut_id),
            "strut": self._compact_strut_record(key),
            "stations": _records(stations),
        }

    def station_frame(self, key: str) -> pd.DataFrame:
        indices = self.station_indices_by_key.get(key, [])
        return self.defect_by_station.iloc[indices].drop(columns="_strut_key")


def load_store() -> DataStore:
    return DataStore(
        _env_path("NAPARI_CENTERLINES_CSV", DEFAULT_CENTERLINES),
        _env_path("DEFECT_BY_STRUT_CSV", DEFAULT_DEFECT_BY_STRUT),
        _env_path("DEFECT_BY_STATION_CSV", DEFAULT_DEFECT_BY_STATION),
    )


class GlobalState(BaseModel):
    active_strut_id: Optional[int] = None
    active_strut_ids: List[int] = Field(default_factory=list)
    active_station_id: Optional[int] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


class SelectionRequest(BaseModel):
    strut_ids: List[int] = Field(min_length=1)
    active_strut_id: Optional[int] = None


class ActiveStrutRequest(BaseModel):
    strut_id: int


class ReviewDecisionRequest(BaseModel):
    decision: Literal["nominal", "needs_review", "confirmed_defect"]


class ReviewDecisionStore:
    """Persist final human decisions independently of automated analysis outputs."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._decisions = self._load()

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.is_file():
            return {}
        with self.path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1 or not isinstance(payload.get("decisions"), dict):
            raise ValueError(f"Invalid review decisions file: {self.path}")
        return payload["decisions"]

    def items(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"strut_id": int(strut_id), **record}
                for strut_id, record in sorted(self._decisions.items(), key=lambda item: int(item[0]))
            ]

    def set(self, strut_id: int, decision: str) -> dict[str, Any]:
        record = {"decision": decision, "updated_utc": datetime.now(timezone.utc).isoformat()}
        with self._lock:
            self._decisions[str(strut_id)] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema_version": 1, "decisions": self._decisions}
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp", delete=False
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                temp_path = Path(handle.name)
            temp_path.replace(self.path)
        return {"strut_id": strut_id, **record}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    active_strut_id: Optional[int] = None
    chat_history: List[Dict[str, str]] = Field(default_factory=list, max_length=12)
    use_gemini: bool = False


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send(self, websocket: WebSocket, message: Dict[str, Any]):
        await websocket.send_json(message)

    async def broadcast(self, message: Dict[str, Any]):
        stale = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                stale.append(connection)
        for connection in stale:
            self.disconnect(connection)


app = FastAPI(title="CT Defect Analysis Dashboard Broker", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "DASHBOARD_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = load_store()
review_decisions = ReviewDecisionStore(_env_path("DASHBOARD_REVIEW_DECISIONS_JSON", DEFAULT_REVIEW_DECISIONS))
current_state = GlobalState()
manager = ConnectionManager()
_gemini_cache: Dict[str, tuple[float, str, List[int] | None]] = {}


@app.get("/")
def read_root():
    return {"status": "Central Broker Running", "active_clients": len(manager.active_connections)}


@app.get("/health")
def health():
    return {"status": "ok", "data_loaded": True}


@app.get("/frontend-config")
def frontend_config():
    return {
        "api_status": "ok",
        "websocket_path": "/ws",
        "capabilities": ["summary", "strut_listing", "strut_detail", "stations", "selection", "chat"],
        "summary": store.summary(),
    }


@app.get("/summary")
def get_summary():
    return store.summary()


@app.get("/state")
def get_state():
    return current_state.model_dump() if hasattr(current_state, "model_dump") else current_state.dict()


@app.get("/dashboard/struts")
def dashboard_struts():
    """Return the lightweight, complete defect table used by Streamlit."""
    return {"count": len(store.dashboard_records), "items": store.dashboard_records}


@app.get("/reviews")
def get_reviews():
    items = review_decisions.items()
    return {"count": len(items), "items": items}


@app.post("/reviews/{strut_id}")
def set_review(strut_id: int, request: ReviewDecisionRequest):
    store.require_key(strut_id)
    return review_decisions.set(strut_id, request.decision)


@app.get("/struts")
def list_struts(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    classification: Optional[str] = None,
    primary_defect: Optional[str] = None,
    needs_review: Optional[bool] = None,
    unit_cell_id: Optional[str] = None,
    search: Optional[str] = None,
):
    frame = store.defect_by_strut
    if classification and "stage2_classification" in frame.columns:
        frame = frame[frame["stage2_classification"].astype(str).str.casefold() == classification.casefold()]
    elif classification:
        centerline_keys = set(
            store.centerlines.loc[
                store.centerlines["inventory_classification"].astype(str).str.casefold() == classification.casefold(),
                "_strut_key",
            ]
        )
        frame = frame[frame["_strut_key"].isin(centerline_keys)]
    if primary_defect and "primary_defect" in frame.columns:
        frame = frame[frame["primary_defect"].astype(str).str.casefold() == primary_defect.casefold()]
    if needs_review is not None and "needs_review" in frame.columns:
        review_values = frame["needs_review"].astype(str).str.casefold().isin({"true", "1", "yes"})
        frame = frame[review_values == needs_review]
    if unit_cell_id and "unit_cell_ids" in frame.columns:
        frame = frame[frame["unit_cell_ids"].astype(str).str.split(",").apply(lambda values: unit_cell_id in values)]
    if search:
        frame = frame[frame["_strut_key"].str.contains(search, case=False, regex=False)]
    total = len(frame)
    result_keys = frame.iloc[offset : offset + limit]["_strut_key"].tolist()
    return {
        "offset": offset,
        "limit": limit,
        "total": total,
        "items": [store._compact_strut_record(key) for key in result_keys],
    }


@app.get("/struts/{strut_id}")
def get_strut(strut_id: int):
    return store.strut_detail(strut_id)


@app.get("/struts/{strut_id}/stations")
def get_stations(strut_id: int):
    key = store.require_key(strut_id)
    frame = store.station_frame(key)
    return {"strut_id": strut_id, "count": len(frame), "items": _records(frame)}


async def publish_selection(
    strut_ids: List[int], active_strut_id: Optional[int] = None
) -> Dict[str, Any]:
    if not strut_ids:
        raise HTTPException(status_code=422, detail="At least one strut_id is required")
    selected_ids = list(dict.fromkeys(strut_ids))
    for strut_id in selected_ids:
        store.require_key(strut_id)
    active_id = selected_ids[0] if active_strut_id is None else active_strut_id
    if active_id not in selected_ids:
        raise HTTPException(status_code=422, detail="active_strut_id must be one of strut_ids")

    current_state.active_strut_ids = selected_ids
    current_state.active_strut_id = active_id
    event = {
        "event_type": "STRUTS_SELECTED",
        "data": {
            "strut_ids": current_state.active_strut_ids,
            "active_strut_id": current_state.active_strut_id,
        },
    }
    await manager.broadcast(event)
    return {"status": "success", **event["data"]}


async def publish_active_strut(strut_id: int) -> Dict[str, Any]:
    store.require_key(strut_id)
    if strut_id not in current_state.active_strut_ids:
        raise HTTPException(status_code=422, detail="strut_id must be one of the selected strut IDs")

    current_state.active_strut_id = strut_id
    event = {
        "event_type": "STRUT_ACTIVE_CHANGED",
        "data": {
            "strut_id": current_state.active_strut_id,
            "strut_ids": current_state.active_strut_ids,
        },
    }
    await manager.broadcast(event)
    return {"status": "success", **event["data"]}


@app.post("/select_strut")
async def select_strut(strut_id: int):
    return await publish_selection([strut_id])


@app.post("/select_struts")
async def select_struts(request: SelectionRequest):
    return await publish_selection(request.strut_ids, request.active_strut_id)


@app.post("/select_active_strut")
async def select_active_strut(request: ActiveStrutRequest):
    return await publish_active_strut(request.strut_id)


def _chat_filter(
    primary_defect: Optional[str] = None,
    stage2_classification: Optional[str] = None,
    needs_review: Optional[bool] = None,
    strut_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Return a safely filtered defect table for Gemini's read-only tools."""
    frame = store.defect_by_strut
    if primary_defect:
        frame = frame[frame["primary_defect"].astype(str).str.casefold() == primary_defect.casefold()]
    if stage2_classification:
        frame = frame[frame["stage2_classification"].astype(str).str.casefold() == stage2_classification.casefold()]
    if needs_review is not None:
        flags = frame["needs_review"].astype(str).str.casefold().isin({"true", "1", "yes"})
        frame = frame[flags == needs_review]
    if strut_ids is not None:
        wanted = {int(value) for value in strut_ids}
        frame = frame[frame["strut_id"].astype(int).isin(wanted)]
    return frame


CHAT_METRICS: Dict[str, Dict[str, Any]] = {
    "length": {"column": "length_um", "label": "length", "unit": "µm", "aliases": ("length",)},
    "occupancy": {"column": "sampled_occupancy", "label": "material occupancy", "unit": "%", "percent": True, "aliases": ("occupancy", "material fraction")},
    "median_radius": {"column": "median_cross_section_radius_um", "label": "median radius", "unit": "µm", "aliases": ("median radius", "radius")},
    "median_diameter": {"column": "median_cross_section_radius_um", "label": "median diameter", "unit": "µm", "multiplier": 2, "aliases": ("diameter",)},
    "max_deviation": {"column": "max_centerline_offset_um", "label": "maximum centerline offset", "unit": "µm", "aliases": ("maximum deviation", "max deviation", "largest deviation", "centerline offset", "offset", "deviation")},
    "rms_deviation": {"column": "rms_centerline_offset_um", "label": "RMS centerline offset", "unit": "µm", "aliases": ("rms deviation", "rms offset")},
    "outside_nominal": {"column": "median_outside_nominal_fraction", "label": "median outside-nominal material", "unit": "%", "percent": True, "aliases": ("outside nominal", "excess material")},
    "intensity": {"column": "sampled_mean_intensity", "label": "mean CT intensity", "unit": "a.u.", "aliases": ("intensity",)},
    "confidence": {"column": "confidence", "label": "confidence", "unit": "", "aliases": ("confidence",)},
    "missing_score": {"column": "missing_score", "label": "missing score", "unit": "", "aliases": ("missing score",)},
    "broken_score": {"column": "broken_score", "label": "broken score", "unit": "", "aliases": ("broken score",)},
    "thin_score": {"column": "thin_score", "label": "thin score", "unit": "", "aliases": ("thin score",)},
    "inflated_score": {"column": "inflated_score", "label": "inflated score", "unit": "", "aliases": ("inflated score",)},
    "bend_score": {"column": "bend_score", "label": "bend score", "unit": "", "aliases": ("bend score",)},
}

CHAT_GLOSSARY = {
    "stage 2 classification": "Stage 2 classification describes whether material is present by design, missing by design, missing by accident, or unexpectedly present relative to CAD.",
    "primary defect": "Primary defect is the dominant automated geometry-defect label assigned to a strut. Secondary defects record additional concurrent signals.",
    "needs review": "Needs review is an automated flag indicating that a strut should receive human inspection; it is not a final review decision.",
    "material occupancy": "Material occupancy is the sampled fraction of the expected strut region containing CT material. It is reported as a percentage.",
    "centerline offset": "Centerline offset is the distance between the measured and expected centerline at a station, reported in µm.",
    "outside nominal": "Outside-nominal material is measured material outside the expected strut region; it is reported as a percentage.",
    "station": "A station is a sampled position along a strut. Station records contain local occupancy, equivalent radius, centerline offset, intensity, and outside-nominal material.",
}

DISPLAY_CATEGORY_NAMES = {
    "Missing_Intentional": "Missing by Design",
    "Missing_Unintentional": "Missing by Accident",
}


def _chat_display_category(value: str) -> str:
    return DISPLAY_CATEGORY_NAMES.get(value, value)


def _chat_summary_frame() -> pd.DataFrame:
    """Return the summary table used for deterministic aggregate answers."""
    return store.dashboard_summary.copy()


def _chat_metric_key(lower: str) -> Optional[str]:
    for metric, spec in CHAT_METRICS.items():
        if any(alias in lower for alias in spec["aliases"]):
            return metric
    return None


def _chat_metric_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    spec = CHAT_METRICS[metric]
    if spec["column"] not in frame:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[spec["column"]], errors="coerce")
    return values * spec.get("multiplier", 1)


def _chat_format_metric(value: float, metric: str) -> str:
    spec = CHAT_METRICS[metric]
    if spec.get("percent"):
        value = value * 100 if abs(value) <= 1 else value
    return f"{value:,.2f}{spec['unit'] and ' ' + spec['unit']}"


def _chat_format_optional_metric(value: Any, metric: str) -> str:
    """Render a nullable numeric chat field with the shared two-decimal rule."""
    numeric = pd.to_numeric(value, errors="coerce")
    return "N/A" if pd.isna(numeric) else _chat_format_metric(float(numeric), metric)


def _chat_secondary_defects(value: Any) -> str:
    """Return a human-readable secondary-defect value from CSV-backed data."""
    text = "" if value is None else str(value).strip()
    return "Not Listed" if text.casefold() in {"", "nan", "none", "null", "<na>"} else text.replace(";", ", ")


def _chat_raw_explicit_ids(text: str) -> List[int]:
    """Extract mentioned IDs without mistaking ranking sizes such as 'top 10' for a strut ID."""
    matches = re.findall(r"\b(?:strut(?:s)?|inspect|compare|select)\s+([\d,\sand]+)", text.casefold())
    values = [int(value) for match in matches for value in re.findall(r"\d+", match)]
    return list(dict.fromkeys(values))


def _chat_explicit_ids(text: str) -> List[int]:
    return [value for value in _chat_raw_explicit_ids(text) if str(value) in store.strut_keys]


def _chat_context_ids(message: str, active_strut_id: Optional[int]) -> List[int]:
    explicit = _chat_explicit_ids(message)
    if explicit:
        return explicit
    lower = message.casefold()
    if any(token in lower for token in ("selected struts", "these struts", "those struts", "selection", "compare them")):
        return list(current_state.active_strut_ids)
    if active_strut_id is not None and str(active_strut_id) in store.strut_keys:
        return [int(active_strut_id)]
    return [current_state.active_strut_id] if current_state.active_strut_id is not None else []


def _chat_station_extreme(strut_id: int, lower: str) -> Optional[Dict[str, Any]]:
    detail = store.strut_detail(strut_id)
    stations = pd.DataFrame(detail["stations"])
    choices = (
        (("occupancy", "low material"), "material_occupancy", "lowest", "material occupancy", "%", True),
        (("radius", "diameter"), "equivalent_radius_um", "highest" if any(word in lower for word in ("largest", "highest", "maximum")) else "lowest", "equivalent radius", "µm", False),
        (("outside nominal", "excess material"), "outside_nominal_fraction", "highest", "outside-nominal material", "%", True),
        (("deviation", "offset"), "centroid_offset_um", "highest", "centerline offset", "µm", False),
    )
    for terms, column, direction, label, unit, percent in choices:
        if not any(term in lower for term in terms) or column not in stations:
            continue
        values = pd.to_numeric(stations[column], errors="coerce")
        values = values.dropna()
        if values.empty:
            return None
        index = values.idxmin() if direction == "lowest" else values.idxmax()
        value = float(values.loc[index])
        shown = value * 100 if percent and abs(value) <= 1 else value
        position = pd.to_numeric(stations.loc[index].get("position_fraction"), errors="coerce")
        position_text = f" at position fraction {float(position):.2f} ({float(position) * 100:.2f}% along the strut)" if pd.notna(position) else ""
        return {
            "reply": f"For strut {strut_id}, the {direction} {label} is {shown:,.2f} {unit}{position_text}.",
            "result_type": "station_extreme", "strut_id": strut_id, "metric": column,
            "value": value, "position_fraction": _json_value(position), "references": [f"/struts/{strut_id}/stations"],
        }
    return None


def _gemini_chat_call(question: str, active_strut_id: Optional[int], chat_history: List[Dict[str, str]]) -> tuple[str, List[int] | None]:
    """Run Gemini with fixed CSV-analysis and selection tools only."""
    from google import genai
    from google.genai import types

    selected_ids: List[int] | None = None

    def count_struts(
        primary_defect: str | None = None,
        stage2_classification: str | None = None,
        needs_review: bool | None = None,
    ) -> Dict[str, Any]:
        """Count struts matching optional primary defect, classification, or review filters."""
        frame = _chat_filter(primary_defect, stage2_classification, needs_review)
        return {"count": int(len(frame))}

    def aggregate_struts(
        metric: Literal["median_cross_section_radius_um", "max_centerline_offset_um", "rms_centerline_offset_um", "sampled_occupancy"],
        aggregation: Literal["mean", "min", "max"],
        group_by: Literal["primary_defect", "stage2_classification"] | None = None,
        primary_defect: str | None = None,
        stage2_classification: str | None = None,
    ) -> Dict[str, Any]:
        """Compute an allowed aggregate over the defect-by-strut CSV, optionally grouped by a category."""
        frame = _chat_filter(primary_defect, stage2_classification)
        values = pd.to_numeric(frame[metric], errors="coerce")
        if group_by:
            grouped = frame.assign(__value=values).groupby(group_by, dropna=False)["__value"].agg(aggregation)
            return {"metric": metric, "aggregation": aggregation, "groups": _json_value(grouped.dropna().to_dict())}
        return {"metric": metric, "aggregation": aggregation, "value": _json_value(getattr(values, aggregation)())}

    def get_strut_detail(strut_id: int) -> Dict[str, Any]:
        """Get compact defect summary and station records for one strut ID."""
        return store.strut_detail(strut_id)

    def select_struts(
        primary_defect: str | None = None,
        stage2_classification: str | None = None,
        needs_review: bool | None = None,
    ) -> Dict[str, Any]:
        """Select every strut matching the requested filters in the dashboard and visualizer."""
        nonlocal selected_ids
        frame = _chat_filter(primary_defect, stage2_classification, needs_review)
        selected_ids = frame["strut_id"].astype(int).tolist()
        return {"selected_count": len(selected_ids), "preview_strut_ids": selected_ids[:20], "active_strut_id": selected_ids[0] if selected_ids else None}

    def select_strut_ids(strut_ids: List[int]) -> Dict[str, Any]:
        """Select explicitly named strut IDs after validating them against the loaded dataset."""
        nonlocal selected_ids
        selected_ids = [int(value) for value in strut_ids if str(int(value)) in store.strut_keys]
        return {"selected_count": len(selected_ids), "preview_strut_ids": selected_ids[:20], "active_strut_id": selected_ids[0] if selected_ids else None}

    selected_context: Dict[str, Any] = {}
    if active_strut_id is not None and str(active_strut_id) in store.strut_keys:
        selected_context = store.strut_detail(active_strut_id)
    prompt = (
        "You are a lattice defect-inspection assistant. Use tools for CSV-derived facts and selections. "
        "Never claim a calculation you did not obtain from a tool. Requests to identify, list, show, find, or ask "
        "which struts match a category are selection requests: call select_struts or select_strut_ids and state "
        "the exact selected count. You cannot save final reviews.\n\n"
        f"Active inspection context: {json.dumps(selected_context, default=str)[:16000]}\n"
        f"Recent conversation: {json.dumps(chat_history[-12:], default=str)}\n"
        f"User question: {question}"
    )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
        config=types.GenerateContentConfig(tools=[count_struts, aggregate_struts, get_strut_detail, select_struts, select_strut_ids]),
    )
    return response.text or "Gemini returned an empty response.", selected_ids


@app.post("/chat")
async def chat(request: ChatRequest):
    """HTTP chat endpoint for clients that do not maintain a WebSocket."""
    deterministic = _chat_response(request.message, request.active_strut_id)
    if any(key in deterministic for key in ("count", "total", "strut_ids", "select_strut_ids", "strut")):
        result = deterministic
    elif request.use_gemini and os.getenv("GEMINI_ENABLED", "false").casefold() in {"1", "true", "yes"} and os.getenv("GEMINI_API_KEY"):
        try:
            cache_key = json.dumps([request.message.casefold().strip(), request.active_strut_id, request.chat_history[-4:]], sort_keys=True)
            cached = _gemini_cache.get(cache_key)
            if cached and monotonic() - cached[0] < 300:
                _, reply, selected_ids = cached
            else:
                reply, selected_ids = await asyncio.to_thread(
                    _gemini_chat_call, request.message, request.active_strut_id, request.chat_history[-4:]
                )
                _gemini_cache[cache_key] = (monotonic(), reply, selected_ids)
            result = {"reply": reply, "select_strut_ids": selected_ids, "references": []}
        except Exception as exc:
            logger.exception("Gemini chat failed; using deterministic fallback")
            result = _chat_response(request.message, request.active_strut_id)
            result["reply"] = f"Gemini was unavailable ({exc}); {result['reply']}"
    else:
        result = deterministic
    selected_ids = result.pop("select_strut_ids", None)
    selection = None
    if selected_ids:
        # Return the broker state as well as broadcasting it.  The Streamlit
        # client reruns after a chat message; without this acknowledgement it
        # can republish its stale, single selected ID and erase a chat-driven
        # multi-selection before the visualizer receives it.
        selection = await publish_selection(selected_ids)
    return {
        "message_id": str(uuid4()), **result, "selection": selection,
        "selection_changed": selection is not None,
        "matched_count": result.get("total", result.get("count")),
    }


def _chat_response(message: str, active_strut_id: Optional[int] = None) -> Dict[str, Any]:
    text = message.strip()
    lower = text.casefold()
    normalized_text = re.sub(r"[\s_-]+", "", lower)
    for phrase, explanation in CHAT_GLOSSARY.items():
        if phrase in lower and any(token in lower for token in ("what is", "what does", "explain", "meaning", "define")):
            return {"reply": explanation, "result_type": "glossary", "topic": phrase, "references": ["summary"]}

    ids = _chat_context_ids(text, active_strut_id)
    explicit_ids = _chat_explicit_ids(text)
    raw_explicit_ids = _chat_raw_explicit_ids(text)
    if raw_explicit_ids and not explicit_ids:
        return {
            "reply": f"None of the requested strut IDs are in the loaded dataset: {', '.join(map(str, raw_explicit_ids))}.",
            "result_type": "unknown_strut", "references": [],
        }
    if any(token in lower for token in ("inspect", "detail", "information", "summarize")) and len(ids) > 1:
        frame = _chat_summary_frame()
        selected = frame[frame["strut_id"].astype(int).isin(ids)]
        columns = ["strut_id", "stage2_classification", "primary_defect", "needs_review"]
        metric = _chat_metric_key(lower)
        if metric and CHAT_METRICS[metric]["column"] in selected:
            selected = selected.assign(__metric=_chat_metric_values(selected, metric))
            columns.append("__metric")
        preview = selected[columns].head(20).to_dict(orient="records")
        return {
            "reply": f"Inspection summary for {len(selected)} selected struts. Showing the first {len(preview)} records.",
            "result_type": "multi_strut_summary", "strut_ids": ids, "total": len(selected), "items": _records(pd.DataFrame(preview)),
            "references": [f"/struts/{strut_id}" for strut_id in ids[:20]],
        }

    if "compare" in lower:
        subject_ids = ids
        if not subject_ids:
            return {"reply": "Select or name at least one strut to compare with all nominal struts.", "result_type": "comparison", "references": []}
        frame = _chat_summary_frame()
        subject = frame[frame["strut_id"].astype(int).isin(subject_ids)]
        baseline = frame[frame["stage2_classification"].astype(str) == "Nominal"]
        metric_keys = [_chat_metric_key(lower)] if _chat_metric_key(lower) else ["length", "median_diameter", "occupancy", "max_deviation"]
        comparison = {}
        for metric in metric_keys:
            if metric is None:
                continue
            subject_values = _chat_metric_values(subject, metric).dropna()
            baseline_values = _chat_metric_values(baseline, metric).dropna()
            if not subject_values.empty and not baseline_values.empty:
                comparison[metric] = {
                    "selected_median": float(subject_values.median()),
                    "nominal_median": float(baseline_values.median()),
                }
        if not comparison:
            return {"reply": "The requested comparison metric is not available for the selected struts and nominal baseline.", "result_type": "comparison", "references": ["summary"]}
        clauses = [
            f"{CHAT_METRICS[metric]['label']}: selected median {_chat_format_metric(values['selected_median'], metric)} vs nominal median {_chat_format_metric(values['nominal_median'], metric)}"
            for metric, values in comparison.items()
        ]
        return {
            "reply": f"Comparison for {len(subject)} strut(s) against all nominal struts — " + "; ".join(clauses) + ".",
            "result_type": "comparison", "strut_ids": subject_ids, "comparison": comparison, "references": ["summary"],
        }

    rank_match = re.search(r"\b(top|bottom|highest|lowest|largest|smallest)\s+(\d+)?", lower)
    metric = _chat_metric_key(lower)
    is_rank_request = bool(rank_match and (rank_match.group(1) in {"top", "bottom"} or rank_match.group(2) or "struts" in lower))
    if is_rank_request and metric:
        direction, count_text = rank_match.groups()
        count = min(int(count_text or 10), 100)
        frame = _chat_summary_frame().assign(__metric=_chat_metric_values(_chat_summary_frame(), metric)).dropna(subset=["__metric"])
        ascending = direction in {"bottom", "lowest", "smallest"}
        ranked = frame.sort_values(["__metric", "strut_id"], ascending=[ascending, True]).head(count)
        ranked_ids = ranked["strut_id"].astype(int).tolist()
        return {
            "reply": f"The {direction} {len(ranked_ids)} struts by {CHAT_METRICS[metric]['label']} are {', '.join(map(str, ranked_ids))}.",
            "result_type": "ranking", "metric": metric, "total": len(ranked_ids), "strut_ids": ranked_ids,
            "select_strut_ids": ranked_ids, "references": ["summary"],
        }

    if ids and any(word in lower for word in ("station", "deviation", "offset", "occupancy", "radius", "diameter", "outside nominal", "excess material")):
        station_result = _chat_station_extreme(ids[0], lower)
        if station_result is not None:
            return station_result

    if ids and any(word in lower for word in ("inspect", "detail", "information", "summarize", "this strut", "active strut")):
        detail = store.strut_detail(ids[0])
        strut = detail["strut"]
        defect_summary = strut.get("defect_summary", {})
        length = strut.get("length_um")
        radius = defect_summary.get("median_cross_section_radius_um")
        numeric_radius = pd.to_numeric(radius, errors="coerce")
        diameter = None if pd.isna(numeric_radius) else float(numeric_radius) * 2
        needs_review = strut.get("needs_review")
        review_text = "Yes" if str(needs_review).strip().casefold() in {"true", "1", "yes"} else "No" if str(needs_review).strip().casefold() in {"false", "0", "no"} else str(needs_review)
        return {
            "reply": "\n".join((
                f"### Strut {ids[0]}",
                f"- **Material status:** {strut.get('classification', 'Unknown')}",
                f"- **Primary defect:** {strut.get('primary_defect', 'Unknown')}",
                f"- **Secondary defects:** {_chat_secondary_defects(defect_summary.get('secondary_defects'))}",
                f"- **Needs review:** {review_text}",
                f"- **Length:** {_chat_format_optional_metric(length, 'length')}",
                f"- **Median diameter:** {_chat_format_optional_metric(diameter, 'median_diameter')}",
                f"- **Station measurements:** {len(detail['stations']):,}",
            )),
            "result_type": "strut_summary", "strut": detail,
            "references": [f"struts/{ids[0]}", f"struts/{ids[0]}/stations"],
        }

    ids = explicit_ids
    if "select" in lower and ids:
        valid = [value for value in ids if str(value) in store.strut_keys]
        if valid:
            return {"reply": f"Selecting strut(s): {', '.join(map(str, valid))}.", "select_strut_ids": valid, "references": []}

    if ids and ("station" in lower or "detail" in lower or "inspect" in lower or "information" in lower):
        detail = store.strut_detail(ids[0])
        strut = detail["strut"]
        return {
            "reply": (
                f"Strut {ids[0]} is classified as {strut.get('classification', 'unknown')} "
                f"with primary defect {strut.get('primary_defect', 'unknown')}. "
                f"It has {len(detail['stations'])} station measurements."
            ),
            "strut": detail,
            "references": [f"struts/{ids[0]}", f"struts/{ids[0]}/stations"],
        }

    requested_strut_id = ids[0] if ids else active_strut_id
    if requested_strut_id is not None and any(word in lower for word in ("deviation", "offset", "largest", "where")):
        detail = store.strut_detail(requested_strut_id)
        stations = pd.DataFrame(detail["stations"])
        offset_column = next(
            (
                column
                for column in ("centroid_offset_um", "centroid_offset", "centerline_offset_um")
                if column in stations.columns
            ),
            None,
        )
        if offset_column is not None:
            offsets = pd.to_numeric(stations[offset_column], errors="coerce").dropna()
            if not offsets.empty:
                max_index = offsets.idxmax()
                position = pd.to_numeric(stations.loc[max_index].get("position_fraction"), errors="coerce")
                if pd.notna(position):
                    return {
                        "reply": (
                            f"For strut {requested_strut_id}, the largest uploaded centerline offset is "
                            f"{float(offsets.loc[max_index]):,.2f} µm at position fraction {float(position):.2f}."
                        ),
                        "strut_id": requested_strut_id,
                        "max_offset_value": float(offsets.loc[max_index]),
                        "max_offset_position_fraction": float(position),
                        "references": [f"/struts/{requested_strut_id}/stations"],
                    }
        return {
            "reply": f"The station data does not provide a usable centerline-offset series for strut {requested_strut_id}.",
            "strut_id": requested_strut_id,
            "references": [f"/struts/{requested_strut_id}/stations"],
        }

    list_intent = any(token in lower for token in ("which", "what", "list", "show", "find"))
    all_struts_request = bool(re.search(r"\ball\s+(?:the\s+)?struts?\b", lower))
    direct_selection_intent = any(
        token in lower for token in ("select", "highlight", "visualize")
    )
    collection_intent = list_intent or all_struts_request or direct_selection_intent

    if "review" in lower:
        review_values = store.defect_by_strut["needs_review"].astype(str).str.casefold().isin({"true", "1", "yes"})
        frame = store.defect_by_strut[review_values]
        ids_for_review = [int(value) for value in frame["strut_id"].tolist()]
        selecting = collection_intent and not any(
            phrase in lower for phrase in ("how many", "count", "summary")
        )
        return {
            "reply": (
                f"{len(ids_for_review)} struts are marked as needing review. "
                + (f"Selected all {len(ids_for_review)}." if selecting else "")
            ),
            "select_strut_ids": ids_for_review if selecting else None,
            "strut_ids": ids_for_review if selecting else ids_for_review[:100],
            "total": len(ids_for_review),
            "references": ["/struts?needs_review=true"],
        }

    defect_names = sorted(store.defect_by_strut["primary_defect"].dropna().astype(str), key=str.casefold)
    classification_column = "stage2_classification"
    classification_names = sorted(
        store.defect_by_strut[classification_column].dropna().astype(str), key=str.casefold
    )

    def normalized_label(value: str) -> str:
        """Make display labels match natural spacing and punctuation variants."""
        return re.sub(r"[\s_-]+", "", value.casefold())

    def matching_name(names: List[str]) -> Optional[str]:
        return next(
            (
                name for name in names
                if normalized_label(name) in normalized_text
                or normalized_label(_chat_display_category(name)) in normalized_text
            ),
            None,
        )

    requested_defect = matching_name(defect_names)
    requested_classification = matching_name(classification_names)
    explicitly_defect = "primary defect" in lower
    explicitly_classification = "stage 2" in lower or "stage2" in lower or "classification" in lower

    if explicitly_defect:
        category_field, category_value = "primary_defect", requested_defect
    elif explicitly_classification:
        category_field, category_value = classification_column, requested_classification
    else:
        # A bare value that could be in either column (such as Nominal) means
        # classification; values unique to primary_defect (such as Bent) still
        # resolve naturally to their defect row.
        category_field, category_value = (
            (classification_column, requested_classification)
            if requested_classification
            else ("primary_defect", requested_defect)
        )

    missing_word = bool(re.search(r"\bmissing\b", re.sub(r"[_-]", " ", lower)))
    if missing_word and not category_value:
        missing_counts = {
            name: int((store.defect_by_strut["primary_defect"].astype(str) == name).sum())
            for name in ("Missing_Intentional", "Missing_Unintentional")
        }
        return {
            "reply": (
                "Please specify whether you mean Missing by Design "
                f"({missing_counts['Missing_Intentional']} struts) or "
                f"Missing by Accident ({missing_counts['Missing_Unintentional']} struts)."
            ),
            "missing_subtypes": missing_counts,
            "references": [
                "/struts?primary_defect=Missing_Intentional",
                "/struts?primary_defect=Missing_Unintentional",
            ],
        }

    if category_value and collection_intent:
        frame = store.defect_by_strut[
            store.defect_by_strut[category_field].astype(str).str.casefold() == category_value.casefold()
        ]
        strut_ids = [int(value) for value in frame["strut_id"].tolist()]
        label = "primary defect" if category_field == "primary_defect" else "Stage 2 classification"
        display_value = _chat_display_category(category_value)
        return {
            "reply": (
                f"There are {len(strut_ids)} struts with {label} {display_value}. "
                + f"Selected all {len(strut_ids)} matching struts."
            ),
            "select_strut_ids": strut_ids,
            "strut_ids": strut_ids,
            "total": len(strut_ids),
            "field": category_field,
            "value": category_value,
            "references": [f"/struts?{'primary_defect' if category_field == 'primary_defect' else 'classification'}={category_value}"],
        }

    if category_value and ("how many" in lower or "count" in lower or "summary" in lower):
        frame = store.defect_by_strut[
            store.defect_by_strut[category_field].astype(str).str.casefold() == category_value.casefold()
        ]
        label = "primary defect" if category_field == "primary_defect" else "Stage 2 classification"
        return {
            "reply": f"There are {len(frame)} struts with {label} {_chat_display_category(category_value)}.",
            "result_type": "category_count", "field": category_field, "value": category_value, "count": len(frame),
            "references": [f"/struts?{'primary_defect' if category_field == 'primary_defect' else 'classification'}={category_value}"],
        }

    if requested_defect and ("how many" in lower or "count" in lower or "summary" in lower):
        count = int((store.defect_by_strut["primary_defect"].astype(str) == requested_defect).sum())
        selected_ids = store.defect_by_strut.loc[
            store.defect_by_strut["primary_defect"].astype(str) == requested_defect, "strut_id"
        ].astype(int).tolist() if "select" in lower else None
        return {
            "reply": f"There are {count} struts with primary defect {_chat_display_category(requested_defect)}." + (f" Selected all {count}." if selected_ids else ""),
            "defect": requested_defect,
            "count": count,
            "select_strut_ids": selected_ids,
            "references": [f"/struts?primary_defect={requested_defect}"],
        }

    if "how many" in lower or "count" in lower or "summary" in lower or "defect" in lower:
        summary = store.summary()
        return {
            "reply": (
                f"The dataset contains {summary['strut_count']} struts. Primary defects: "
                + ", ".join(f"{key}: {value}" for key, value in summary["primary_defect_counts"].items())
                + "."
            ),
            "summary": summary,
            "references": ["summary"],
        }

    return {
        "reply": (
            "I can inspect the active or named strut, compare struts with the nominal baseline, "
            "find station extremes, list/select defect or review groups, rank struts by a measurement, "
            "summarize the dataset, and explain inspection fields. Try: 'Inspect strut 42', "
            "'Compare this strut with all nominal struts', or 'Show the top 10 struts by maximum deviation'."
        ),
        "result_type": "help",
        "references": [],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        state = get_state()
        await manager.send(websocket, {"event_type": "INIT_STATE", "data": state})
        while True:
            try:
                incoming = await websocket.receive_json()
                if not isinstance(incoming, dict):
                    raise ValueError("WebSocket event must be a JSON object")
                event_type = incoming.get("event_type", incoming.get("type", "")).upper()
                data = incoming.get("data", incoming)
                if not isinstance(data, dict):
                    raise ValueError("WebSocket event data must be a JSON object")
                request_id = str(data.get("request_id", "")) or None
                if event_type in {"CHAT_REQUEST", "CHAT"}:
                    request = ChatRequest(message=str(data.get("message", "")))
                    result = _chat_response(request.message)
                    selected_ids = result.pop("select_strut_ids", None)
                    if selected_ids:
                        await publish_selection(selected_ids)
                    await manager.send(
                        websocket,
                        {
                            "event_type": "CHAT_RESPONSE",
                            "data": {"message_id": str(uuid4()), "request_id": request_id, **result},
                        },
                    )
                elif event_type in {"STRUT_SELECTED", "STRUTS_SELECTED"}:
                    requested = data.get("strut_ids", [data.get("strut_id")])
                    active = data.get("active_strut_id")
                    await publish_selection(
                        [int(value) for value in requested if value is not None],
                        int(active) if active is not None else None,
                    )
                elif event_type in {"STRUT_ACTIVE_CHANGED", "ACTIVE_STRUT_CHANGED"}:
                    await publish_active_strut(int(data.get("strut_id")))
                else:
                    raise ValueError(f"Unknown event type: {event_type or '(empty)'}")
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.exception("WebSocket event failed from %s: %s", websocket.client, exc)
                try:
                    await manager.send(
                        websocket,
                        {"event_type": "ERROR", "data": {"message": f"Unable to process request: {exc}"}},
                    )
                except Exception:
                    logger.info("WebSocket error response could not be delivered to %s", websocket.client)
                    raise WebSocketDisconnect(code=1011)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
    except Exception:
        logger.exception("Unhandled WebSocket failure for %s", websocket.client)
    finally:
        manager.disconnect(websocket)
