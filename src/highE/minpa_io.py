"""Read-only I/O for Tianwen-1 MINPA ``ori`` MAT and public ``.2B`` files."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np


@dataclass(frozen=True)
class MinpaRecord:
    time_unix_s: float
    mode: int
    ion_dpf: np.ndarray
    quality_native: int = 0
    start_stop_counts: np.ndarray = field(default_factory=lambda: np.full(4, np.nan))
    high_voltage_v: np.ndarray = field(default_factory=lambda: np.full(5, np.nan))
    instrument_solar_angles_deg: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    source_path: str = ""


@dataclass(frozen=True)
class Field:
    name: str
    location: int
    length: int
    unit: str = ""


@dataclass(frozen=True)
class ScienceGroup:
    location: int
    repetitions: int
    field_length: int


@dataclass(frozen=True)
class MinpaLabel:
    fields: dict[str, Field]
    ion_group: ScienceGroup
    record_bytes: int


def mode_from_filename(path: str | Path) -> int:
    match = re.search(r"MINPA-MOD(\d+)-", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot infer MINPA mode from {Path(path).name}")
    mode = int(match.group(1))
    if not 1 <= mode <= 12:
        raise ValueError(f"Unsupported MINPA mode {mode}")
    return mode


def parse_utc(text: str) -> float:
    value = text.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _strip_tag(name: str) -> str:
    return name.split("}", 1)[-1]


def parse_label(path: str | Path) -> MinpaLabel:
    root = ET.parse(Path(path)).getroot()
    fields: dict[str, Field] = {}
    record_bytes = 0
    ion_group: ScienceGroup | None = None
    for elem in root.iter():
        if _strip_tag(elem.tag) == "record_length":
            record_bytes = int(float((elem.text or "0").strip()))
            break
    for elem in root.iter():
        if _strip_tag(elem.tag) != "Field_Character":
            continue
        values = {_strip_tag(c.tag): (c.text or "").strip() for c in elem}
        if {"name", "field_location", "field_length"} <= values.keys():
            name = values["name"]
            fields[name] = Field(name, int(float(values["field_location"])), int(float(values["field_length"])), values.get("unit", ""))
    for group in root.iter():
        if _strip_tag(group.tag) != "Group_Field_Character":
            continue
        name, location, repetitions, width = "", None, None, None
        for child in group:
            tag = _strip_tag(child.tag)
            if tag == "name":
                name = (child.text or "").strip()
            elif tag == "group_location":
                location = int(float((child.text or "0").strip()))
            elif tag == "repetitions":
                repetitions = int(float((child.text or "0").strip()))
            elif tag == "Field_Character":
                for grandchild in child:
                    if _strip_tag(grandchild.tag) == "field_length":
                        width = int(float((grandchild.text or "0").strip()))
        if name == "Ion_Science_Data" and None not in (location, repetitions, width):
            ion_group = ScienceGroup(int(location), int(repetitions), int(width))
    if ion_group is None:
        raise ValueError(f"Ion_Science_Data is absent from {path}")
    return MinpaLabel(fields, ion_group, record_bytes)


def label_for_data_file(path: str | Path) -> Path:
    raw = Path(path)
    candidates = [raw.parent / "xml" / raw.with_suffix(".xml").name, raw.with_suffix(raw.suffix + "L"), raw.with_suffix(".xml")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No XML/.2BL label found for {raw}")


def _read_text(line: bytes, field: Field) -> str:
    start = field.location - 1
    return line[start : start + field.length].decode("ascii", errors="ignore").strip()


def _read_float(line: bytes, field: Field | None) -> float:
    if field is None:
        return math.nan
    text = _read_text(line, field)
    try:
        return float(text) if text and text != "NUL" else math.nan
    except ValueError:
        return math.nan


def read_public_records(path: str | Path, start_s: float = -np.inf, end_s: float = np.inf) -> list[MinpaRecord]:
    """Read the released fixed-width ASCII product without modifying it."""
    source = Path(path)
    mode = mode_from_filename(source)
    label = parse_label(label_for_data_file(source))
    f = label.fields
    rows: list[MinpaRecord] = []
    with source.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                time_s = parse_utc(_read_text(line, f["UTC"]))
            except (KeyError, ValueError):
                continue
            if not start_s <= time_s < end_s:
                continue
            g = label.ion_group
            begin = g.location - 1
            stop = begin + g.repetitions * (g.field_length + 1)
            dpf = np.fromstring(line[begin:stop].decode("ascii", errors="ignore"), sep=" ")
            if dpf.size != g.repetitions:
                raise ValueError(f"Expected {g.repetitions} ion values in {source.name}, got {dpf.size}")
            quality = _read_float(line, f.get("Quality"))
            rows.append(MinpaRecord(
                time_unix_s=time_s,
                mode=mode,
                ion_dpf=dpf.astype(float, copy=False),
                quality_native=int(quality) if np.isfinite(quality) else 0,
                start_stop_counts=np.array([_read_float(line, f.get(name)) for name in ("Ion_StartA_Count", "Ion_StartB_Count", "Ion_StopA_Count", "Ion_StopB_Count")]),
                high_voltage_v=np.array([_read_float(line, f.get(name)) for name in ("Accelerated_High_Pressure_Detection", "Upper_Deflection_Plate_High_Pressure_Detection", "Lower_Deflection_Plate_High_Pressure_Detection", "Ion_Track_Adjustment_High_Pressure_Detection", "Electrostatic_Analyzer_High_Voltage_Detection")]),
                instrument_solar_angles_deg=np.array([_read_float(line, f.get("Instrument_Solar_Incident_Angle")), _read_float(line, f.get("Instrument_Solar_Azimuth_Angle"))]),
                source_path=str(source),
            ))
    return rows


def _decode_char(handle: h5py.File, value: object) -> str:
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    arr = np.asarray(handle[value]) if isinstance(value, h5py.Reference) else np.asarray(value)
    if arr.dtype.kind in "ui" and np.all(arr < 0x110000):
        return "".join(chr(int(v)) for v in arr.reshape(-1, order="F") if int(v))
    if arr.dtype.kind == "S":
        return b"".join(arr.reshape(-1, order="F")).decode(errors="ignore")
    return str(arr.squeeze())


def _matlab_values(handle: h5py.File, entry: h5py.Reference) -> list[np.ndarray]:
    dataset = handle[entry]["t"]
    arr = np.asarray(dataset)
    if arr.dtype == object:
        return [np.asarray(handle[ref]) for ref in arr.reshape(-1, order="F")]
    if arr.ndim == 2 and arr.shape[1] > 1:
        return [arr[:, i] for i in range(arr.shape[1])]
    return [arr.reshape(-1)]


def _scalar(values: list[np.ndarray], index: int) -> float:
    if index >= len(values):
        return math.nan
    arr = np.asarray(values[index], dtype=float).reshape(-1)
    return float(arr[0]) if arr.size else math.nan


def read_ori_records(path: str | Path, start_s: float = -np.inf, end_s: float = np.inf) -> list[MinpaRecord]:
    """Read local MATLAB ``ori`` records, including native diagnostics.

    The 1-based field positions follow the accompanying MINPA .2BL label:
    quality=37, START/STOP=38:41, ion science=42, HV=22:26, and instrument
    solar incidence/azimuth=32:33.  These four START/STOP values are monitor
    totals, not independent background measurements.
    """
    source = Path(path)
    mode = mode_from_filename(source)
    rows: list[MinpaRecord] = []
    with h5py.File(source, "r") as handle:
        group = handle["tw1_MINPA_data"]
        refs = group["value"][:, 0]
        fields = {i + 1: _matlab_values(handle, ref) for i, ref in enumerate(refs) if i + 1 in {1,22,23,24,25,26,32,33,37,38,39,40,41,42}}
        n = min(len(fields[1]), len(fields[42]))
        for i in range(n):
            try:
                time_s = parse_utc(_decode_char(handle, fields[1][i]))
            except ValueError:
                continue
            if not start_s <= time_s < end_s:
                continue
            quality = _scalar(fields[37], i)
            rows.append(MinpaRecord(
                time_unix_s=time_s,
                mode=mode,
                ion_dpf=np.asarray(fields[42][i], dtype=float).reshape(-1),
                quality_native=int(quality) if np.isfinite(quality) else 0,
                start_stop_counts=np.array([_scalar(fields[j], i) for j in (38,39,40,41)]),
                high_voltage_v=np.array([_scalar(fields[j], i) for j in (22,23,24,25,26)]),
                instrument_solar_angles_deg=np.array([_scalar(fields[j], i) for j in (32,33)]),
                source_path=str(source),
            ))
    return rows


def read_records(path: str | Path, start_s: float = -np.inf, end_s: float = np.inf) -> list[MinpaRecord]:
    source = Path(path)
    if source.suffix.lower() == ".mat":
        return read_ori_records(source, start_s, end_s)
    if source.suffix.upper() == ".2B":
        return read_public_records(source, start_s, end_s)
    raise ValueError(f"Unsupported MINPA file type: {source}")

