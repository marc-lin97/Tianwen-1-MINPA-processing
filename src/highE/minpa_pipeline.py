"""High-level, auditable MINPA record-to-product workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .coordinates import tw1_orbiter_to_mso_matrix
from .minpa_io import MinpaRecord
from .minpa_modes import minpa_to_body_direction, mode_layout, split_raw_record
from .minpa_processing import (
    MomentResult,
    VelocitySamples,
    compute_moments,
    finite_velocity_cell_samples,
    species_spectrum,
)
from .minpa_quality import evaluate_quality, reject_for_background, reject_for_moments


@dataclass(frozen=True)
class AttitudeSeries:
    time_unix_s: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    yaw_deg: np.ndarray
    spacecraft_velocity_mso_km_s: np.ndarray

    def nearest(self, time_s: float, tolerance_s: float = 5.0) -> tuple[tuple[float,float,float] | None, np.ndarray | None, float]:
        time = np.asarray(self.time_unix_s, dtype=float)
        if time.size == 0:
            return None, None, np.nan
        pos = int(np.searchsorted(time, time_s))
        choices = [i for i in (pos-1,pos,pos+1) if 0 <= i < time.size]
        index = min(choices, key=lambda i: abs(time[i] - time_s))
        dt = abs(float(time[index] - time_s))
        angles = (float(self.roll_deg[index]), float(self.pitch_deg[index]), float(self.yaw_deg[index]))
        if dt > tolerance_s or not np.all(np.isfinite(angles)):
            return None, None, dt
        velocity = np.asarray(self.spacecraft_velocity_mso_km_s[index], dtype=float)
        return angles, velocity if np.all(np.isfinite(velocity)) else None, dt


@dataclass(frozen=True)
class MinpaProductSeries:
    mode: int
    species: str
    time_unix_s: np.ndarray
    energy_eV: np.ndarray
    dpf: np.ndarray
    deflux: np.ndarray
    density_cm3: np.ndarray
    velocity_minpa_km_s: np.ndarray
    velocity_body_km_s: np.ndarray
    velocity_mso_km_s: np.ndarray
    temperature_eV: np.ndarray
    native_quality: np.ndarray
    project_quality: np.ndarray
    accepted: np.ndarray
    status: np.ndarray
    attitude_match_dt_s: np.ndarray
    source_path: np.ndarray


def save_product_series(series: MinpaProductSeries, output_stem: str | Path) -> tuple[Path, Path]:
    """Save compact NPZ arrays plus a human-readable per-record CSV."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    npz_path, csv_path = stem.with_suffix(".npz"), stem.with_suffix(".csv")
    np.savez_compressed(
        npz_path,
        mode=np.int16(series.mode), species=series.species,
        time_unix_s=series.time_unix_s, energy_eV=series.energy_eV,
        dpf=series.dpf, deflux=series.deflux,
        density_cm3=series.density_cm3,
        velocity_minpa_km_s=series.velocity_minpa_km_s,
        velocity_body_km_s=series.velocity_body_km_s,
        velocity_mso_km_s=series.velocity_mso_km_s,
        temperature_eV=series.temperature_eV,
        native_quality=series.native_quality, project_quality=series.project_quality,
        accepted=series.accepted, status=series.status.astype(str),
        attitude_match_dt_s=series.attitude_match_dt_s, source_path=series.source_path.astype(str),
    )
    frame = pd.DataFrame({
        "time_utc": pd.to_datetime(series.time_unix_s, unit="s", utc=True),
        "time_unix_s": series.time_unix_s,
        "mode": series.mode,
        "species": series.species,
        "density_cm3": series.density_cm3,
        "v_minpa_x_km_s": series.velocity_minpa_km_s[:,0],
        "v_minpa_y_km_s": series.velocity_minpa_km_s[:,1],
        "v_minpa_z_km_s": series.velocity_minpa_km_s[:,2],
        "v_body_x_km_s": series.velocity_body_km_s[:,0],
        "v_body_y_km_s": series.velocity_body_km_s[:,1],
        "v_body_z_km_s": series.velocity_body_km_s[:,2],
        "v_mso_x_km_s": series.velocity_mso_km_s[:,0],
        "v_mso_y_km_s": series.velocity_mso_km_s[:,1],
        "v_mso_z_km_s": series.velocity_mso_km_s[:,2],
        "temperature_eV": series.temperature_eV,
        "native_quality": series.native_quality,
        "project_quality": series.project_quality,
        "accepted": series.accepted,
        "status": series.status,
        "attitude_match_dt_s": series.attitude_match_dt_s,
        "source_path": series.source_path,
    })
    frame.to_csv(csv_path,index=False)
    return npz_path,csv_path


def process_records(
    records: list[MinpaRecord],
    species: str,
    *,
    attitude: AttitudeSeries | None = None,
    energy_min_eV: float = 0.0,
    energy_max_eV: float = np.inf,
    quality_policy: Literal["none", "moments", "background"] = "moments",
    reject_high_channel: bool = False,
    add_spacecraft_velocity: bool = False,
) -> MinpaProductSeries:
    """Convert records from a single mode to spectra and moments.

    Rejected records remain in the spectrogram and carry flags, while their
    moment fields are masked.  This makes the effect of quality selection
    visible instead of silently dropping data.
    """
    if not records:
        raise ValueError("records is empty")
    modes = {record.mode for record in records}
    if len(modes) != 1:
        raise ValueError(f"process_records requires one mode at a time, got {sorted(modes)}")
    mode = modes.pop()
    layout = mode_layout(mode)
    split: list[tuple[MinpaRecord,float,np.ndarray]] = []
    for record in records:
        for time_s, values in split_raw_record(mode, record.time_unix_s, record.ion_dpf):
            split.append((record,time_s,values))
    split.sort(key=lambda item: item[1])
    n = len(split)
    dpf = np.full((n,layout.energy_eV.size), np.nan)
    deff = np.full_like(dpf, np.nan)
    density = np.full(n,np.nan)
    v_minpa = np.full((n,3),np.nan)
    v_body = np.full((n,3),np.nan)
    v_mso = np.full((n,3),np.nan)
    temp = np.full(n,np.nan)
    native = np.zeros(n,dtype=np.uint32)
    project = np.zeros(n,dtype=np.uint32)
    accepted = np.ones(n,dtype=bool)
    status = np.full(n,"",dtype=object)
    attitude_dt = np.full(n,np.nan)
    source = np.full(n,"",dtype=object)
    times = np.array([item[1] for item in split],dtype=float)
    for i,(record,time_s,values) in enumerate(split):
        spectrum = species_spectrum(values,mode,species)
        q = evaluate_quality(values,mode,species)
        dpf[i], deff[i] = spectrum.dpf, spectrum.deflux
        native[i], project[i], source[i] = np.uint32(record.quality_native), q.flag, record.source_path
        if quality_policy == "moments":
            accepted[i] = not reject_for_moments(int(q.flag), reject_high_channel=reject_high_channel)
        elif quality_policy == "background":
            accepted[i] = not reject_for_background(int(q.flag)) and record.quality_native == 0
        elif quality_policy != "none":
            raise ValueError(f"Unknown quality_policy {quality_policy}")
        angles, sc_velocity, dt = (attitude.nearest(time_s) if attitude is not None else (None,None,np.nan))
        attitude_dt[i] = dt
        moment: MomentResult = compute_moments(
            values,mode,species,
            energy_min_eV=energy_min_eV,energy_max_eV=energy_max_eV,
            attitude_roll_pitch_yaw_deg=angles,
            spacecraft_velocity_mso_km_s=sc_velocity,
            add_spacecraft_velocity=add_spacecraft_velocity,
        )
        status[i] = moment.status if accepted[i] else "rejected_by_quality"
        if accepted[i]:
            density[i],v_minpa[i],v_body[i],v_mso[i],temp[i] = moment.density_cm3,moment.velocity_minpa_km_s,moment.velocity_body_km_s,moment.velocity_mso_km_s,moment.scalar_temperature_eV
    return MinpaProductSeries(mode,species,times,layout.energy_eV.copy(),dpf,deff,density,v_minpa,v_body,v_mso,temp,native,project,accepted,status,attitude_dt,source)


def interval_velocity_samples(
    records: list[MinpaRecord], species: str, start_s: float, end_s: float,
    *, energy_min_eV: float = 0.0, energy_max_eV: float = np.inf,
    quality_policy: Literal["none", "moments", "background"] = "moments",
    reject_high_channel: bool = False,
    attitude: AttitudeSeries | None = None,
    frame: Literal["MINPA", "BODY", "MSO"] = "MINPA",
    angular_samples: int = 3,
    energy_samples: int = 3,
) -> VelocitySamples:
    """Average finite-cell VDF samples, rotating every record before aggregation.

    The transform follows the latest validated chain for every record:
    ``MINPA payload [-V2,+V3,-V1] -> spacecraft body -> MSO``.
    Geometric support is rotated and retained separately from positive signal.
    """
    selected: list[VelocitySamples] = []
    for record in records:
        for time_s,values in split_raw_record(record.mode,record.time_unix_s,record.ion_dpf):
            if start_s <= time_s < end_s:
                q = evaluate_quality(values, record.mode, species)
                if quality_policy == "moments" and reject_for_moments(int(q.flag), reject_high_channel=reject_high_channel):
                    continue
                if quality_policy == "background" and (reject_for_background(int(q.flag)) or record.quality_native != 0):
                    continue
                if quality_policy not in ("none", "moments", "background"):
                    raise ValueError(f"Unknown quality_policy {quality_policy}")
                sample = finite_velocity_cell_samples(
                    values,
                    record.mode,
                    species,
                    energy_min_eV,
                    energy_max_eV,
                    angular_samples=angular_samples,
                    energy_samples=energy_samples,
                )
                signal_velocity = sample.velocity_minpa_km_s
                coverage = sample.coverage_velocity_minpa_km_s
                if frame in ("BODY", "MSO"):
                    signal_velocity = minpa_to_body_direction(signal_velocity)
                    if coverage.size:
                        coverage = minpa_to_body_direction(coverage)
                if frame == "MSO":
                    angles,_,_ = attitude.nearest(time_s) if attitude is not None else (None,None,np.nan)
                    if angles is None:
                        continue
                    roll,pitch,yaw = angles
                    rotation = tw1_orbiter_to_mso_matrix(roll,pitch,yaw)
                    signal_velocity = signal_velocity @ rotation.T
                    if coverage.size:
                        coverage = coverage @ rotation.T
                if frame not in ("MINPA", "BODY", "MSO"):
                    raise ValueError(f"Unknown frame {frame}")
                selected.append(VelocitySamples(
                    signal_velocity,
                    sample.density_weight_cm3,
                    sample.energy_eV,
                    sample.mass_amu,
                    coverage_velocity_minpa_km_s=coverage,
                    angular_samples_per_dimension=angular_samples,
                    energy_samples_per_cell=energy_samples,
                ))
    if not selected:
        raise ValueError("No direction-resolved records in the requested interval")
    n = len(selected)
    return VelocitySamples(
        np.concatenate([item.velocity_minpa_km_s for item in selected]),
        np.concatenate([item.density_weight_cm3 / n for item in selected]),
        np.concatenate([item.energy_eV for item in selected]),
        np.concatenate([item.mass_amu for item in selected]),
        coverage_velocity_minpa_km_s=np.concatenate(
            [item.coverage_velocity_minpa_km_s for item in selected if item.coverage_velocity_minpa_km_s.size],
            axis=0,
        ) if any(item.coverage_velocity_minpa_km_s.size for item in selected) else np.empty((0, 3)),
        record_count=n,
        angular_samples_per_dimension=angular_samples,
        energy_samples_per_cell=energy_samples,
    )
