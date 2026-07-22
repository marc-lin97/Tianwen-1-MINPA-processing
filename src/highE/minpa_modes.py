"""Calibrated Tianwen-1 MINPA mode layouts and angular geometry.

The tables are transcribed from the mission L2 processing chain used by this
project.  Ion science values are differential particle flux (DPF) in
``1 / (s cm2 sr eV)``.  Array order is energy, pitch-major azimuth, mass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


E1 = np.array([2.81,3.548928,4.482167,5.660813,7.149401,9.029433,11.40385,14.40264,18.19001,22.97332,29.01447,36.64422,46.28032,58.45036,73.82067,93.23282,117.7497,148.7135,187.8198,237.2095,299.587,378.3675,477.8644,603.5253,762.2305,962.6694,1215.816,1535.532,1939.321,2449.292,3093.366,3906.809,4934.157,6231.661,7870.361,9939.98,12553.83,15855.03,20024.33,25290.0])
E3 = np.array([2.81,3.246924,3.751784,4.335145,5.009211,5.788088,6.688071,7.727991,8.929607,10.31806,11.9224,13.77621,15.91825,18.39336,21.25332,24.55798,28.37647,32.7887,37.88697,43.77797,50.58496,58.45036,67.53873,78.04025,90.17464,104.1958,120.3971,139.1175,160.7487,185.7433,214.6243,247.996,286.5566,331.113,382.5974,442.087,510.8266,590.2544,682.0324,788.0808,910.6185,1052.21,1215.816,1404.862,1623.303,1875.708,2167.36,2504.36,2893.76,3343.707,3863.617,4464.366,5158.525,5960.618,6887.428,7958.346,9195.78,10625.62,12277.79,14186.84,16392.74,18941.63,21886.84,25290.0])
E6 = np.array([2.81,3.410666,4.139729,5.024638,6.098705,7.402364,8.984693,10.90526,13.23637,16.06578,19.5,23.66831,28.72765,34.86848,42.32196,51.3687,62.34928,75.67706,91.85379,111.4885,135.3202,164.2463,199.3556,241.9698,293.6933,356.4731,432.6728,525.161,637.4194,773.6741,939.0547,1139.787,1383.428,1679.149,2038.084,2473.745,3002.533,3644.355,4423.372,5368.912,6516.57,7909.553,9600.298,11652.46,14143.29,17166.56,20836.08,25290.0])
E9 = np.array([44.96,49.67818,54.89149,60.65189,67.0168,74.04965,81.82054,90.40692,99.89437,110.3775,121.9606,134.7594,148.9013,164.5272,181.793,200.8706,221.9503,245.2421,270.9783,299.4152,330.8363,365.5548,403.9167,446.3044,493.1403,544.8912,602.073,665.2555,735.0686,812.2079,897.4423,991.6213,1095.684,1210.667,1337.716,1478.098,1633.212,1804.604,1993.982,2203.234,2434.445,2689.919,2972.204,3284.112,3628.752,4009.559,4430.329,4895.255,5408.971,5976.597,6603.791,7296.804,8062.543,8908.639,9843.526,10876.52,12017.92,13279.1,14672.63,16212.4,17913.76,19793.66,21870.84,24166.0])
E12 = np.array([2.81,3.267539,3.799578,4.418245,5.137647,5.974187,6.946936,8.078073,9.393388,10.92287,12.70139,14.7695,17.17435,19.97077,23.22251,27.00373,31.40062,36.51344,42.45875,49.37211,57.41115,66.75914,77.62922,90.26922,104.9673,122.0587,141.9329,165.0432,191.9164,223.1653,259.5023,301.7558,350.8893,408.023,474.4595,551.7135,641.5465,746.0065,867.4753,1008.722,1172.968,1363.957,1586.043,1844.292,2144.589,2493.783,2899.834,3372.0])

M8 = np.array([1,2,4,16,38,32,44,65], dtype=float)
M16 = np.array([1,2,3,4,6,8,10,12,14,16,18,22,28,32,44,69], dtype=float)
M12 = np.array([0.76,1.25,1.75,2.25,2.75,3.25,3.75,4.26,5.66,6.34,7.65,8.35,9.62,10.38,11.61,12.39,13.55,14.47,15.56,16.45,17.45,18.58,20.98,23.04,26.97,28.92,30.94,33.17,40.07,48.12,64.68,68.95])


@dataclass(frozen=True)
class ModeLayout:
    mode: int
    energy_eV: np.ndarray
    mass_amu: np.ndarray
    pitch_edges_deg: np.ndarray
    azimuth_count: int
    raw_subrecords: int = 1
    directional_velocity_available: bool = True

    @property
    def pitch_count(self) -> int:
        return self.pitch_edges_deg.size - 1

    @property
    def angle_count(self) -> int:
        return self.pitch_count * self.azimuth_count

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.energy_eV.size, self.angle_count, self.mass_amu.size)

    @property
    def values_per_subrecord(self) -> int:
        return int(np.prod(self.shape))


def mode_layout(mode: int) -> ModeLayout:
    """Return the calibrated layout for MINPA modes 1--12.

    Mode 13 is intentionally rejected because it is an engineering/nonstandard
    layout in the local processing chain.
    """
    if mode in (1, 2):
        energy = E1
    elif mode in (3, 4, 5, 7, 8):
        energy = E3
    elif mode == 6:
        energy = E6
    elif mode in (9, 10, 11):
        energy = E9
    elif mode == 12:
        energy = E12
    else:
        raise ValueError(f"Unsupported MINPA mode {mode}; supported science modes are 1--12")

    if 1 <= mode <= 6:
        masses, pitch, az = M8, np.linspace(0.0, 90.0, 5), 16
    elif mode in (7, 8):
        masses, pitch, az = M16, np.linspace(0.0, 90.0, 17), 16
    elif mode in (9, 10, 11):
        masses, pitch, az = np.array([1.0]), np.linspace(0.0, 90.0, 17), 16
    else:
        # Each Mode-12 raw record contains two 48x1x32 subrecords.  Its lone
        # angular cell is azimuth-integrated, so three-component flow and a 2-D
        # VDF cannot be reconstructed from the released product.
        masses, pitch, az = M12, np.array([67.4, 90.0]), 1
    return ModeLayout(
        mode=mode,
        energy_eV=energy.copy(),
        mass_amu=masses.copy(),
        pitch_edges_deg=pitch,
        azimuth_count=az,
        raw_subrecords=2 if mode == 12 else 1,
        directional_velocity_available=mode != 12,
    )


def species_mass_indices(mode: int, species: str) -> np.ndarray:
    """Indices used for H+, O+ or O2+ in a calibrated mode."""
    key = species.replace("+", "").upper()
    target = {"H": 1.0, "O": 16.0, "O2": 32.0}.get(key)
    if target is None:
        raise ValueError(f"Unsupported species {species!r}; choose H+, O+, or O2+")
    layout = mode_layout(mode)
    exact = np.flatnonzero(np.isclose(layout.mass_amu, target))
    if exact.size:
        return exact.astype(int)
    if mode == 12:
        return np.asarray({"H": [0, 1], "O": [18, 19], "O2": [26, 27]}[key], dtype=int)
    return np.array([], dtype=int)


def angular_geometry(mode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pitch, azimuth and solid angle for each flattened angular cell."""
    layout = mode_layout(mode)
    p0, p1 = layout.pitch_edges_deg[:-1], layout.pitch_edges_deg[1:]
    pitch = (p0 + p1) / 2.0
    az_edges = np.linspace(0.0, 360.0, layout.azimuth_count + 1)
    azimuth = (az_edges[:-1] + az_edges[1:]) / 2.0
    dphi = np.deg2rad(360.0 / layout.azimuth_count)
    domega_pitch = (np.cos(np.deg2rad(p0 + 90.0)) - np.cos(np.deg2rad(p1 + 90.0))) * dphi
    return (
        np.repeat(pitch, layout.azimuth_count),
        np.tile(azimuth, pitch.size),
        np.repeat(domega_pitch, layout.azimuth_count),
    )


def energy_bin_edges_eV(energy_centers_eV: np.ndarray) -> np.ndarray:
    """Return logarithmic bin edges inferred from adjacent energy centers.

    Interior edges are geometric means. The first and last half-bin ratios are
    extrapolated from their nearest adjacent centers. This reproduces the
    energy-table convention used by the validated ``NV_MSO_2`` density update
    while retaining the small per-bin differences present in rounded tables.
    """
    centers = np.asarray(energy_centers_eV, dtype=float)
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("energy_centers_eV must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(centers)) or np.any(centers <= 0.0) or np.any(np.diff(centers) <= 0.0):
        raise ValueError("energy centers must be finite, positive, and strictly increasing")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return edges


def energy_bin_widths_eV(mode: int) -> np.ndarray:
    """Return the actual full integration width of every calibrated energy bin."""
    edges = energy_bin_edges_eV(mode_layout(mode).energy_eV)
    return np.diff(edges)


def base_to_legacy_direction(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    """Reproduce ``TW1_MINPA_deal_v5.m`` look-direction conversion.

    The returned vectors are in the intermediate convention stored by the
    legacy ``NV`` products.  Keeping this stage explicit is important: the
    subsequent ``TW1_MINPA_redeal_3.m`` processing applies the same signed
    permutation twice before the MOMAG attitude rotation.
    """
    theta = np.asarray(theta_deg, dtype=float)
    phi = np.asarray(phi_deg, dtype=float)
    bb = np.deg2rad(theta + 90.0)
    pr = np.deg2rad(phi)
    base = np.column_stack((-np.cos(pr) * np.sin(bb), -np.sin(pr) * np.sin(bb), -np.cos(bb)))
    ry90 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    rxm90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return base @ (ry90 @ rxm90).T


def legacy_to_probe_direction(directions: np.ndarray) -> np.ndarray:
    """Apply the historical one-step ``[-V3,-V1,+V2]`` mapping.

    This function is retained only for reproducing older workflow text. The
    validated local NV-to-MSO comparison uses :func:`minpa_to_body_direction`.
    """
    values = np.asarray(directions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"directions must have shape (N, 3), got {values.shape}")
    return np.column_stack((-values[:, 2], -values[:, 0], values[:, 1]))


def minpa_to_body_direction(directions: np.ndarray) -> np.ndarray:
    """Map MINPA payload components into spacecraft body coordinates.

    This signed permutation is the best match found by the all-species local
    ``NV``/``NV_MSO`` rotation audit and is the convention used by the latest
    MINPA--SWIA comparison: ``[Xb,Yb,Zb]=[-V2,+V3,-V1]``.
    """
    values = np.asarray(directions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"directions must have shape (N, 3), got {values.shape}")
    return np.column_stack((-values[:, 1], values[:, 2], -values[:, 0]))


def legacy_nv_to_probe_direction(directions: np.ndarray) -> np.ndarray:
    """Compatibility alias; use :func:`minpa_to_body_direction`.

    The older function name called the target frame "Probe", but the target is
    the Tianwen-1 spacecraft body frame.
    """
    return minpa_to_body_direction(directions)


def minpa_directions(mode: int) -> np.ndarray:
    """Intermediate MINPA directions matching the legacy ``NV`` vectors."""
    theta, phi, _ = angular_geometry(mode)
    return base_to_legacy_direction(theta, phi)


def minpa_probe_directions(mode: int) -> np.ndarray:
    """Compatibility helper returning spacecraft-body-frame directions."""
    return minpa_body_directions(mode)


def minpa_body_directions(mode: int) -> np.ndarray:
    """Return MINPA look directions expressed in spacecraft body coordinates."""
    return minpa_to_body_direction(minpa_directions(mode))


def split_raw_record(mode: int, time_unix_s: float, values: np.ndarray, cadence_s: float = 4.1) -> list[tuple[float, np.ndarray]]:
    """Validate a flat record and split Mode 12 into its two timed subrecords."""
    layout = mode_layout(mode)
    flat = np.asarray(values, dtype=float).reshape(-1)
    n = layout.values_per_subrecord
    if mode == 12 and flat.size == 2 * n:
        return [(time_unix_s + cadence_s / 4.0, flat[:n]), (time_unix_s + 3.0 * cadence_s / 4.0, flat[n:])]
    if flat.size != n:
        raise ValueError(f"Mode {mode} expects {n * layout.raw_subrecords} raw values ({n} per subrecord), got {flat.size}")
    return [(float(time_unix_s), flat)]
