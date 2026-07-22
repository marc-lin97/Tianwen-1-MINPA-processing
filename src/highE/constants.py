"""Physical constants and project defaults."""

from __future__ import annotations

from pathlib import Path

AMU_KG = 1.66053906660e-27
EV_TO_J = 1.602176634e-19
MARS_RADIUS_KM = 3389.5

DEFAULT_MAVEN_D1_ROOT = Path(r"D:\Data\MAVEN\rawdata\static\l2\d1-32e4d16a8m")
DEFAULT_MAVEN_MAG_ROOT = Path(r"D:\Data\MAVEN\result\mag\ss1s")
DEFAULT_MAVEN_R_ROOT = Path(r"D:\Data\MAVEN\result\R_MSO2MSE")
DEFAULT_TW1_MINPA_ORI_ROOT = Path(r"D:\Data\TW-1\result\MINPA\ori")
DEFAULT_TW1_MINPA_PUBLIC_ROOT = Path(r"D:\Data\TW-1\rawdata\MINPA\public")
DEFAULT_TW1_MOMAG_ROOT = Path(r"D:\Data\TW-1\result\MOMAG\C\01Hz_all")
DEFAULT_TW1_R_ROOT = Path(r"D:\Data\TW-1\result\R_MSO2MSE")
DEFAULT_TW1_NV_MSO2_ROOT = Path(r"D:\Data\TW-1\result\MINPA\NV_MSO_2")
DEFAULT_OUTPUT_ROOT = Path(r"E:\Data\highE")

HIGH_ENERGY_MIN_EV = 1000.0
MATCH_MAX_GAP_S = 8.0

# STATIC's deflected FOV is fixed in STATIC coordinates. These are D1 theta
# bin-edge limits: outer center +/- half bin width, consistent with the
# instrument manual's approximate 360 deg x 90 deg FOV.
STATIC_FOV_THETA_MIN_DEG = -45.866667
STATIC_FOV_THETA_MAX_DEG = 45.866667

SPECIES = {
    "Oplus": {
        "label": "O+",
        "code": 1,
        "mass_window_amu": (14.5, 17.5),
    },
    "O2plus": {
        "label": "O2+",
        "code": 2,
        "mass_window_amu": (30.0, 34.0),
    },
}

D1_BAD_QUALITY_BITS = (6, 7)
