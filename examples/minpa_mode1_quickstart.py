"""Complete real-data example used for the README figures.

Run from the repository root after ``pip install -e .``.  Change only the
three paths below for another workstation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
INPUT=Path(r"D:\Data\TW-1\result\MINPA\ori\HX1-Or_GRAS_MINPA-MOD1-DEF_SCI_N_20211202191251_20211203011723_00639_A.mat")
MOMAG=Path(r"D:\Data\TW-1\result\MOMAG\C\01Hz_all")
OUTPUT=ROOT/"docs"/"assets"/"minpa_mode1_example"

command=[
    sys.executable,str(ROOT/"scripts"/"run_minpa_pipeline.py"),str(INPUT),
    "--species","H+",
    "--start","2021-12-03T00:00:00Z","--stop","2021-12-03T01:00:00Z",
    "--vdf-start","2021-12-03T00:30:00Z","--vdf-stop","2021-12-03T00:31:00Z",
    "--momag-root",str(MOMAG),"--output",str(OUTPUT),
]
subprocess.run(command,cwd=ROOT,check=True)
