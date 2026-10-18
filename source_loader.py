"""Load the Lastrada DTW replica source branch for Streamlit Cloud.

The user's repository owns the deployment, while the application source stays
on the shared Lastrada development branch. The clone is cached in /tmp for the
lifetime of the Streamlit container.
"""

from __future__ import annotations

import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

SOURCE_REPOSITORY = "https://github.com/tankr98/Lastrada.git"
SOURCE_BRANCH = "feat/lastrada-dtw-replica"
SOURCE_DIR = Path(os.getenv("LASTRADA_REPLICA_SOURCE_DIR", "/tmp/lastrada-dtw-replica"))


def ensure_source() -> Path:
    marker = SOURCE_DIR / ".git"
    if not marker.exists():
        shutil.rmtree(SOURCE_DIR, ignore_errors=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                SOURCE_BRANCH,
                SOURCE_REPOSITORY,
                str(SOURCE_DIR),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return SOURCE_DIR


def run_source_script(relative_path: str) -> None:
    root = ensure_source()
    script = root / relative_path
    if not script.is_file():
        raise FileNotFoundError(f"Lastrada source script not found: {script}")

    os.chdir(root)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    runpy.run_path(str(script), run_name="__main__")
