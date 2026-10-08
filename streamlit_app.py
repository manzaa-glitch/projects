"""Streamlit Cloud entry point for the Applied AI SampleID app.

This keeps the cloud deployment path simple while the actual app code
remains in sony_sampleid/app.py.
"""

from pathlib import Path
import runpy

APP = Path(__file__).parent / "sony_sampleid" / "app.py"
runpy.run_path(str(APP), run_name="__main__")
