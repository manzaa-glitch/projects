# Lastrada DTW replica

This branch deploys a replica of Sri's Lastrada Nature of Use app from the shared repository branch `feat/lastrada-dtw-replica`.

## Streamlit Community Cloud settings

- Repository: `manzaa-glitch/projects`
- Branch: `lastrada-dtw-replica`
- Main file path: `streamlit_app.py`
- Python: `3.12`

The main page runs Sri's Lastrada application. Expand the Streamlit page navigation and open **DTW Lab** to test manual candidate DTW and the engineering alignment tools.

The replica source is cloned into `/tmp/lastrada-dtw-replica` when the app container starts. The deployment branch remains separate from the existing SampleID proof-of-concept on `main`.

## Current limits

- Manual DTW testing works without Sony's model checkpoint.
- Automatic SampleID matching still requires the Sony checkpoint and package.
- Demucs stem separation is optional and is not installed in this lightweight cloud deployment by default.
- Results are experimental candidate evidence, not verified sample-clearance conclusions.
