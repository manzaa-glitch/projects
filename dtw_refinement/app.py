"""Streamlit test interface for DTW/librosa candidate refinement."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from refine_alignment import refine_candidate


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - 60 * minutes
    return f"{minutes}:{secs:05.2f}"


st.set_page_config(page_title="Applied AI DTW Refiner", page_icon="🎯", layout="wide")
st.title("Applied AI — DTW Timestamp Refiner")
st.caption(
    "Use this AFTER Sony SampleID has proposed a source region and derivative region. "
    "DTW searches around the derivative candidate and tightens the placement."
)

with st.expander("How to use this", expanded=True):
    st.markdown(
        """
1. Upload the **original/source song** and **sampled/derivative song**.
2. Copy one candidate region from the Sony output.
3. Enter Sony's source start/end and derivative start/end below.
4. Click **Refine candidate**.
5. Compare the refined derivative boundaries with what you hear.

This POC treats the Sony **source window as the query excerpt**. It refines the
derivative placement around that query. It does not yet discover a shorter source
excerpt inside a rough 5-second Sony source window.
        """
    )

left, right = st.columns(2)
with left:
    source = st.file_uploader(
        "Original / source song",
        type=["wav", "mp3", "flac", "m4a"],
        key="dtw_source",
    )
with right:
    derivative = st.file_uploader(
        "Sampled / derivative song",
        type=["wav", "mp3", "flac", "m4a"],
        key="dtw_derivative",
    )

st.subheader("Sony candidate timestamps")
c1, c2, c3, c4 = st.columns(4)
with c1:
    source_start = st.number_input("Source start (s)", min_value=0.0, value=0.0, step=0.5)
with c2:
    source_end = st.number_input("Source end (s)", min_value=0.1, value=5.0, step=0.5)
with c3:
    derivative_start = st.number_input("Derivative start (s)", min_value=0.0, value=0.0, step=0.5)
with c4:
    derivative_end = st.number_input("Derivative end (s)", min_value=0.1, value=5.0, step=0.5)

with st.expander("Refinement settings", expanded=False):
    padding = st.slider(
        "Search padding around Sony derivative candidate (seconds)",
        min_value=1.0,
        max_value=30.0,
        value=8.0,
        step=1.0,
    )
    max_pitch = st.slider(
        "Pitch-shift search (+/- semitones)",
        min_value=0,
        max_value=12,
        value=6,
        step=1,
    )

if source and derivative:
    st.audio(source)
    st.audio(derivative)

    if st.button("Refine candidate", type="primary"):
        with st.spinner("Running chroma + subsequence DTW..."):
            with tempfile.TemporaryDirectory() as tmp:
                source_path = Path(tmp) / source.name
                derivative_path = Path(tmp) / derivative.name
                source_path.write_bytes(source.getbuffer())
                derivative_path.write_bytes(derivative.getbuffer())

                try:
                    result, wp = refine_candidate(
                        str(source_path),
                        str(derivative_path),
                        source_start_s=float(source_start),
                        source_end_s=float(source_end),
                        derivative_hint_start_s=float(derivative_start),
                        derivative_hint_end_s=float(derivative_end),
                        search_padding_s=float(padding),
                        max_pitch_shift_semitones=int(max_pitch),
                    )
                except Exception as exc:
                    st.error(f"Refinement failed: {exc}")
                else:
                    st.success("Refinement complete")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(
                        "Refined derivative start",
                        fmt_time(result.derivative_refined_start_s),
                    )
                    m2.metric(
                        "Refined derivative end",
                        fmt_time(result.derivative_refined_end_s),
                    )
                    m3.metric(
                        "Matched duration",
                        f"{result.derivative_matched_duration_s:.2f}s",
                    )
                    m4.metric(
                        "Estimated pitch shift",
                        f"{result.best_pitch_shift_semitones:+d} st",
                    )

                    st.markdown(
                        f"**Source query:** {fmt_time(result.source_start_s)}–"
                        f"{fmt_time(result.source_end_s)}  \\n"
                        f"**Sony derivative hint:** {fmt_time(result.derivative_hint_start_s)}–"
                        f"{fmt_time(result.derivative_hint_end_s)}  \\n"
                        f"**DTW refined derivative:** {fmt_time(result.derivative_refined_start_s)}–"
                        f"{fmt_time(result.derivative_refined_end_s)}"
                    )

                    summary = pd.DataFrame([result.to_dict()])
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                    st.caption(
                        "DTW cost is a relative alignment cost: lower is better. "
                        "It is not a probability that sampling occurred."
                    )
                    st.caption(
                        "Duration ratio > 1 means the derivative placement is longer than "
                        "the supplied source excerpt; < 1 means shorter."
                    )

                    st.download_button(
                        "Download refinement CSV",
                        summary.to_csv(index=False).encode("utf-8"),
                        file_name="dtw_refinement.csv",
                        mime="text/csv",
                    )

st.info(
    "Use this as a timing experiment, not as a standalone detector. Chroma can align "
    "musically similar material, so Sony's candidate localization (or a human-provided candidate) "
    "should come first."
)
