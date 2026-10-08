"""Simple Streamlit UI for the Lastrada Sony SampleID proof of concept."""

from __future__ import annotations

import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch

from sampleid import SampleID


SR = 16_000
CKPT_URL = "https://zenodo.org/records/17413869/files/sampleid-best.ckpt?download=1"
CACHE_DIR = Path.home() / ".lastrada_sampleid"
CKPT_PATH = CACHE_DIR / "sampleid-best.ckpt"


def ensure_checkpoint() -> Path:
    """Download Sony's checkpoint to a normal user-writable folder.

    Sony's default loader tries to place the checkpoint inside site-packages.
    On Windows that can fail or behave inconsistently, so this app uses a local cache.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT_PATH.exists():
        urlretrieve(CKPT_URL, CKPT_PATH)
    return CKPT_PATH


@st.cache_resource
def load_model():
    ckpt = ensure_checkpoint()
    model = SampleID.load_checkpoint(ckpt_path=str(ckpt))
    model.eval()
    return model


def chunk_audio(path: str, chunk_seconds: float, hop_seconds: float):
    y, _ = librosa.load(path, sr=SR, mono=True)
    chunk_n = int(chunk_seconds * SR)
    hop_n = int(hop_seconds * SR)

    chunks, spans = [], []
    if len(y) <= chunk_n:
        padded = np.pad(y, (0, max(0, chunk_n - len(y))))[:chunk_n].astype(np.float32)
        return np.stack([padded]), [(0.0, len(y) / SR)]

    for start_n in range(0, len(y) - chunk_n + 1, hop_n):
        end_n = start_n + chunk_n
        chunks.append(y[start_n:end_n].astype(np.float32))
        spans.append((start_n / SR, end_n / SR))

    return np.stack(chunks), spans


def embed(model, audio_chunks: np.ndarray, batch_size: int = 32) -> np.ndarray:
    outputs = []
    for i in range(0, len(audio_chunks), batch_size):
        x = torch.from_numpy(audio_chunks[i : i + batch_size])
        with torch.inference_mode():
            z = model(x, audio=True)
        z = torch.nn.functional.normalize(z.squeeze(1), dim=-1)
        outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def analyze(original_path: str, derivative_path: str, chunk_seconds: float, hop_seconds: float, min_similarity: float):
    model = load_model()

    o_audio, o_spans = chunk_audio(original_path, chunk_seconds, hop_seconds)
    d_audio, d_spans = chunk_audio(derivative_path, chunk_seconds, hop_seconds)

    o = embed(model, o_audio)
    d = embed(model, d_audio)

    sim = d @ o.T
    flat_idx = np.argsort(sim.ravel())[::-1]

    rows = []
    for idx in flat_idx:
        di, oi = np.unravel_index(idx, sim.shape)
        score = float(sim[di, oi])
        if score < min_similarity:
            break
        rows.append({
            "similarity": round(score, 4),
            "original_start_s": round(o_spans[oi][0], 2),
            "original_end_s": round(o_spans[oi][1], 2),
            "derivative_start_s": round(d_spans[di][0], 2),
            "derivative_end_s": round(d_spans[di][1], 2),
        })

    return pd.DataFrame(rows)


st.set_page_config(page_title="Applied AI SampleID", page_icon="🎵", layout="wide")

st.title("Applied AI SampleID")
st.caption("Upload a known original song and a sampled/derivative song to find likely matching regions using Sony's pretrained SampleID model.")

with st.expander("Settings", expanded=False):
    chunk_seconds = st.number_input("Chunk size (seconds)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
    hop_seconds = st.number_input("Hop size (seconds)", min_value=0.5, max_value=10.0, value=2.5, step=0.5)
    min_similarity = st.slider(
        "Minimum match threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
        help="Only candidate window pairs with cosine similarity at or above this value will be shown. This threshold is experimental and must be calibrated on Applied AI project examples.",
    )

left, right = st.columns(2)
with left:
    original = st.file_uploader("Original / source song", type=["wav", "mp3", "flac", "m4a"], key="original")
with right:
    derivative = st.file_uploader("Sampled / derivative song", type=["wav", "mp3", "flac", "m4a"], key="derivative")

if original and derivative:
    st.audio(original)
    st.audio(derivative)

    if st.button("Analyze", type="primary"):
        with st.spinner("Running Sony SampleID..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                orig_path = Path(tmpdir) / original.name
                deriv_path = Path(tmpdir) / derivative.name
                orig_path.write_bytes(original.getbuffer())
                deriv_path.write_bytes(derivative.getbuffer())

                try:
                    df = analyze(str(orig_path), str(deriv_path), float(chunk_seconds), float(hop_seconds), float(min_similarity))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                else:
                    st.success("Analysis complete")
                    if df.empty:
                        st.warning(f"No candidate matches met the {min_similarity:.2f} minimum threshold.")
                    else:
                        st.write(f"{len(df)} candidate window pair(s) met the threshold.")
                        st.dataframe(df, use_container_width=True)

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download matches CSV",
                            data=csv,
                            file_name="sampleid_matches.csv",
                            mime="text/csv",
                        )

                        best = df.iloc[0]
                        st.markdown(
                            f"**Best candidate:** original {best.original_start_s:.2f}–{best.original_end_s:.2f}s "
                            f"↔ derivative {best.derivative_start_s:.2f}–{best.derivative_end_s:.2f}s "
                            f"(similarity {best.similarity:.4f})"
                        )

st.info("These are candidate matching windows, not final legal-grade sample boundaries. Use them as inputs to a second timing-refinement / Nature of Use step.")
