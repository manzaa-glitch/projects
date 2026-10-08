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


def _top_rank_points(sim: np.ndarray, top_k: int):
    """Keep only the best few source windows for each derivative window.

    Raw cosine values are not treated as probabilities. Rank is relative to
    the other source windows for the same derivative window.
    """
    points = []
    row_order = np.argsort(-sim, axis=1)

    for d_idx in range(sim.shape[0]):
        row = sim[d_idx]
        median = float(np.median(row))
        mad = float(1.4826 * np.median(np.abs(row - median)) + 1e-9)

        for rank0, o_idx in enumerate(row_order[d_idx, :top_k]):
            score = float(sim[d_idx, o_idx])
            relative_z = (score - median) / mad
            points.append(
                {
                    "d_idx": int(d_idx),
                    "o_idx": int(o_idx),
                    "rank": int(rank0 + 1),
                    "similarity": score,
                    "relative_z": float(relative_z),
                }
            )
    return points


def _candidate_sequences(
    sim: np.ndarray,
    o_spans,
    d_spans,
    *,
    top_k: int = 3,
    min_support: int = 3,
    slope_min: float = 0.5,
    slope_max: float = 2.0,
    slope_step: float = 0.05,
    line_tolerance_windows: float = 1.0,
    max_derivative_gap: int = 2,
    max_candidates: int = 50,
):
    """Find coherent time-progressing paths through top-ranked window matches.

    A genuine reused excerpt should create a roughly diagonal path in the
    derivative-time x source-time similarity map. A generic chord may score
    highly once, but it is much less useful unless neighboring derivative
    windows also point to neighboring source windows in the same order.

    We scan plausible alignment slopes to tolerate moderate time stretching.
    This is a heuristic localization layer around Sony embeddings, not part of
    Sony's published model and not a probability of sampling.
    """
    from collections import defaultdict

    points = _top_rank_points(sim, top_k=top_k)
    candidates = []

    # Search for approximate straight lines o ~= slope*d + intercept.
    # Bucket width is two source-window indices, corresponding to +/-1-window tolerance.
    bucket_width = max(1e-6, 2.0 * line_tolerance_windows)

    for slope in np.arange(slope_min, slope_max + 1e-9, slope_step):
        buckets = defaultdict(list)

        for p in points:
            intercept = p["o_idx"] - slope * p["d_idx"]
            bucket = int(round(intercept / bucket_width))
            buckets[bucket].append(p)

        for bucket_points in buckets.values():
            bucket_points.sort(key=lambda x: (x["d_idx"], x["rank"]))

            # Split into local runs instead of allowing a path to jump across the song.
            runs = []
            current = []
            last_d = None

            for p in bucket_points:
                if last_d is None or p["d_idx"] - last_d <= max_derivative_gap:
                    current.append(p)
                else:
                    if current:
                        runs.append(current)
                    current = [p]
                last_d = p["d_idx"]

            if current:
                runs.append(current)

            for run in runs:
                # One point per derivative window: retain its best rank.
                best_by_d = {}
                for p in run:
                    old = best_by_d.get(p["d_idx"])
                    if old is None or p["rank"] < old["rank"]:
                        best_by_d[p["d_idx"]] = p

                run = sorted(best_by_d.values(), key=lambda x: x["d_idx"])
                if len(run) < min_support:
                    continue

                d_idx = np.array([p["d_idx"] for p in run], dtype=float)
                o_idx = np.array([p["o_idx"] for p in run], dtype=float)

                fitted_slope, fitted_intercept = np.polyfit(d_idx, o_idx, 1)
                predicted = fitted_slope * d_idx + fitted_intercept
                rmse = float(np.sqrt(np.mean((o_idx - predicted) ** 2)))

                if not (slope_min <= fitted_slope <= slope_max):
                    continue
                if rmse > line_tolerance_windows:
                    continue

                mean_rank = float(np.mean([p["rank"] for p in run]))
                mean_similarity = float(np.mean([p["similarity"] for p in run]))
                mean_relative_z = float(np.mean([p["relative_z"] for p in run]))

                # Relative heuristic score only. Longer, better-ranked, straighter
                # paths rise to the top. This is NOT a calibrated probability.
                rank_quality = (top_k + 1.0 - mean_rank) / top_k
                straightness = 1.0 / (1.0 + rmse)
                contrast_bonus = 1.0 + 0.10 * max(0.0, min(mean_relative_z, 3.0))
                sequence_score = len(run) * rank_quality * straightness * contrast_bonus

                first_d = int(min(p["d_idx"] for p in run))
                last_d = int(max(p["d_idx"] for p in run))
                first_o = int(min(p["o_idx"] for p in run))
                last_o = int(max(p["o_idx"] for p in run))

                candidates.append(
                    {
                        "sequence_score": sequence_score,
                        "matched_windows": len(run),
                        "mean_source_rank": mean_rank,
                        "mean_similarity": mean_similarity,
                        "alignment_slope": float(fitted_slope),
                        "line_rmse_windows": rmse,
                        "original_start_s": float(o_spans[first_o][0]),
                        "original_end_s": float(o_spans[last_o][1]),
                        "derivative_start_s": float(d_spans[first_d][0]),
                        "derivative_end_s": float(d_spans[last_d][1]),
                        "_d_indices": {int(p["d_idx"]) for p in run},
                        "_o_range": (first_o, last_o),
                    }
                )

    # Keep strongest non-duplicate regions. Multiple scanned slopes can discover
    # effectively the same path.
    candidates.sort(key=lambda x: x["sequence_score"], reverse=True)
    kept = []

    for cand in candidates:
        duplicate = False
        for old in kept:
            d_intersection = len(cand["_d_indices"] & old["_d_indices"])
            d_denom = max(1, min(len(cand["_d_indices"]), len(old["_d_indices"])))
            d_overlap = d_intersection / d_denom

            a0, a1 = cand["_o_range"]
            b0, b1 = old["_o_range"]
            o_intersection = max(0, min(a1, b1) - max(a0, b0) + 1)
            o_denom = max(1, min(a1 - a0 + 1, b1 - b0 + 1))
            o_overlap = o_intersection / o_denom

            if d_overlap >= 0.60 and o_overlap >= 0.50:
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)
        if len(kept) >= max_candidates:
            break

    rows = []
    for i, cand in enumerate(kept, start=1):
        rows.append(
            {
                "candidate": i,
                "sequence_score": round(cand["sequence_score"], 3),
                "matched_windows": cand["matched_windows"],
                "mean_source_rank": round(cand["mean_source_rank"], 2),
                "original_start_s": round(cand["original_start_s"], 2),
                "original_end_s": round(cand["original_end_s"], 2),
                "derivative_start_s": round(cand["derivative_start_s"], 2),
                "derivative_end_s": round(cand["derivative_end_s"], 2),
                "alignment_slope": round(cand["alignment_slope"], 3),
                "mean_similarity": round(cand["mean_similarity"], 4),
            }
        )

    return pd.DataFrame(rows)


def analyze(
    original_path: str,
    derivative_path: str,
    chunk_seconds: float,
    hop_seconds: float,
    top_k: int,
    min_support: int,
):
    model = load_model()

    o_audio, o_spans = chunk_audio(original_path, chunk_seconds, hop_seconds)
    d_audio, d_spans = chunk_audio(derivative_path, chunk_seconds, hop_seconds)

    o = embed(model, o_audio)
    d = embed(model, d_audio)

    # Sony's evaluation uses normalized embeddings and dot products, which are
    # cosine similarities. We use the full matrix for RANKING, not a universal
    # absolute threshold.
    sim = d @ o.T

    sequences = _candidate_sequences(
        sim,
        o_spans,
        d_spans,
        top_k=top_k,
        min_support=min_support,
    )

    diagnostics = {
        "num_original_windows": len(o_spans),
        "num_derivative_windows": len(d_spans),
        "num_pairwise_scores": int(sim.size),
        "similarity_min": float(sim.min()),
        "similarity_median": float(np.median(sim)),
        "similarity_max": float(sim.max()),
    }
    return sequences, diagnostics

st.set_page_config(page_title="Applied AI SampleID", page_icon="🎵", layout="wide")

st.title("Applied AI SampleID")
st.caption("Upload a known original song and a sampled/derivative song to find likely matching regions using Sony's pretrained SampleID model.")

with st.expander("Settings", expanded=False):
    chunk_seconds = st.number_input("Chunk size (seconds)", min_value=2.0, max_value=15.0, value=5.0, step=0.5)
    hop_seconds = st.number_input("Hop size (seconds)", min_value=0.5, max_value=10.0, value=2.5, step=0.5)
    top_k = st.slider(
        "Top source candidates per derivative window",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="For each derivative window, keep only its best-ranked source windows. The raw cosine value is not treated as a probability.",
    )
    min_support = st.slider(
        "Minimum sequence support (windows)",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
        help="A candidate region must be supported by this many time-progressing window matches. Higher values are stricter but can miss very short samples.",
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
                    df, diagnostics = analyze(
                        str(orig_path),
                        str(deriv_path),
                        float(chunk_seconds),
                        float(hop_seconds),
                        int(top_k),
                        int(min_support),
                    )
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                else:
                    st.success("Analysis complete")

                    with st.expander("Similarity diagnostics"):
                        st.write(
                            f"Compared {diagnostics['num_derivative_windows']} derivative windows "
                            f"against {diagnostics['num_original_windows']} source windows "
                            f"({diagnostics['num_pairwise_scores']:,} raw pairwise scores)."
                        )
                        st.write(
                            "Raw cosine range: "
                            f"{diagnostics['similarity_min']:.4f} to {diagnostics['similarity_max']:.4f}; "
                            f"median {diagnostics['similarity_median']:.4f}."
                        )
                        st.caption(
                            "These cosine values are not probabilities. The localization below uses relative rank plus temporal consistency."
                        )

                    if df.empty:
                        st.warning(
                            "No coherent time-progressing sequence met the current settings. "
                            "Try reducing Minimum sequence support for very short samples."
                        )
                    else:
                        st.write(f"{len(df)} coherent candidate region(s) found.")
                        st.dataframe(df, use_container_width=True)

                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download candidate regions CSV",
                            data=csv,
                            file_name="sampleid_candidate_regions.csv",
                            mime="text/csv",
                        )

                        best = df.iloc[0]
                        st.markdown(
                            f"**Top sequence:** original {best.original_start_s:.2f}–{best.original_end_s:.2f}s "
                            f"↔ derivative {best.derivative_start_s:.2f}–{best.derivative_end_s:.2f}s "
                            f"supported by {int(best.matched_windows)} ranked windows."
                        )
                        st.caption(
                            "Sequence score is a relative heuristic used to rank candidate regions; it is not a probability of sampling."
                        )

st.info(
    "This version does not call a sample from one high cosine score. It looks for ranked matches that move through both songs in a coherent time sequence. "
    "The resulting regions are still candidates, not final legal-grade boundaries; pass them to the timing-refinement / Nature of Use step."
)
