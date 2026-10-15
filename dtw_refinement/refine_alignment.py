"""DTW/librosa timing refinement for a known Sony SampleID candidate.

This module assumes we already have a plausible source region and a plausible
derivative region. It does NOT decide whether sampling occurred. It uses
subsequence Dynamic Time Warping (DTW) on chroma features to tighten the
DERIVATIVE placement around the supplied Sony candidate.

Important: the source start/end supplied by the user are treated as the query
excerpt. This POC refines where that excerpt best aligns in the derivative; it
does not yet infer a shorter source excerpt hidden inside a rough Sony window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import librosa
import numpy as np


@dataclass
class AlignmentResult:
    source_start_s: float
    source_end_s: float
    derivative_hint_start_s: float
    derivative_hint_end_s: float
    derivative_refined_start_s: float
    derivative_refined_end_s: float
    source_duration_s: float
    derivative_matched_duration_s: float
    playback_duration_ratio: float
    best_pitch_shift_semitones: int
    mean_dtw_cost: float
    search_start_s: float
    search_end_s: float
    path_points: int

    def to_dict(self):
        return asdict(self)


def _duration(path: str) -> float:
    return float(librosa.get_duration(path=path))


def _load_segment(path: str, start_s: float, end_s: float, sr: int) -> np.ndarray:
    if end_s <= start_s:
        raise ValueError("End time must be after start time.")
    y, _ = librosa.load(
        path,
        sr=sr,
        mono=True,
        offset=max(0.0, float(start_s)),
        duration=float(end_s - start_s),
    )
    if y.size == 0:
        raise ValueError(f"No audio found in requested segment {start_s:.2f}-{end_s:.2f}s.")
    return y.astype(np.float32)


def _chroma(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Pitch-class representation used for flexible musical alignment.

    Chroma intentionally reduces octave/timbre detail. That makes it useful for
    timing alignment after Sony has already proposed a candidate, but it is NOT
    discriminative enough to serve as a standalone sample detector.
    """
    c = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=hop_length,
        n_chroma=12,
        bins_per_octave=36,
    ).astype(np.float64)

    # L2-normalize non-silent frames. Keep silent frames near zero.
    norms = np.linalg.norm(c, axis=0, keepdims=True)
    c = c / np.maximum(norms, 1e-8)
    return c


def _path_mean_cost(X: np.ndarray, Y: np.ndarray, wp: np.ndarray) -> float:
    if wp.size == 0:
        return float("inf")
    vals = []
    for i, j in wp:
        vals.append(float(np.linalg.norm(X[:, int(i)] - Y[:, int(j)])))
    return float(np.mean(vals)) if vals else float("inf")


def refine_candidate(
    source_path: str,
    derivative_path: str,
    *,
    source_start_s: float,
    source_end_s: float,
    derivative_hint_start_s: float,
    derivative_hint_end_s: float,
    search_padding_s: float = 8.0,
    max_pitch_shift_semitones: int = 6,
    sr: int = 22050,
    hop_length: int = 512,
) -> tuple[AlignmentResult, np.ndarray]:
    """Refine a Sony candidate placement using subsequence DTW.

    Returns:
      AlignmentResult
      warping path as Nx2 [source_frame, derivative_search_frame]
    """
    source_total = _duration(source_path)
    derivative_total = _duration(derivative_path)

    source_start_s = max(0.0, min(float(source_start_s), source_total))
    source_end_s = max(source_start_s, min(float(source_end_s), source_total))

    derivative_hint_start_s = max(0.0, min(float(derivative_hint_start_s), derivative_total))
    derivative_hint_end_s = max(
        derivative_hint_start_s,
        min(float(derivative_hint_end_s), derivative_total),
    )

    search_start_s = max(0.0, derivative_hint_start_s - float(search_padding_s))
    search_end_s = min(derivative_total, derivative_hint_end_s + float(search_padding_s))

    source_y = _load_segment(source_path, source_start_s, source_end_s, sr)
    derivative_y = _load_segment(derivative_path, search_start_s, search_end_s, sr)

    X = _chroma(source_y, sr, hop_length)
    Y = _chroma(derivative_y, sr, hop_length)

    if X.shape[1] < 2 or Y.shape[1] < 2:
        raise ValueError("Candidate region is too short to align.")

    if X.shape[1] > Y.shape[1]:
        raise ValueError(
            "Source excerpt is longer than the derivative search window. "
            "Increase search padding or shorten the source candidate."
        )

    best = None
    shifts = range(-int(max_pitch_shift_semitones), int(max_pitch_shift_semitones) + 1)

    for shift in shifts:
        X_shifted = np.roll(X, int(shift), axis=0)

        # subseq=True aligns the entire source query against the best subsequence
        # inside the larger derivative search region.
        _D, wp = librosa.sequence.dtw(
            X=X_shifted,
            Y=Y,
            metric="euclidean",
            subseq=True,
            backtrack=True,
        )

        if wp is None or len(wp) == 0:
            continue

        mean_cost = _path_mean_cost(X_shifted, Y, wp)
        y_frames = wp[:, 1].astype(int)
        local_start = float(librosa.frames_to_time(int(y_frames.min()), sr=sr, hop_length=hop_length))
        local_end = float(
            librosa.frames_to_time(int(y_frames.max()) + 1, sr=sr, hop_length=hop_length)
        )

        candidate = {
            "shift": int(shift),
            "cost": mean_cost,
            "wp": wp,
            "local_start": local_start,
            "local_end": local_end,
        }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    if best is None:
        raise RuntimeError("DTW could not produce an alignment path.")

    refined_start = max(0.0, search_start_s + best["local_start"])
    refined_end = min(derivative_total, search_start_s + best["local_end"])

    source_duration = source_end_s - source_start_s
    derivative_duration = max(1e-9, refined_end - refined_start)

    # >1 means the derivative placement is longer than the source excerpt.
    duration_ratio = derivative_duration / max(source_duration, 1e-9)

    result = AlignmentResult(
        source_start_s=round(source_start_s, 3),
        source_end_s=round(source_end_s, 3),
        derivative_hint_start_s=round(derivative_hint_start_s, 3),
        derivative_hint_end_s=round(derivative_hint_end_s, 3),
        derivative_refined_start_s=round(refined_start, 3),
        derivative_refined_end_s=round(refined_end, 3),
        source_duration_s=round(source_duration, 3),
        derivative_matched_duration_s=round(derivative_duration, 3),
        playback_duration_ratio=round(duration_ratio, 4),
        best_pitch_shift_semitones=int(best["shift"]),
        mean_dtw_cost=round(float(best["cost"]), 6),
        search_start_s=round(search_start_s, 3),
        search_end_s=round(search_end_s, 3),
        path_points=int(len(best["wp"])),
    )

    return result, best["wp"]
