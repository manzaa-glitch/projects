"""Reusable Sony SampleID localization logic for Applied AI experiments."""

from __future__ import annotations

from collections import defaultdict
import librosa
import numpy as np
import pandas as pd
import torch

SR = 16_000


def chunk_audio(path: str, chunk_seconds: float = 5.0, hop_seconds: float = 2.5):
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
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    outputs = []
    for i in range(0, len(audio_chunks), batch_size):
        x = torch.from_numpy(audio_chunks[i:i + batch_size]).to(device)
        with torch.inference_mode():
            z = model(x, audio=True)
        z = torch.nn.functional.normalize(z.squeeze(1), dim=-1)
        outputs.append(z.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _top_rank_points(sim: np.ndarray, top_k: int):
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


def candidate_sequences(
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
    """Find roughly time-consistent paths through top-ranked Sony matches."""
    points = _top_rank_points(sim, top_k=top_k)
    candidates = []
    bucket_width = max(1e-6, 2.0 * line_tolerance_windows)

    for slope in np.arange(slope_min, slope_max + 1e-9, slope_step):
        buckets = defaultdict(list)

        for p in points:
            intercept = p["o_idx"] - slope * p["d_idx"]
            bucket = int(round(intercept / bucket_width))
            buckets[bucket].append(p)

        for bucket_points in buckets.values():
            bucket_points.sort(key=lambda x: (x["d_idx"], x["rank"]))

            runs, current = [], []
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

                if not (slope_min <= fitted_slope <= slope_max) or rmse > line_tolerance_windows:
                    continue

                mean_rank = float(np.mean([p["rank"] for p in run]))
                mean_similarity = float(np.mean([p["similarity"] for p in run]))
                mean_relative_z = float(np.mean([p["relative_z"] for p in run]))

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


def _interval_overlap_fraction(a_start, a_end, b_start, b_end):
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    denom = max(1e-9, min(a_end - a_start, b_end - b_start))
    return inter / denom


def _union_duration(intervals):
    if not intervals:
        return 0.0
    spans = sorted((float(s), float(e)) for s, e in intervals)
    cur_s, cur_e = spans[0]
    total = 0.0
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


def _fmt_time(seconds):
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - 60 * minutes
    return f"{minutes}:{secs:04.1f}"


def group_candidate_regions(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    records = df.sort_values(
        ["sequence_score", "matched_windows", "mean_source_rank"],
        ascending=[False, False, True],
    ).to_dict("records")

    groups = []
    for row in records:
        target_group = None
        for group in groups:
            anchor = group["anchor"]
            overlap = _interval_overlap_fraction(
                row["original_start_s"], row["original_end_s"],
                anchor["original_start_s"], anchor["original_end_s"],
            )
            if overlap >= 0.50:
                target_group = group
                break

        if target_group is None:
            target_group = {"anchor": row, "rows": []}
            groups.append(target_group)

        duplicate_idx = None
        for i, old in enumerate(target_group["rows"]):
            d_overlap = _interval_overlap_fraction(
                row["derivative_start_s"], row["derivative_end_s"],
                old["derivative_start_s"], old["derivative_end_s"],
            )
            if d_overlap >= 0.60:
                duplicate_idx = i
                break

        if duplicate_idx is None:
            target_group["rows"].append(row)
        elif row["sequence_score"] > target_group["rows"][duplicate_idx]["sequence_score"]:
            target_group["rows"][duplicate_idx] = row

    groups.sort(key=lambda g: g["anchor"]["sequence_score"], reverse=True)
    summary_rows, detail_rows = [], []

    for group_num, group in enumerate(groups, start=1):
        anchor = group["anchor"]
        placements = sorted(group["rows"], key=lambda r: r["derivative_start_s"])
        if not placements:
            placements = [anchor]

        intervals = [(r["derivative_start_s"], r["derivative_end_s"]) for r in placements]

        summary_rows.append(
            {
                "group": group_num,
                "source_region": f"{_fmt_time(anchor['original_start_s'])}–{_fmt_time(anchor['original_end_s'])}",
                "placements": len(placements),
                "derivative_locations": "; ".join(
                    f"{_fmt_time(r['derivative_start_s'])}–{_fmt_time(r['derivative_end_s'])}"
                    for r in placements
                ),
                "total_detected_seconds": round(_union_duration(intervals), 2),
                "best_sequence_score": round(max(r["sequence_score"] for r in placements), 3),
                "supported_windows": int(sum(r["matched_windows"] for r in placements)),
                "best_mean_source_rank": round(min(r["mean_source_rank"] for r in placements), 2),
            }
        )

        for occurrence_num, r in enumerate(placements, start=1):
            detail_rows.append(
                {
                    "group": group_num,
                    "occurrence": occurrence_num,
                    "source_start": _fmt_time(r["original_start_s"]),
                    "source_end": _fmt_time(r["original_end_s"]),
                    "derivative_start": _fmt_time(r["derivative_start_s"]),
                    "derivative_end": _fmt_time(r["derivative_end_s"]),
                    "duration_s": round(r["derivative_end_s"] - r["derivative_start_s"], 2),
                    "sequence_score": round(r["sequence_score"], 3),
                    "supported_windows": int(r["matched_windows"]),
                    "mean_source_rank": round(r["mean_source_rank"], 2),
                    "alignment_slope": round(r["alignment_slope"], 3),
                    "mean_similarity": round(r["mean_similarity"], 4),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def analyze_with_model(
    model,
    original_path: str,
    derivative_path: str,
    *,
    chunk_seconds: float = 5.0,
    hop_seconds: float = 2.5,
    top_k: int = 3,
    min_support: int = 3,
):
    o_audio, o_spans = chunk_audio(original_path, chunk_seconds, hop_seconds)
    d_audio, d_spans = chunk_audio(derivative_path, chunk_seconds, hop_seconds)

    o = embed(model, o_audio)
    d = embed(model, d_audio)
    sim = d @ o.T

    candidates = candidate_sequences(
        sim,
        o_spans,
        d_spans,
        top_k=top_k,
        min_support=min_support,
    )
    summary, detail = group_candidate_regions(candidates)

    diagnostics = {
        "num_original_windows": len(o_spans),
        "num_derivative_windows": len(d_spans),
        "num_pairwise_scores": int(sim.size),
        "similarity_min": float(sim.min()),
        "similarity_median": float(np.median(sim)),
        "similarity_max": float(sim.max()),
    }
    return candidates, summary, detail, diagnostics
