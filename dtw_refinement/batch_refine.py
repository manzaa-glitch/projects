"""Batch DTW refinement for the complete detailed Sony SampleID output."""

from __future__ import annotations

from collections import Counter
from typing import Callable

import librosa
import pandas as pd

from .refine_alignment import refine_candidate


def parse_time_value(value) -> float:
    """Accept seconds as a number or time strings such as '1:23.5'."""
    if pd.isna(value):
        raise ValueError("Missing timestamp")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if ":" not in text:
        return float(text)

    parts = text.split(":")
    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)
    return total


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes}:{secs:05.2f}"


def _union_duration(intervals) -> float:
    spans = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not spans:
        return 0.0

    total = 0.0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


def _mode_int(values) -> int | None:
    vals = [int(v) for v in values if pd.notna(v)]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def refine_sony_detail(
    source_path: str,
    derivative_path: str,
    sony_detail_df: pd.DataFrame,
    *,
    search_padding_s: float = 8.0,
    max_pitch_shift_semitones: int = 6,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run DTW refinement across every placement in Sony's detailed output.

    Expected Sony columns:
      group, occurrence, source_start, source_end,
      derivative_start, derivative_end

    Returns:
      refined_detail_df
      refined_group_summary_df
    """
    required = {
        "source_start",
        "source_end",
        "derivative_start",
        "derivative_end",
    }
    missing = required - set(sony_detail_df.columns)
    if missing:
        raise ValueError(
            "Sony detailed output is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    derivative_total = float(librosa.get_duration(path=derivative_path))
    rows = []
    total = len(sony_detail_df)

    for idx, sony_row in sony_detail_df.reset_index(drop=True).iterrows():
        if progress_callback:
            progress_callback(idx, total)

        base = sony_row.to_dict()
        try:
            source_start = parse_time_value(sony_row["source_start"])
            source_end = parse_time_value(sony_row["source_end"])
            derivative_start = parse_time_value(sony_row["derivative_start"])
            derivative_end = parse_time_value(sony_row["derivative_end"])

            result, _wp = refine_candidate(
                source_path,
                derivative_path,
                source_start_s=source_start,
                source_end_s=source_end,
                derivative_hint_start_s=derivative_start,
                derivative_hint_end_s=derivative_end,
                search_padding_s=float(search_padding_s),
                max_pitch_shift_semitones=int(max_pitch_shift_semitones),
            )

            row = {
                **base,
                "status": "ok",
                "refined_derivative_start_s": result.derivative_refined_start_s,
                "refined_derivative_end_s": result.derivative_refined_end_s,
                "refined_derivative_start": format_time(result.derivative_refined_start_s),
                "refined_derivative_end": format_time(result.derivative_refined_end_s),
                "refined_duration_s": result.derivative_matched_duration_s,
                "duration_ratio": result.playback_duration_ratio,
                "pitch_shift_semitones": result.best_pitch_shift_semitones,
                "dtw_cost": result.mean_dtw_cost,
                "dtw_path_points": result.path_points,
                "error": "",
            }
        except Exception as exc:
            row = {
                **base,
                "status": "error",
                "refined_derivative_start_s": None,
                "refined_derivative_end_s": None,
                "refined_derivative_start": "",
                "refined_derivative_end": "",
                "refined_duration_s": None,
                "duration_ratio": None,
                "pitch_shift_semitones": None,
                "dtw_cost": None,
                "dtw_path_points": None,
                "error": str(exc),
            }

        rows.append(row)

    if progress_callback:
        progress_callback(total, total)

    detail = pd.DataFrame(rows)

    summaries = []
    if "group" not in detail.columns:
        detail["group"] = 1

    for group, g in detail.groupby("group", dropna=False, sort=False):
        ok = g[g["status"] == "ok"].copy()
        failed = len(g) - len(ok)

        if ok.empty:
            summaries.append(
                {
                    "group": group,
                    "placements_refined": 0,
                    "failed_placements": failed,
                    "refined_locations": "",
                    "total_refined_seconds": 0.0,
                    "coverage_percent_of_derivative": 0.0,
                    "typical_pitch_shift_semitones": None,
                    "median_dtw_cost": None,
                }
            )
            continue

        intervals = list(
            zip(
                ok["refined_derivative_start_s"].astype(float),
                ok["refined_derivative_end_s"].astype(float),
            )
        )
        total_refined = _union_duration(intervals)

        ok = ok.sort_values("refined_derivative_start_s")
        locations = "; ".join(
            f"{format_time(s)}–{format_time(e)}" for s, e in intervals
        )

        summaries.append(
            {
                "group": group,
                "placements_refined": len(ok),
                "failed_placements": failed,
                "refined_locations": locations,
                "total_refined_seconds": round(total_refined, 3),
                "coverage_percent_of_derivative": round(
                    100.0 * total_refined / max(derivative_total, 1e-9), 2
                ),
                "typical_pitch_shift_semitones": _mode_int(
                    ok["pitch_shift_semitones"].tolist()
                ),
                "median_dtw_cost": round(float(ok["dtw_cost"].median()), 6),
            }
        )

    summary = pd.DataFrame(summaries)

    if not summary.empty:
        summary = summary.sort_values(
            ["total_refined_seconds", "placements_refined"],
            ascending=[False, False],
        ).reset_index(drop=True)

    sort_cols = [c for c in ["group", "occurrence"] if c in detail.columns]
    if sort_cols:
        detail = detail.sort_values(sort_cols).reset_index(drop=True)

    return detail, summary
