"""Runtime patch that places guarded DTW inside Sri's Step 3 review workflow.

The replica intentionally executes the shared Lastrada source branch. This patch
keeps that source intact while adding a DTW reviewer panel directly above the
existing timing table in `Listen & check`.
"""

from __future__ import annotations

from pathlib import Path


PATCH_MARKER = "# LASTRADA_DTW_STEP3_INTEGRATION_V1"


IMPORT_ANCHOR = "from lastrada.alignment.intervals import coverage\n"
IMPORT_REPLACEMENT = (
    IMPORT_ANCHOR
    + "from lastrada.alignment.dtw import refine_candidate_dtw_details\n"
)

LISTEN_INTERVAL_ANCHOR = '    intervals = suggested_intervals(result, "master")\n'
LISTEN_INTERVAL_REPLACEMENT = (
    '    intervals = st.session_state.get("review_master_seed_intervals") or '
    'suggested_intervals(result, "master")\n'
)

SEED_ANCHOR = '''                    seeds = st.session_state[key + "seed_intervals"]
                    agreement = st.radio(
'''

SEED_REPLACEMENT = '''                    seeds = st.session_state[key + "seed_intervals"]

                    # LASTRADA_DTW_STEP3_INTEGRATION_V1
                    if rights == "master" and seeds:
                        with st.expander(
                            "Refine a suggested moment with DTW · experimental",
                            expanded=True,
                        ):
                            st.caption(
                                "DTW lines up one approximate source region with one "
                                "suggested region in your track. It can tighten timing "
                                "after pitch or speed changes, but it cannot prove that "
                                "sampling occurred. Listen before accepting the result."
                            )

                            candidate_index = st.selectbox(
                                "Suggested moment to refine",
                                options=list(range(len(seeds))),
                                format_func=lambda index: (
                                    f"Your track {seeds[index].new_start:.2f}–"
                                    f"{seeds[index].new_end:.2f}s"
                                    + (
                                        f" · Original {seeds[index].original_start:.2f}–"
                                        f"{seeds[index].original_end:.2f}s"
                                        if seeds[index].original_start is not None
                                        and seeds[index].original_end is not None
                                        else " · original location needed"
                                    )
                                ),
                                key=key + "dtw_candidate_index",
                            )
                            selected = seeds[candidate_index]
                            original_duration = float(result.durations["original"])
                            selected_duration = max(
                                0.5, selected.new_end - selected.new_start
                            )
                            default_source_start = float(
                                selected.original_start
                                if selected.original_start is not None
                                else 0.0
                            )
                            default_source_start = min(
                                default_source_start,
                                max(0.0, original_duration - 0.1),
                            )
                            default_source_end = float(
                                selected.original_end
                                if selected.original_end is not None
                                else min(
                                    original_duration,
                                    default_source_start
                                    + max(1.0, selected_duration),
                                )
                            )
                            if default_source_end <= default_source_start:
                                default_source_end = min(
                                    original_duration,
                                    default_source_start + 0.5,
                                )

                            if selected.original_start is None:
                                st.info(
                                    "The current PF-NMF suggestion has no original-song "
                                    "timestamp. Enter the approximate original region you "
                                    "believe was used, then DTW will refine the new-track "
                                    "placement. Sony candidates will fill these fields "
                                    "automatically when SampleID is enabled."
                                )

                            source_columns = st.columns(2)
                            with source_columns[0]:
                                source_start = st.number_input(
                                    "Approximate original start (s)",
                                    min_value=0.0,
                                    max_value=original_duration,
                                    value=default_source_start,
                                    step=0.1,
                                    format="%.2f",
                                    key=key
                                    + f"dtw_source_start_{candidate_index}",
                                )
                            with source_columns[1]:
                                source_end = st.number_input(
                                    "Approximate original end (s)",
                                    min_value=0.0,
                                    max_value=original_duration,
                                    value=default_source_end,
                                    step=0.1,
                                    format="%.2f",
                                    key=key
                                    + f"dtw_source_end_{candidate_index}",
                                )

                            with st.expander("DTW guardrails", expanded=False):
                                dtw_columns = st.columns(3)
                                with dtw_columns[0]:
                                    dtw_padding = st.number_input(
                                        "Search padding (s)",
                                        min_value=0.0,
                                        max_value=30.0,
                                        value=float(
                                            settings.dtw_search_padding_seconds
                                        ),
                                        step=0.5,
                                        key=key + "dtw_padding",
                                    )
                                with dtw_columns[1]:
                                    dtw_min_ratio = st.number_input(
                                        "Minimum duration ratio",
                                        min_value=0.1,
                                        max_value=1.0,
                                        value=float(settings.dtw_min_duration_ratio),
                                        step=0.05,
                                        key=key + "dtw_min_ratio",
                                    )
                                with dtw_columns[2]:
                                    dtw_max_ratio = st.number_input(
                                        "Maximum duration ratio",
                                        min_value=1.0,
                                        max_value=3.0,
                                        value=float(settings.dtw_max_duration_ratio),
                                        step=0.05,
                                        key=key + "dtw_max_ratio",
                                    )

                            if st.button(
                                "Run DTW on this moment",
                                type="primary",
                                key=key + "run_dtw",
                            ):
                                if source_end <= source_start:
                                    st.error(
                                        "Original end must be after original start."
                                    )
                                else:
                                    dtw_candidate = Interval(
                                        new_start=selected.new_start,
                                        new_end=selected.new_end,
                                        original_start=float(source_start),
                                        original_end=float(source_end),
                                        source="review_dtw_candidate",
                                        status="candidate",
                                        similarity=selected.similarity,
                                    )
                                    with st.spinner(
                                        "Aligning the selected source and new-track regions…"
                                    ):
                                        dtw_detail = refine_candidate_dtw_details(
                                            st.session_state.original_audio,
                                            st.session_state.new_audio,
                                            dtw_candidate,
                                            search_padding=float(dtw_padding),
                                            max_pitch_shift_semitones=(
                                                settings.dtw_max_pitch_shift_semitones
                                            ),
                                            minimum_duration_ratio=float(
                                                dtw_min_ratio
                                            ),
                                            maximum_duration_ratio=float(
                                                dtw_max_ratio
                                            ),
                                            minimum_duration=(
                                                settings.dtw_minimum_duration_seconds
                                            ),
                                        )
                                    st.session_state[key + "dtw_result"] = {
                                        "candidate_index": candidate_index,
                                        "accepted": dtw_detail.accepted,
                                        "reason": dtw_detail.reason,
                                        "interval": dtw_detail.interval.model_dump(),
                                        "mean_cost": dtw_detail.mean_cost,
                                        "pitch_shift": (
                                            dtw_detail.pitch_shift_semitones
                                        ),
                                        "duration_ratio": dtw_detail.duration_ratio,
                                        "source_duration": dtw_detail.source_duration,
                                        "derivative_duration": (
                                            dtw_detail.derivative_duration
                                        ),
                                    }

                            dtw_result = st.session_state.get(
                                key + "dtw_result"
                            )
                            if (
                                dtw_result
                                and dtw_result["candidate_index"]
                                == candidate_index
                            ):
                                refined = dtw_result["interval"]
                                comparison = pd.DataFrame(
                                    [
                                        {
                                            "Evidence": "Before DTW",
                                            "Original start": source_start,
                                            "Original end": source_end,
                                            "Your track start": selected.new_start,
                                            "Your track end": selected.new_end,
                                        },
                                        {
                                            "Evidence": "DTW proposal",
                                            "Original start": refined[
                                                "original_start"
                                            ],
                                            "Original end": refined[
                                                "original_end"
                                            ],
                                            "Your track start": refined[
                                                "new_start"
                                            ],
                                            "Your track end": refined[
                                                "new_end"
                                            ],
                                        },
                                    ]
                                )
                                st.dataframe(
                                    comparison,
                                    use_container_width=True,
                                    hide_index=True,
                                )

                                metric_columns = st.columns(3)
                                metric_columns[0].metric(
                                    "Duration ratio",
                                    (
                                        f"{dtw_result['duration_ratio']:.2f}×"
                                        if dtw_result["duration_ratio"]
                                        is not None
                                        else "—"
                                    ),
                                )
                                metric_columns[1].metric(
                                    "Pitch shift",
                                    (
                                        f"{dtw_result['pitch_shift']:+d} st"
                                        if dtw_result["pitch_shift"]
                                        is not None
                                        else "—"
                                    ),
                                )
                                metric_columns[2].metric(
                                    "DTW cost",
                                    (
                                        f"{dtw_result['mean_cost']:.3f}"
                                        if dtw_result["mean_cost"] is not None
                                        else "—"
                                    ),
                                )

                                if dtw_result["accepted"]:
                                    st.success(
                                        "DTW found a timing proposal that passed "
                                        "the overlap and duration guardrails."
                                    )
                                    if st.button(
                                        "Use this DTW timing in the review table",
                                        key=key + "apply_dtw",
                                    ):
                                        updated_seeds = list(seeds)
                                        updated_seeds[candidate_index] = Interval(
                                            **refined
                                        )
                                        st.session_state[
                                            key + "seed_intervals"
                                        ] = updated_seeds
                                        st.session_state.pop(
                                            key + "intervals", None
                                        )
                                        st.session_state[key + "agreement"] = (
                                            "Not checked"
                                        )
                                        st.session_state[
                                            key + "times_confirmed"
                                        ] = False
                                        st.session_state.pop(
                                            key + "dtw_result", None
                                        )
                                        st.session_state.pop(
                                            "comparison_signature", None
                                        )
                                        st.rerun(scope="app")
                                else:
                                    st.warning(
                                        "DTW abstained: "
                                        + dtw_result["reason"]
                                        + ". The existing timing has not been changed."
                                    )

                    agreement = st.radio(
'''


def patch_app(source_root: Path) -> None:
    """Apply the Step 3 integration to the cloned shared source, once."""
    app_path = source_root / "app.py"
    text = app_path.read_text(encoding="utf-8")

    if PATCH_MARKER in text:
        return

    missing = []
    for label, anchor in (
        ("DTW import", IMPORT_ANCHOR),
        ("comparison interval source", LISTEN_INTERVAL_ANCHOR),
        ("review timing insertion", SEED_ANCHOR),
    ):
        if anchor not in text:
            missing.append(label)

    if missing:
        raise RuntimeError(
            "Could not integrate DTW into the shared Lastrada app; missing "
            + ", ".join(missing)
        )

    text = text.replace(IMPORT_ANCHOR, IMPORT_REPLACEMENT, 1)
    text = text.replace(
        LISTEN_INTERVAL_ANCHOR,
        LISTEN_INTERVAL_REPLACEMENT,
        1,
    )
    text = text.replace(SEED_ANCHOR, SEED_REPLACEMENT, 1)
    app_path.write_text(text, encoding="utf-8")
