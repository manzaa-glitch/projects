"""Integrate Sony SampleID and guarded DTW into the main Lastrada analysis flow.

The deployed replica executes Sri's shared ``app.py``. This module applies a
small, deterministic patch before that script runs so the normal seven-step
workflow uses Sony candidate matching by default and displays the resulting DTW
provenance inside Step 3. PF-NMF remains a fallback when the model is unavailable.
"""

from __future__ import annotations

from pathlib import Path


PATCH_MARKER = "# LASTRADA_SONY_DTW_DEFAULT_ANALYSIS_V1"

SETTINGS_ANCHOR = "settings = load_settings()\n\n"
SETTINGS_REPLACEMENT = '''settings = load_settings()

# LASTRADA_SONY_DTW_DEFAULT_ANALYSIS_V1
def _valid_sampleid_checkpoint(path):
    """Reject missing, partial, or HTML/error-page downloads."""
    if path is None:
        return False
    path = Path(path).expanduser()
    if not path.is_file() or path.stat().st_size < 100 * 1024 * 1024:
        return False
    try:
        import zipfile

        return zipfile.is_zipfile(path)
    except OSError:
        return False


def resolve_sampleid_checkpoint(override=None, allow_download=False):
    """Find a local Sony checkpoint, optionally downloading a configured mirror.

    Search order:
    1. path entered in the app;
    2. LASTRADA_SAMPLEID_CHECKPOINT;
    3. repository checkpoints/sampleid-best.ckpt;
    4. ~/.lastrada_sampleid/sampleid-best.ckpt;
    5. LASTRADA_SAMPLEID_CHECKPOINT_URL, downloaded atomically into the cache.
    """
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    env_path = os.getenv("LASTRADA_SAMPLEID_CHECKPOINT")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            Path("checkpoints/sampleid-best.ckpt"),
            Path.home() / ".lastrada_sampleid" / "sampleid-best.ckpt",
        ]
    )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _valid_sampleid_checkpoint(resolved):
            return str(resolved), None

    if allow_download:
        url = os.getenv("LASTRADA_SAMPLEID_CHECKPOINT_URL", "").strip()
        if url:
            cache_root = Path(os.getenv("LASTRADA_CACHE_DIR", "cache")) / "sampleid"
            cache_root.mkdir(parents=True, exist_ok=True)
            destination = cache_root / "sampleid-best.ckpt"
            partial = destination.with_suffix(".ckpt.part")
            if _valid_sampleid_checkpoint(destination):
                return str(destination.resolve()), None
            try:
                import os as _os
                import requests

                partial.unlink(missing_ok=True)
                with requests.get(
                    url,
                    stream=True,
                    timeout=180,
                    headers={"User-Agent": "Lastrada-SampleID/1.0"},
                ) as response:
                    response.raise_for_status()
                    with partial.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if not _valid_sampleid_checkpoint(partial):
                    raise RuntimeError("downloaded file is incomplete or not a PyTorch checkpoint")
                _os.replace(partial, destination)
                return str(destination.resolve()), None
            except Exception as exc:
                partial.unlink(missing_ok=True)
                return None, f"Sony checkpoint download failed: {exc}"

    return None, (
        "Sony checkpoint not found. Put sampleid-best.ckpt in "
        "~/.lastrada_sampleid/, checkpoints/, enter its path in Advanced options, "
        "or configure LASTRADA_SAMPLEID_CHECKPOINT_URL."
    )

'''

SIDEBAR_ANCHOR = '''        use_sampleid = st.checkbox("Include optional SampleID candidate matching", value=False)
        st.caption(
            "Experimental matching needs calibration. Scores do not establish accurate sample timestamps."
        )
'''

SIDEBAR_REPLACEMENT = '''        use_sampleid = st.checkbox(
            "Use Sony SampleID + guarded DTW timing analysis",
            value=True,
            help=(
                "Sony proposes paired original/new-track candidate regions. "
                "Guarded DTW then tightens each candidate when the timing remains plausible."
            ),
        )
        detected_checkpoint, _checkpoint_note = resolve_sampleid_checkpoint(
            allow_download=False
        )
        sampleid_checkpoint_override = st.text_input(
            "Sony checkpoint path · optional override",
            value=detected_checkpoint or "",
            placeholder=r"C:\\Users\\you\\.lastrada_sampleid\\sampleid-best.ckpt",
            disabled=not use_sampleid,
        )
        st.caption(
            "Default master analysis: Sony pairs source and derivative timestamps; "
            "direct waveform matching or guarded DTW refines them. PF-NMF remains "
            "available as contextual fallback evidence."
        )
'''

CHECKPOINT_ANCHOR = '''                checkpoint = os.getenv("LASTRADA_SAMPLEID_CHECKPOINT") or (
                    "checkpoints/sampleid-best.ckpt"
                    if Path("checkpoints/sampleid-best.ckpt").is_file()
                    else None
                )
                if use_sampleid and "master" in intake.analysis_scope:
                    try:
                        adapter = SampleIDAdapter(
                            checkpoint,
                            os.getenv("LASTRADA_CACHE_DIR", "cache") + "/sampleid",
                            batch_size=settings.sampleid_batch_size,
                        )
                    except SampleIDUnavailable as exc:
                        adapter_error = str(exc)
                        st.warning(adapter_error)
'''

CHECKPOINT_REPLACEMENT = '''                checkpoint, checkpoint_note = resolve_sampleid_checkpoint(
                    sampleid_checkpoint_override,
                    allow_download=use_sampleid,
                )
                if use_sampleid and "master" in intake.analysis_scope:
                    if checkpoint is None:
                        adapter_error = checkpoint_note
                        st.warning(
                            "Sony + DTW could not start, so this run will fall back "
                            "to PF-NMF timing evidence. " + str(adapter_error)
                        )
                    else:
                        try:
                            adapter = SampleIDAdapter(
                                checkpoint,
                                os.getenv("LASTRADA_CACHE_DIR", "cache") + "/sampleid",
                                batch_size=settings.sampleid_batch_size,
                            )
                        except SampleIDUnavailable as exc:
                            adapter_error = str(exc)
                            st.warning(
                                "Sony + DTW could not start, so this run will fall back "
                                "to PF-NMF timing evidence. " + adapter_error
                            )
'''

MASTER_ANCHOR = '''                    if rights == "master":
                        seconds, percent = (
'''

MASTER_REPLACEMENT = '''                    if rights == "master":
                        automated_occurrences = list(path.locate.sample_occurrences)
                        sampleid_method = result.methods.get("sampleid", "")
                        alignment_method = result.methods.get("alignment", "")
                        if automated_occurrences:
                            source_names = [item.source for item in automated_occurrences]
                            method_labels = {
                                "dtw_chroma_local_alignment": "guarded DTW",
                                "direct_waveform_local_alignment": "direct waveform",
                                "sampleid_chunk_cosine": "coarse Sony",
                            }
                            method_summary = ", ".join(
                                f"{method_labels.get(name, name)}: {source_names.count(name)}"
                                for name in dict.fromkeys(source_names)
                            )
                            st.success(
                                "Sony SampleID paired candidate regions in both tracks. "
                                "DTW was applied automatically where a plausible refinement "
                                "passed the timing guardrails."
                            )
                            st.caption(
                                "Automated timing evidence · " + method_summary
                                + (f" · pipeline: {alignment_method}" if alignment_method else "")
                            )
                        elif sampleid_method.startswith("unavailable"):
                            st.warning(
                                "Sony + DTW were unavailable for this run. The moments below "
                                "come from PF-NMF and may not include original-track timestamps."
                            )

                        seconds, percent = (
'''

SEEDS_ANCHOR = '''                    seeds = st.session_state[key + "seed_intervals"]
                    agreement = st.radio(
'''

SEEDS_REPLACEMENT = '''                    seeds = st.session_state[key + "seed_intervals"]
                    if rights == "master" and seeds:
                        paired = [
                            item for item in seeds
                            if item.original_start is not None
                            and item.original_end is not None
                        ]
                        if paired:
                            evidence_sources = [item.source for item in paired]
                            st.info(
                                f"{len(paired)} paired original ↔ new-track timing suggestion(s) "
                                "came from Sony and the automatic refinement pipeline. "
                                "Rows marked dtw_chroma_local_alignment were tightened by DTW; "
                                "other Sony rows were retained when DTW abstained."
                            )
                            with st.expander("Automated timing provenance", expanded=False):
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "method": item.source,
                                                "original_start": item.original_start,
                                                "original_end": item.original_end,
                                                "new_start": item.new_start,
                                                "new_end": item.new_end,
                                            }
                                            for item in paired
                                        ]
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                    agreement = st.radio(
'''


def patch_app(source_root: Path) -> None:
    """Apply the default Sony → DTW analysis integration once per clone."""
    app_path = source_root / "app.py"
    text = app_path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return

    replacements = [
        ("checkpoint resolver", SETTINGS_ANCHOR, SETTINGS_REPLACEMENT),
        ("default analysis option", SIDEBAR_ANCHOR, SIDEBAR_REPLACEMENT),
        ("checkpoint setup", CHECKPOINT_ANCHOR, CHECKPOINT_REPLACEMENT),
        ("Step 3 analysis status", MASTER_ANCHOR, MASTER_REPLACEMENT),
        ("Step 3 provenance", SEEDS_ANCHOR, SEEDS_REPLACEMENT),
    ]
    missing = [label for label, anchor, _ in replacements if anchor not in text]
    if missing:
        raise RuntimeError(
            "Could not integrate default Sony + DTW analysis; missing anchors: "
            + ", ".join(missing)
        )
    for _label, anchor, replacement in replacements:
        text = text.replace(anchor, replacement, 1)
    app_path.write_text(text, encoding="utf-8")
