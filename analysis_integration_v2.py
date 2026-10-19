"""Strict Sony -> guarded-DTW integration for the Lastrada replica.

This revision stops the analysis instead of silently falling back to PF-NMF when
Sony SampleID is selected but unavailable. It also exposes the concrete failure
reason so checkpoint/package/model-load problems can be fixed directly.
"""

from __future__ import annotations

from pathlib import Path

import analysis_integration as _base


# Make model readiness visible before the user starts an analysis.
_SIDEBAR_STATUS_ANCHOR = '''        st.caption(
            "Default master analysis: Sony pairs source and derivative timestamps; "
            "direct waveform matching or guarded DTW refines them. PF-NMF remains "
            "available as contextual fallback evidence."
        )
'''

_SIDEBAR_STATUS_REPLACEMENT = '''        if use_sampleid:
            if detected_checkpoint:
                st.success(
                    "Sony model ready: " + str(detected_checkpoint)
                )
            else:
                st.error(
                    "Sony model is not ready. Add a valid sampleid-best.ckpt path "
                    "before running the master analysis."
                )
        st.caption(
            "Default master analysis: Sony pairs source and derivative timestamps; "
            "direct waveform matching or guarded DTW refines them. PF-NMF remains "
            "available only when you turn Sony analysis off."
        )
'''

if _SIDEBAR_STATUS_ANCHOR not in _base.SIDEBAR_REPLACEMENT:
    raise RuntimeError("Sony readiness UI anchor was not found")
_base.SIDEBAR_REPLACEMENT = _base.SIDEBAR_REPLACEMENT.replace(
    _SIDEBAR_STATUS_ANCHOR,
    _SIDEBAR_STATUS_REPLACEMENT,
    1,
)


# Do not continue into a misleading PF-NMF-only result when Sony is the selected
# default analysis path.
_base.CHECKPOINT_REPLACEMENT = '''                checkpoint, checkpoint_note = resolve_sampleid_checkpoint(
                    sampleid_checkpoint_override,
                    allow_download=use_sampleid,
                )
                if use_sampleid and "master" in intake.analysis_scope:
                    if checkpoint is None:
                        raise ValueError(
                            "Sony + DTW is enabled, but the Sony checkpoint is unavailable. "
                            + str(checkpoint_note)
                        )
                    try:
                        adapter = SampleIDAdapter(
                            checkpoint,
                            os.getenv("LASTRADA_CACHE_DIR", "cache") + "/sampleid",
                            batch_size=settings.sampleid_batch_size,
                        )
                    except SampleIDUnavailable as exc:
                        raise RuntimeError(
                            "Sony + DTW could not initialize: " + str(exc)
                        ) from exc
'''


# Preserve a detailed message for old sessions that already contain a fallback
# result, rather than showing only a generic banner.
_GENERIC_STATUS = '''                        elif sampleid_method.startswith("unavailable"):
                            st.warning(
                                "Sony + DTW were unavailable for this run. The moments below "
                                "come from PF-NMF and may not include original-track timestamps."
                            )
'''
_DETAILED_STATUS = '''                        elif sampleid_method.startswith("unavailable"):
                            sampleid_detail = sampleid_method.split(
                                "unavailable:", 1
                            )[-1].strip()
                            st.error(
                                "Sony + DTW did not run: " + sampleid_detail
                                + ". Start a new analysis after fixing this issue; "
                                "the moments below are PF-NMF fallback evidence."
                            )
'''
if _GENERIC_STATUS not in _base.MASTER_REPLACEMENT:
    raise RuntimeError("Detailed SampleID status anchor was not found")
_base.MASTER_REPLACEMENT = _base.MASTER_REPLACEMENT.replace(
    _GENERIC_STATUS,
    _DETAILED_STATUS,
    1,
)


_POST_ANALYSIS_ANCHOR = '''                if adapter_error:
                    result.methods["sampleid"] = f"unavailable: {adapter_error}"
                    result.warnings.append(adapter_error)
'''

_POST_ANALYSIS_REPLACEMENT = '''                if use_sampleid and "master" in intake.analysis_scope:
                    sampleid_status = str(result.methods.get("sampleid", ""))
                    if sampleid_status.startswith("unavailable") or sampleid_status.startswith(
                        "not_run"
                    ):
                        raise RuntimeError(
                            "Sony + DTW analysis failed: " + sampleid_status
                        )
                if adapter_error:
                    result.methods["sampleid"] = f"unavailable: {adapter_error}"
                    result.warnings.append(adapter_error)
'''


def patch_app(source_root: Path) -> None:
    """Apply the base integration, then enforce strict model execution."""
    _base.patch_app(source_root)

    app_path = source_root / "app.py"
    text = app_path.read_text(encoding="utf-8")
    if "# LASTRADA_SONY_DTW_STRICT_V2" in text:
        return
    if _POST_ANALYSIS_ANCHOR not in text:
        raise RuntimeError("Post-analysis Sony enforcement anchor was not found")
    text = text.replace(
        _POST_ANALYSIS_ANCHOR,
        "                # LASTRADA_SONY_DTW_STRICT_V2\n"
        + _POST_ANALYSIS_REPLACEMENT,
        1,
    )
    app_path.write_text(text, encoding="utf-8")
