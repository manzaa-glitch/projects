"""Add actionable diagnostics to the default Sony -> DTW integration."""

from __future__ import annotations

from pathlib import Path

import analysis_integration as _base

_GENERIC = '''                        elif sampleid_method.startswith("unavailable"):
                            st.warning(
                                "Sony + DTW were unavailable for this run. The moments below "
                                "come from PF-NMF and may not include original-track timestamps."
                            )
'''

_DETAILED = '''                        elif sampleid_method.startswith("unavailable"):
                            unavailable_reason = sampleid_method.removeprefix(
                                "unavailable:"
                            ).strip()
                            st.warning(
                                "Sony + DTW did not run, so the moments below are PF-NMF "
                                "fallback estimates and may not include original-track "
                                "timestamps. Reason: "
                                + (unavailable_reason or "unknown SampleID setup error")
                            )
'''


def patch_app(source_root: Path) -> None:
    _base.patch_app(source_root)
    app_path = source_root / "app.py"
    text = app_path.read_text(encoding="utf-8")
    if _GENERIC in text:
        app_path.write_text(text.replace(_GENERIC, _DETAILED, 1), encoding="utf-8")
