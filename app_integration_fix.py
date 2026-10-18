"""Small compatibility adjustment for the Step 3 DTW integration."""

from __future__ import annotations

from pathlib import Path

import app_integration as _base


_OLD = '''                        with st.expander(
                            "Refine a suggested moment with DTW · experimental",
                            expanded=True,
                        ):
                            st.caption(
'''

_NEW = '''                        with st.container(border=True):
                            st.markdown(
                                "#### Refine a suggested moment with DTW · experimental"
                            )
                            st.caption(
'''

if _OLD not in _base.SEED_REPLACEMENT:
    raise RuntimeError("DTW integration compatibility anchor was not found")

_base.SEED_REPLACEMENT = _base.SEED_REPLACEMENT.replace(_OLD, _NEW, 1)


def patch_app(source_root: Path) -> None:
    _base.patch_app(source_root)

    # A Streamlit container can retain the cloned /tmp source across a code
    # reload. Repair the first patch version even when its marker is already
    # present, so users do not need to clear server storage manually.
    app_path = source_root / "app.py"
    text = app_path.read_text(encoding="utf-8")
    if _OLD in text:
        app_path.write_text(text.replace(_OLD, _NEW, 1), encoding="utf-8")
