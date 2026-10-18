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
