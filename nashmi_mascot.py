"""Nashmi, the pixel-art fox that lives on the chat composer's top edge.

The art, the tunable numbers and the animation engine live in ``mascot/``:

    nashmi_fox_config.js   every tunable value (size, speeds, intervals...)
    nashmi_fox_sprites.js  the pixel grids (one character per pixel)
    nashmi_fox.js          the engine (canvas drawing, pointer chase, states)

Streamlit renders components inside iframes, so ``loader_html`` returns a
tiny script for such an iframe that copies the engine into the main page
(same origin) once. The engine keeps running across reruns and replaces
itself only when the files change.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

MASCOT_DIR = Path(__file__).with_name("mascot")
MASCOT_FILES = ("nashmi_fox_config.js", "nashmi_fox_sprites.js", "nashmi_fox.js")


@lru_cache(maxsize=1)
def loader_html() -> str:
    """Script tag that installs the fox in the parent page (empty if missing)."""
    try:
        code = "\n".join(
            (MASCOT_DIR / name).read_text(encoding="utf-8") for name in MASCOT_FILES
        )
    except OSError:
        return ""
    version = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
    code = f"const NASHMI_FOX_VERSION = {json.dumps(version)};\n{code}"
    payload = json.dumps("{\n" + code + "\n}").replace("</", "<\\/")
    return (
        "<script>(() => { try {"
        " const P = window.parent, fox = P.__nashmiFox;"
        f" if (fox && fox.version === {json.dumps(version)}) return;"
        " const s = P.document.createElement('script');"
        f" s.textContent = {payload};"
        " P.document.head.appendChild(s); s.remove();"
        " } catch (_) { /* cross-origin: no fox */ } })();</script>"
    )
