"""The embedded Web UI script must survive Python string processing intact."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from opendrop.ui.web import _WEB_UI


def _script() -> str:
    # The page has exactly one inline script; take the text between its tags.
    start = _WEB_UI.rindex("<script>") + len("<script>")
    end = _WEB_UI.index("</script>", start)
    return _WEB_UI[start:end]


def test_js_newline_escapes_not_turned_into_raw_newlines() -> None:
    # A non-raw Python string turns '\n' into a real newline inside a JS
    # string literal, which is a SyntaxError that kills the whole UI.
    js = _script()
    assert "split('\\n')" in js
    for line in js.splitlines():
        assert line.count("'") % 2 == 0 or "//" in line, line


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_web_ui_script_parses(tmp_path) -> None:
    path = tmp_path / "ui.js"
    path.write_text(_script(), encoding="utf-8")
    result = subprocess.run(
        ["node", "--check", str(path)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
