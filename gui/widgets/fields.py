"""The config-form field factory: ``_widget``/``_read`` map a TOML value to/from
an editor widget, plus the shared label/tooltip helpers built on top."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
)

from gui.theme import tok
from gui.widgets._qt_utils import _no_wheel
from gui.widgets.sample_prompts import _SamplePromptsLauncher
from gui.widgets.target_res import _TargetResWidget

# flash4 is not supported yet (flash-attention-sm120 disabled)
_ATTN_MODES = ["flex", "flash"]


def _widget(v: Any, key: str = "") -> QWidget:
    if key == "target_res":
        sel = v if isinstance(v, (list, tuple)) else ([v] if v else [1024])
        return _TargetResWidget(sel)
    if key == "sample_prompts":
        return _SamplePromptsLauncher(v)
    if key == "sample_ratio":
        # Data-scope quick-pick: the common ratios (the CLI [half]/[quarter]/
        # [tenth] presets set the same key) plus free-typed values. Still one
        # flat float key — _read's QComboBox branch coerces via the float orig.
        w = QComboBox()
        w.setEditable(True)
        w.addItems(["1.0", "0.5", "0.25", "0.1"])
        try:
            w.setCurrentText(f"{float(v):g}")
        except (TypeError, ValueError):
            w.setCurrentText(str(v))
        return _no_wheel(w)
    if key == "attn_mode":
        w = QComboBox()
        w.addItems(_ATTN_MODES)
        idx = w.findText(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if key == "sample_decode_inline":
        # Tri-state: stored as the literal string "auto" / "true" / "false"
        # (all three are accepted by library.config.cli_args._optional_bool).
        # Must precede the bool branch below so a bool value gets the combo,
        # not a plain checkbox that can't express "auto".
        w = QComboBox()
        w.addItems(["auto", "true", "false"])
        if v is None:
            cur = "auto"
        elif isinstance(v, bool):
            cur = "true" if v else "false"
        else:
            cur = str(v).strip().lower()
            if cur in ("", "none"):
                cur = "auto"
        idx = w.findText(cur)
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if isinstance(v, bool):
        w = QCheckBox()
        w.setChecked(v)
        return w
    if isinstance(v, int):
        w = QSpinBox()
        # 10k default cap guards against typos; per-key overrides for fields that legitimately exceed it.
        if key == "min_pixels":
            w.setRange(0, 100_000_000)  # 100MP — covers any real image
        else:
            w.setRange(0, 10000)
        w.setValue(v)
        return _no_wheel(w)
    if isinstance(v, float):
        return QLineEdit(f"{v:g}")
    if isinstance(v, list):
        return QLineEdit(json.dumps(v))
    return QLineEdit(str(v))


def _read(w: QWidget, orig: Any = None) -> Any:
    if isinstance(w, _TargetResWidget):
        return w.value()
    if isinstance(w, _SamplePromptsLauncher):
        return w.value()
    if isinstance(w, QPlainTextEdit):
        # sample_prompts box → list of non-empty, non-comment lines.
        return [
            ln.strip()
            for ln in w.toPlainText().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    if isinstance(w, QComboBox):
        txt = w.currentText()
        # Editable numeric combos (sample_ratio) round-trip as their orig type;
        # plain string combos (attn_mode, …) have string origs and fall through.
        if isinstance(orig, float):
            try:
                return float(txt)
            except ValueError:
                pass
        return txt
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    txt = w.text()
    if isinstance(orig, float):
        try:
            return float(txt)
        except ValueError:
            pass
    if isinstance(orig, list):
        try:
            return json.loads(txt)
        except (json.JSONDecodeError, ValueError):
            pass
    # Normalize pasted Windows backslashes — forward slashes work everywhere and dodge TOML escape errors (e.g. "C:\Users" → \U is invalid).
    if "\\" in txt:
        txt = txt.replace("\\", "/")
    return txt


class ClickableLabel(QLabel):
    """QLabel that emits `clicked` on left-click.

    The config-style tabs use it for field labels: clicking the label routes
    that field's help text into the explanation panel.
    """

    clicked = Signal()

    def __init__(self, text: str = ""):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, ev):  # noqa: N802 — Qt event handler name
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


# Long single-line tooltips render as one screen-wide strip because Qt only
# word-wraps tooltip text it recognizes as rich text — and the rich-text paths
# (a width-bound table / a wrapping <div>) clip CJK at the right edge instead
# of reflowing. So we wrap to *plain* text with explicit newlines, measuring
# each line in display columns (CJK glyphs count as 2) so ko/ja/cn lines stay
# bounded and nothing is ever clipped.
TOOLTIP_WRAP_COLS = 56
_TOOLTIP_WRAP_MIN_LEN = 80
# Cheap "is this already HTML?" probe (PySide6 doesn't expose Qt.mightBeRichText).
_LOOKS_RICH = re.compile(r"<[a-zA-Z!/]")


def _display_width(s: str) -> int:
    """Terminal-style display width: East-Asian wide/fullwidth glyphs = 2."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _wrap_plain(text: str, cols: int) -> str:
    """Greedy word-wrap to *cols* display columns; hard-break runs with no
    spaces (CJK / long paths) so a single token can't overflow."""
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        while _display_width(word) > cols:  # token alone exceeds the line
            head = ""
            for ch in word:
                if _display_width(head + ch) > cols:
                    break
                head += ch
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(head)
            word = word[len(head) :]
        if not cur:
            cur = word
        elif _display_width(cur) + 1 + _display_width(word) <= cols:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def wrap_tooltip(text: str | None) -> str | None:
    """Wrap a long, plain tooltip to multiple display-bounded lines.

    Returns *text* unchanged when it is empty, already short, already
    multi-line, or already rich text (an author-formatted tooltip is left
    alone). Otherwise returns plain text with explicit newlines wrapped at
    :data:`TOOLTIP_WRAP_COLS` display columns — Qt shows it as a multi-line
    box with no right-edge clipping."""
    if not text or "\n" in text or len(text) <= _TOOLTIP_WRAP_MIN_LEN:
        return text
    if _LOOKS_RICH.search(text):  # author-formatted tooltip — leave it alone
        return text
    return _wrap_plain(text, TOOLTIP_WRAP_COLS)


def make_field_label(
    text: str,
    *,
    style: str | None = None,
    tooltip: str | None = None,
    on_click=None,
) -> ClickableLabel:
    """Build a form field's :class:`ClickableLabel` (style + tooltip + click wire).

    Centralises the label half of the per-row form boilerplate the tabs share;
    each caller still builds its own editor widget and supplies its own click
    handler (a zero-arg callable, usually a lambda closing over the field key).
    """
    lbl = ClickableLabel(text)
    if style:
        lbl.setStyleSheet(style)
    if tooltip:
        lbl.setToolTip(tooltip)
    if on_click is not None:
        lbl.clicked.connect(on_click)
    return lbl


def hint_label(text: str = "") -> QLabel:
    """A secondary/hint :class:`QLabel` in the theme's dim text color.

    Replaces the recurring ``QLabel(...)`` + ``setStyleSheet(f"color:{tok('text_dim')}")``
    pair scattered across the tabs. Reads the token at build time, so it follows
    the active theme (call sites are rebuilt on a theme switch)."""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{tok('text_dim')};")
    return lbl
