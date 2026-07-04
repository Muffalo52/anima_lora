"""The multi-scale ``target_res`` tier checkbox row."""

from __future__ import annotations

import functools

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from gui.i18n import t


@functools.cache
def _target_res_tiers() -> tuple[tuple[int, ...], dict[int, int]]:
    """``(allowed tiers, {edge: max token count})`` sourced from
    ``library.datasets.buckets`` (the single source of truth) instead of a
    hardcoded mirror. Lazy + cached so importing ``widgets`` stays cheap; the
    bucket module is torch-free so this never drags the training stack in.

    A tier is "dangerous" (extra compiled block graph + VRAM) when its per-image
    token count exceeds the canonical 1024 tier — which reproduces the previous
    ``{1280: 6300, 1536: 8640}`` flag set, but recomputed from the tables.
    """
    from library.datasets.buckets import ALLOWED_TARGET_RES, EDGE_TOKEN_BANDS

    # A tier's max token count is the high end of its free-fit band.
    max_tok = {edge: hi for edge, (lo, hi) in EDGE_TOKEN_BANDS.items()}
    canonical = max_tok.get(1024, 4200)
    danger = {edge: tok for edge, tok in max_tok.items() if tok > canonical}
    return tuple(ALLOWED_TARGET_RES), danger


@functools.cache
def _target_res_buckets() -> dict[int, tuple[tuple[int, int], ...]]:
    # Free-fit (the only resize mode) preserves native aspect inside each tier's
    # token band — there is no discrete bucket allow-list to choose from, so the
    # per-tier bucket popup is empty. Kept as a stub so the widget API is intact.
    return {}


class _BucketMenuPanel(QWidget):
    """Keep a bucket popup open when the user misses a checkbox row slightly."""

    def mousePressEvent(self, event) -> None:  # noqa: N802
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        event.accept()


class _TargetResWidget(QWidget):
    """Horizontal row of tier checkboxes for the multi-scale ``target_res`` knob.

    Reads/writes a list of edge ints (e.g. ``[1024, 1536]``). Never returns an
    empty list — unchecking everything falls back to ``[1024]`` (the legacy
    single ~1MP tier) so preprocess/train always have a valid tier.

    The optional bucket checklist is a narrow preprocess-only filter. Empty
    selection means "all buckets in the selected tiers".

    The 1280/1536 tiers are visually flagged as "dangerous" (high token count
    + extra compile graph / VRAM) via colour + an i18n tooltip.
    """

    changed = Signal()

    def __init__(self, selected) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        sel = {int(e) for e in selected} if selected else set()
        tiers, danger = _target_res_tiers()
        self._boxes: dict[int, QCheckBox] = {}
        self._bucket_boxes: dict[tuple[int, int], QCheckBox] = {}
        self._explicit_bucket_selection = False
        for edge in tiers:
            edge_box = QWidget()
            edge_lay = QHBoxLayout(edge_box)
            edge_lay.setContentsMargins(0, 0, 0, 0)
            edge_lay.setSpacing(2)
            cb = QCheckBox()
            cb.setChecked(edge in sel)
            cb.toggled.connect(self._on_tier_toggled)
            if edge in danger:
                cb.setStyleSheet("QCheckBox { color: #d9822b; font-weight: bold; }")
                cb.setToolTip(
                    t(
                        "target_res_danger_tooltip",
                        edge=edge,
                        tokens=danger[edge],
                    )
                )
            edge_lay.addWidget(cb)
            edge_buckets = _target_res_buckets().get(edge, ())
            if edge_buckets:
                # Free-fit has no discrete allow-list, so this popup is normally
                # empty and skipped; kept for a possible future snap-style filter.
                toggle = QToolButton()
                toggle.setText(str(edge))
                toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                toggle.setPopupMode(QToolButton.InstantPopup)
                popup = QMenu(toggle)
                panel = _BucketMenuPanel(popup)
                panel_lay = QVBoxLayout(panel)
                panel_lay.setContentsMargins(8, 4, 8, 4)
                panel_lay.setSpacing(0)
                for width, height in edge_buckets:
                    ratio = width / height
                    cb_bucket = QCheckBox(f"{width}x{height} ({ratio:.2f})")
                    cb_bucket.setMinimumHeight(22)
                    cb_bucket.setToolTip(t("target_res_bucket_tooltip", edge=edge))
                    cb_bucket.toggled.connect(self._on_bucket_toggled)
                    panel_lay.addWidget(cb_bucket)
                    self._bucket_boxes[(width, height)] = cb_bucket
                action = QWidgetAction(popup)
                action.setDefaultWidget(panel)
                popup.addAction(action)
                toggle.setMenu(popup)
                if edge in danger:
                    toggle.setStyleSheet(
                        "QToolButton { color: #d9822b; font-weight: bold; }"
                    )
                    toggle.setToolTip(
                        t("target_res_danger_tooltip", edge=edge, tokens=danger[edge])
                    )
                edge_lay.addWidget(toggle)
            else:
                # No popup: show the tier number as a plain label next to its box.
                label = QLabel(str(edge))
                if edge in danger:
                    label.setStyleSheet("QLabel { color: #d9822b; font-weight: bold; }")
                    label.setToolTip(
                        t("target_res_danger_tooltip", edge=edge, tokens=danger[edge])
                    )
                edge_lay.addWidget(label)
            lay.addWidget(edge_box)
            self._boxes[edge] = cb
        lay.addStretch(1)
        self.set_bucket_resos([])

    def value(self) -> list[int]:
        out = [e for e, cb in self._boxes.items() if cb.isChecked()]
        return out or [1024]

    def bucket_resos(self) -> list[str]:
        if not self._explicit_bucket_selection:
            return []
        selected_edges = set(self.value())
        allowed = set()
        for edge in selected_edges:
            allowed.update(_target_res_buckets().get(edge, ()))
        out = [
            f"{width}x{height}"
            for (width, height), cb in self._bucket_boxes.items()
            if cb.isChecked() and (width, height) in allowed
        ]
        return out

    def set_bucket_resos(self, values) -> None:
        selected = set()
        if isinstance(values, str):
            selected = {
                part.strip().lower().replace("×", "x")
                for part in values.split(",")
                if part.strip()
            }
        elif values:
            selected = {
                str(value).strip().lower().replace("×", "x") for value in values
            }
        self._explicit_bucket_selection = bool(selected)
        for (width, height), cb in self._bucket_boxes.items():
            cb.blockSignals(True)
            cb.setChecked((not selected) or f"{width}x{height}" in selected)
            cb.blockSignals(False)
        self._refresh_bucket_enabled()

    def _refresh_bucket_enabled(self) -> None:
        selected_edges = set(self.value())
        allowed = set()
        for edge in selected_edges:
            allowed.update(_target_res_buckets().get(edge, ()))
        for bucket, cb in self._bucket_boxes.items():
            cb.setEnabled(bucket in allowed)

    def refresh_bucket_enabled(self) -> None:
        self._refresh_bucket_enabled()

    def _on_tier_toggled(self) -> None:
        self._refresh_bucket_enabled()
        self.changed.emit()

    def _on_bucket_toggled(self) -> None:
        checked = [cb.isChecked() for cb in self._bucket_boxes.values()]
        if not any(checked):
            sender = self.sender()
            if isinstance(sender, QCheckBox):
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
            return
        self._explicit_bucket_selection = not all(checked)
        self.changed.emit()
