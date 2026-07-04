"""App settings dialog: language / prefs / MCP server registration."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from gui import (
    DEFAULT_AUTOTAG_CONFIDENCE,
    DEFAULT_GROUP_CELL_MATCH_MIN,
    DEFAULT_GROUP_MATCH_FRAC_MIN,
    get_setting,
    set_setting,
)
from gui import theme as gui_theme
from gui._paths import (
    DEFAULT_CAPTION_INSERT_NO_ARTIST,
    DEFAULT_CAPTION_VALIDATE_ARTIST_TAGS,
    ROOT,
)
from gui.i18n import available_languages, current_language, save_language, t

LANG_NAMES = {"en": "English", "ko": "한국어", "cn": "简体中文", "ja": "日本語"}


def _mcp_paths() -> tuple[Path, Path]:
    """(venv python, MCP bridge script) for THIS checkout — real absolute
    paths, not the <repo> placeholder the docs use (anima_daemon/README.md)."""
    venv_python = (
        ROOT
        / ".venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    return venv_python, ROOT / "anima_daemon" / "mcp.py"


def _mcp_add_command() -> str:
    """The `claude mcp add` one-liner for Claude Code."""

    def q(p: Path) -> str:
        return f'"{p}"' if " " in str(p) else str(p)

    venv_python, bridge = _mcp_paths()
    return f"claude mcp add anima-daemon -- {q(venv_python)} {q(bridge)}"


def _mcp_json_config() -> str:
    """The client-agnostic mcpServers JSON block (Claude Desktop, OpenClaw, …).
    json.dumps so Windows backslashes come out escaped and paste-able."""
    venv_python, bridge = _mcp_paths()
    cfg = {
        "mcpServers": {
            "anima-daemon": {"command": str(venv_python), "args": [str(bridge)]}
        }
    }
    return json.dumps(cfg, indent=2)


class SettingsDialog(QDialog):
    """App settings: language + MCP server registration for agent clients."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(560)
        # Set when the user opts into an immediate reload; MainWindow checks it after exec().
        self.reload_requested = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel(t("language")))
        self.lang_combo = QComboBox()
        for code in available_languages():
            self.lang_combo.addItem(LANG_NAMES.get(code, code), code)
        self.lang_combo.setCurrentIndex(available_languages().index(current_language()))
        self.lang_combo.currentIndexChanged.connect(self._change_lang)
        self.lang_combo.setFixedWidth(120)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        lay.addLayout(lang_row)

        prefs_group = QGroupBox(t("settings_prefs_header"))
        prefs_lay = QVBoxLayout(prefs_group)

        # Autotagger confidence floor (applied on top of the model's per-tag F1
        # thresholds; see AnimaTagger.predict_caption min_confidence).
        conf_row = QHBoxLayout()
        conf_label = QLabel(t("settings_autotag_confidence"))
        conf_label.setToolTip(t("settings_autotag_confidence_tooltip"))
        conf_row.addWidget(conf_label)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setToolTip(t("settings_autotag_confidence_tooltip"))
        self.conf_spin.setValue(
            float(get_setting("autotag_confidence", DEFAULT_AUTOTAG_CONFIDENCE))
        )
        self.conf_spin.valueChanged.connect(
            lambda v: set_setting("autotag_confidence", round(float(v), 2))
        )
        self.conf_spin.setFixedWidth(125)
        conf_row.addWidget(self.conf_spin)
        conf_row.addStretch()
        prefs_lay.addLayout(conf_row)

        self.caption_insert_no_artist = QCheckBox(
            t("settings_caption_insert_no_artist")
        )
        self.caption_insert_no_artist.setToolTip(
            t("settings_caption_insert_no_artist_tooltip")
        )
        self.caption_insert_no_artist.setChecked(
            bool(
                get_setting(
                    "caption_insert_no_artist", DEFAULT_CAPTION_INSERT_NO_ARTIST
                )
            )
        )
        self.caption_insert_no_artist.toggled.connect(
            lambda checked: set_setting("caption_insert_no_artist", bool(checked))
        )
        prefs_lay.addWidget(self.caption_insert_no_artist)

        self.caption_validate_artist_tags = QCheckBox(
            t("settings_caption_validate_artist_tags")
        )
        self.caption_validate_artist_tags.setToolTip(
            t("settings_caption_validate_artist_tags_tooltip")
        )
        self.caption_validate_artist_tags.setChecked(
            bool(
                get_setting(
                    "caption_validate_artist_tags",
                    DEFAULT_CAPTION_VALIDATE_ARTIST_TAGS,
                )
            )
        )
        self.caption_validate_artist_tags.toggled.connect(
            lambda checked: set_setting("caption_validate_artist_tags", bool(checked))
        )
        prefs_lay.addWidget(self.caption_validate_artist_tags)

        # Dataset-tab grouping (`curate-group`) tightness. Higher = tighter,
        # cleaner groups. Read at grouping time by ImageViewerTab._rebuild_groups
        # and passed as --match-frac-min / --cell-match-min.
        frac_row = QHBoxLayout()
        frac_label = QLabel(t("settings_group_match_frac"))
        frac_label.setToolTip(t("settings_group_match_frac_tooltip"))
        frac_row.addWidget(frac_label)
        self.group_frac_spin = QDoubleSpinBox()
        self.group_frac_spin.setRange(0.0, 1.0)
        self.group_frac_spin.setSingleStep(0.05)
        self.group_frac_spin.setDecimals(2)
        self.group_frac_spin.setToolTip(t("settings_group_match_frac_tooltip"))
        self.group_frac_spin.setValue(
            float(get_setting("group_match_frac_min", DEFAULT_GROUP_MATCH_FRAC_MIN))
        )
        self.group_frac_spin.valueChanged.connect(
            lambda v: set_setting("group_match_frac_min", round(float(v), 2))
        )
        self.group_frac_spin.setFixedWidth(125)
        frac_row.addWidget(self.group_frac_spin)
        frac_row.addStretch()
        prefs_lay.addLayout(frac_row)

        cell_row = QHBoxLayout()
        cell_label = QLabel(t("settings_group_cell_match"))
        cell_label.setToolTip(t("settings_group_cell_match_tooltip"))
        cell_row.addWidget(cell_label)
        self.group_cell_spin = QDoubleSpinBox()
        self.group_cell_spin.setRange(0.0, 1.0)
        self.group_cell_spin.setSingleStep(0.01)
        self.group_cell_spin.setDecimals(2)
        self.group_cell_spin.setToolTip(t("settings_group_cell_match_tooltip"))
        self.group_cell_spin.setValue(
            float(get_setting("group_cell_match_min", DEFAULT_GROUP_CELL_MATCH_MIN))
        )
        self.group_cell_spin.valueChanged.connect(
            lambda v: set_setting("group_cell_match_min", round(float(v), 2))
        )
        self.group_cell_spin.setFixedWidth(125)
        cell_row.addWidget(self.group_cell_spin)
        cell_row.addStretch()
        prefs_lay.addLayout(cell_row)

        # Closing the dialog rebuilds the window so each tab's per-widget tokens (gui.theme.tok) repaint.
        theme_row = QHBoxLayout()
        theme_label = QLabel(t("settings_theme"))
        theme_label.setToolTip(t("settings_theme_tooltip"))
        theme_row.addWidget(theme_label)
        self.theme_combo = QComboBox()
        for code in gui_theme.THEME_ORDER:
            self.theme_combo.addItem(t(gui_theme.THEME_LABEL_KEYS[code]), code)
        cur = gui_theme.current_theme_name()
        self.theme_combo.setCurrentIndex(gui_theme.THEME_ORDER.index(cur))
        self.theme_combo.setToolTip(t("settings_theme_tooltip"))
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        self.theme_combo.setFixedWidth(140)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        prefs_lay.addLayout(theme_row)

        # Closing the dialog rebuilds the window so widgets sized to the old metrics relayout cleanly.
        font_row = QHBoxLayout()
        font_label = QLabel(t("settings_font_size"))
        font_label.setToolTip(t("settings_font_size_tooltip"))
        font_row.addWidget(font_label)
        self.font_spin = QSpinBox()
        self.font_spin.setRange(gui_theme.FONT_SIZE_MIN, gui_theme.FONT_SIZE_MAX)
        self.font_spin.setSuffix(" pt")
        self.font_spin.setToolTip(t("settings_font_size_tooltip"))
        self.font_spin.setValue(gui_theme.current_font_size())
        self.font_spin.valueChanged.connect(self._change_font_size)
        self.font_spin.setFixedWidth(125)
        font_row.addWidget(self.font_spin)
        font_row.addStretch()
        prefs_lay.addLayout(font_row)

        # Debug mode: turns on DEBUG-level daemon logging (picked up the next time
        # the daemon starts) so a bug report can carry the launch/queue decisions.
        # Pairs with the "Copy debug report" button below.
        self.debug_check = QCheckBox(t("settings_debug_mode"))
        self.debug_check.setToolTip(t("settings_debug_mode_tooltip"))
        self.debug_check.setChecked(bool(get_setting("debug_mode", False)))
        self.debug_check.toggled.connect(self._toggle_debug)
        prefs_lay.addWidget(self.debug_check)

        dbg_desc = QLabel(t("settings_debug_report_desc"))
        dbg_desc.setWordWrap(True)
        prefs_lay.addWidget(dbg_desc)
        dbg_row = QHBoxLayout()
        dbg_row.addStretch()
        self.debug_report_btn = QPushButton(t("settings_debug_copy_report"))
        self.debug_report_btn.clicked.connect(self._copy_debug_report)
        dbg_row.addWidget(self.debug_report_btn)
        prefs_lay.addLayout(dbg_row)

        lay.addWidget(prefs_group)

        mcp_group = QGroupBox(t("settings_mcp_header"))
        mcp_lay = QVBoxLayout(mcp_group)
        self._add_command_block(
            mcp_lay, t("settings_mcp_desc"), _mcp_add_command(), height=64
        )
        self._add_command_block(
            mcp_lay, t("settings_mcp_desc_json"), _mcp_json_config(), height=140
        )
        lay.addWidget(mcp_group)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        close = QPushButton(t("settings_close"))
        close.clicked.connect(self.close)
        btn_bar.addWidget(close)
        lay.addLayout(btn_bar)

    def _add_command_block(
        self, layout: QVBoxLayout, desc: str, text: str, height: int
    ) -> None:
        """A word-wrapped description, a read-only monospace box, and a copy
        button that flashes confirmation."""
        label = QLabel(desc)
        label.setWordWrap(True)
        layout.addWidget(label)

        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        mono = QFont("Consolas", 9)
        mono.setStyleHint(QFont.Monospace)
        edit.setFont(mono)
        edit.setFixedHeight(height)
        layout.addWidget(edit)

        copy_row = QHBoxLayout()
        copy_row.addStretch()
        btn = QPushButton(t("settings_mcp_copy"))
        btn.clicked.connect(lambda: self._copy(text, btn))
        copy_row.addWidget(btn)
        layout.addLayout(copy_row)

    def _copy(self, text: str, btn: QPushButton) -> None:
        QApplication.clipboard().setText(text)
        btn.setText(t("settings_mcp_copied"))
        QTimer.singleShot(1500, lambda: btn.setText(t("settings_mcp_copy")))

    def _toggle_debug(self, checked: bool) -> None:
        """Persist the flag and reflect it into the live process env. A daemon
        already running keeps its old log level until it restarts (the report
        still works); a newly-spawned one inherits ANIMA_DEBUG from here."""
        set_setting("debug_mode", bool(checked))
        if checked:
            os.environ["ANIMA_DEBUG"] = "1"
        else:
            os.environ.pop("ANIMA_DEBUG", None)

    def _copy_debug_report(self) -> None:
        from gui.debug_report import build_debug_report

        try:
            report = build_debug_report()
        except Exception as exc:  # noqa: BLE001 — diagnostics must never crash the dialog
            report = f"failed to build debug report: {exc}"
        QApplication.clipboard().setText(report)
        self.debug_report_btn.setText(t("settings_mcp_copied"))
        QTimer.singleShot(
            1500,
            lambda: self.debug_report_btn.setText(t("settings_debug_copy_report")),
        )

    def _change_lang(self, idx: int):
        lang = self.lang_combo.itemData(idx)
        # save_language also flips the in-process language, so the prompt below already renders in it.
        save_language(lang)
        choice = QMessageBox.question(
            self,
            t("settings_lang_apply_title"),
            t("settings_lang_apply_question"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if choice == QMessageBox.Yes:
            self.reload_requested = True
            self.accept()

    def _change_theme(self, idx: int) -> None:
        """Persist + apply the chosen theme live, then request a window rebuild.

        ``apply_theme`` restyles app-level chrome immediately; the rebuild on
        close makes per-widget ``tok()`` lookups (log boxes, previews, …) repaint
        too. Unlike a language change there's no confirm prompt — it's cheap and
        reversible."""
        name = self.theme_combo.itemData(idx)
        gui_theme.set_theme(name)
        app = QApplication.instance()
        if app is not None:
            gui_theme.apply_theme(app, name)
        self.reload_requested = True

    def _change_font_size(self, size: int) -> None:
        """Persist + apply the chosen app font size live, then request a rebuild.

        ``apply_theme`` re-reads the size into ``app.setFont``; the rebuild on
        close lets widgets that cached the old font metrics relayout. Like the
        theme, it's cheap and reversible, so there's no confirm prompt."""
        gui_theme.set_font_size(size)
        app = QApplication.instance()
        if app is not None:
            gui_theme.apply_theme(app)
        self.reload_requested = True
