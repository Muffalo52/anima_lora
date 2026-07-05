"""SoupTrainTab — uncond-init ΔW-soup pipeline launcher (experimental).

Soup is a bespoke multi-stage pipeline (``scripts/soup/pipeline.py``), **not** a
``train.py --method`` run: a short ``caption_dropout_rate 1.0`` uncond
inter-train on the artist shard containing the target (reused across targets in
the same shard) → N seeded captioned fine-tunes from that init → an exact ΔW
soup SVD-truncated back to ``network_dim`` → ``output/ckpt/anima_soup_<target>``.

It is driven entirely by CLI args (a required ``--target`` artist plus optional
pool / seed / rank knobs), so unlike the SPD / Turbo distill tabs there is no
config TOML to edit — this tab is a small argument form, not a
``_DistillConfigTab``.

Submission mirrors ``scripts/tasks/training.py::cmd_soup`` but submits the
pipeline module *directly* as one daemon command job
(``python -m scripts.soup.pipeline --target … --preset …``) rather than
``tasks.py soup``. Routing through ``tasks.py`` would re-enter ``run_command``
*inside* the daemon and nest-submit a second command job, deadlocking the serial
queue; the pipeline instead runs its ``train.py`` phases as direct subprocesses
within this single job.
"""

from __future__ import annotations

import shlex

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import ROOT, LazyTabMixin
from gui import daemon as gui_daemon
from gui.explanations import method_overview
from gui.i18n import t
from gui._job_mixin import DaemonJobMixin
from gui.progress import TqdmProgressTracker, make_progress_bar
from gui.theme import tok
from gui.widgets import action_button, apply_variant, make_field_label


class SoupTrainTab(DaemonJobMixin, LazyTabMixin, QWidget):
    """Argument form + daemon launcher for the uncond-init soup pipeline."""

    # Reused as the daemon job label and the re-attach match key.
    TRAIN_TASK = "soup"
    METHOD_LABEL = "Soup"

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        # Exposed so MethodsTab can mount its Method picker inline at the front.
        self._top_bar = top
        title = QLabel(self.METHOD_LABEL)
        title.setStyleSheet("font-weight:bold;font-size:14px;")
        top.addWidget(title)
        path_lbl = QLabel("scripts/soup/pipeline.py")
        path_lbl.setStyleSheet(f"color:{tok('text_dim')};")
        top.addWidget(path_lbl)
        top.addStretch()

        self.train_btn = action_button(
            t("train"), variant="primary", on_click=self._start_train
        )
        top.addWidget(self.train_btn)
        self.stop_btn = action_button(t("stop"), variant="danger", on_click=self._stop)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)
        lay.addLayout(top)

        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        form_host = QWidget()
        form = QFormLayout(form_host)
        self._build_form(form)
        sc.setWidget(form_host)
        hsplit.addWidget(sc)

        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            "QTextBrowser { font-size: 120%; padding: 12px; "
            f"background: {tok('panel')}; color: {tok('text')}; }}"
        )
        self._explain.setMinimumWidth(300)
        self._explain.setHtml(method_overview("soup") or "")
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([640, 360])
        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("log_placeholder"))
        vsplit.addWidget(self.log)
        vsplit.setSizes([500, 200])
        lay.addWidget(vsplit)

        # Command job → no progress.jsonl; tqdm parsing drives the bar.
        self._init_job_observer()

    def _build_form(self, form: QFormLayout) -> None:
        def row(key: str, widget: QWidget) -> None:
            tip = t(f"{key}_tip")
            widget.setToolTip(tip)
            lbl = make_field_label(
                t(key),
                style="text-decoration: underline dotted; color:#ddd;",
                tooltip=tip,
            )
            form.addRow(lbl, widget)

        # Required target — an editable combo pre-filled with the artist dirs
        # that actually exist under resized/, so the common case is a click.
        self.target = QComboBox()
        self.target.setEditable(True)
        self.target.addItem("")
        resized = ROOT / "post_image_dataset" / "resized"
        if resized.exists():
            self.target.addItems(
                sorted(p.name for p in resized.iterdir() if p.is_dir())
            )
        row("soup_target", self.target)

        self.pool = QLineEdit()
        self.pool.setPlaceholderText("(blank = shard containing the target)")
        row("soup_pool", self.pool)

        self.pool_shard_n = QSpinBox()
        self.pool_shard_n.setRange(1, 512)
        self.pool_shard_n.setValue(16)
        row("soup_pool_shard_n", self.pool_shard_n)

        self.uncond_ratio = QDoubleSpinBox()
        self.uncond_ratio.setRange(0.0, 1.0)
        self.uncond_ratio.setSingleStep(0.05)
        self.uncond_ratio.setValue(0.5)
        row("soup_uncond_ratio", self.uncond_ratio)

        self.uncond_epochs = QSpinBox()
        self.uncond_epochs.setRange(0, 100)
        self.uncond_epochs.setValue(2)
        row("soup_uncond_epochs", self.uncond_epochs)

        self.seeds = QLineEdit("1001 1002 1003")
        row("soup_seeds", self.seeds)

        self.rank = QSpinBox()
        self.rank.setRange(0, 1024)
        self.rank.setValue(0)  # 0 → omit --rank → pipeline uses method network_dim
        row("soup_rank", self.rank)

        self.ft_args = QLineEdit()
        self.ft_args.setPlaceholderText("--network_dim 32 --max_train_epochs 8")
        row("soup_ft_args", self.ft_args)

    def _lazy_init(self) -> None:
        self._try_reattach()

    # ------------------------------------------------------------------ launch
    def _build_argv(self) -> list[str] | None:
        """Mirror ``cmd_soup``'s pipeline invocation from the form. None if the
        required target is missing (a warning is shown)."""
        target = self.target.currentText().strip()
        if not target:
            QMessageBox.warning(self, t("error"), t("soup_target_required"))
            return None
        argv = [
            "-m",
            "scripts.soup.pipeline",
            "--target",
            target,
            "--preset",
            "default",
            "--pool_shard_n",
            str(self.pool_shard_n.value()),
            "--uncond_ratio",
            str(self.uncond_ratio.value()),
            "--uncond_epochs",
            str(self.uncond_epochs.value()),
        ]
        pool = self.pool.text().split()
        if pool:
            argv += ["--pool", *pool]
        seeds = self.seeds.text().split()
        if seeds:
            argv += ["--seeds", *seeds]
        if self.rank.value() > 0:
            argv += ["--rank", str(self.rank.value())]
        # Unknown args forward to the Phase-2 fine-tunes (make soup ARGS="…").
        try:
            argv += shlex.split(self.ft_args.text())
        except ValueError as e:
            QMessageBox.warning(self, t("error"), str(e))
            return None
        return argv

    def _start_train(self) -> None:
        if self._job_id:
            QMessageBox.information(self, "", t("distill_job_running"))
            return
        argv = self._build_argv()
        if argv is None:
            return
        self._set_busy()
        self.log.clear()
        self.progress.setVisible(True)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_submitting") + "\n")
        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(label=self.TRAIN_TASK, argv=argv),
            on_fail=self._restore_idle,
        )
        if not job_id:
            return
        self._log(t("daemon_queued", job_id=job_id))
        self._attach(job_id, replay=False)

    def _try_reattach(self) -> None:
        """Re-bind to our own soup job still running from a previous session."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "command":
            return
        if gui_daemon.read_job_label(job_id) != self.TRAIN_TASK:
            return
        self.log.clear()
        self.progress.setVisible(True)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_reattached", job_id=job_id))
        self._attach(job_id, replay=True)

    def _attach(self, job_id: str, *, replay: bool) -> None:
        self._set_busy()
        self.stop_btn.setEnabled(True)
        self._watch_job(job_id, replay_log=replay)

    def _emit_log_line(self, line: str) -> None:
        # _log uses insertPlainText (no auto-newline), so add one here.
        self._log(line + "\n")

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        self._drain_job_stdout()
        if self._stdout_buf:
            self._log(self._stdout_buf + "\n")
        self._stdout_buf = ""
        job_id = self._job_id
        self._job_id = None
        self._stdout_tailer.reset()
        self.progress.setVisible(False)
        self._log("\n" + gui_daemon.format_finish_banner(job_id, state) + "\n")
        self._restore_idle()

    def _stop(self) -> None:
        self._stop_job()

    def cleanup_subprocess(self) -> None:
        """App-shutdown hook. Stops observing but leaves the daemon job alive —
        it runs detached so the pipeline survives the GUI closing."""
        self._job_timer.stop()

    def _set_busy(self) -> None:
        self.train_btn.setText(t("train") + " ...")
        apply_variant(self.train_btn, "busy")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _restore_idle(self) -> None:
        self.train_btn.setText(t("train"))
        apply_variant(self.train_btn, "primary")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)
