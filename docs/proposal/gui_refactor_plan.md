# gui/ refactor — module layout catch-up

Status: **IN PROGRESS — Phases 1, 2, and 4 done (2026-07-04); Phase 3 not started.**

Scope: file organization only. The architecture (Qt-free `config_io.py`/`_paths.py`,
the three shared mixins, theme tokens, daemon-observer model) is good and is
**not** being changed. What's lagging is the module layout: a few files became
junk drawers that absorb every new feature. This plan splits them along seams
that already exist, with zero behavior change intended.

Audit snapshot (2026-07-04): `gui/` is ~18.7k lines / 35 py files. The offenders:

| File | Lines | Problem |
|---|---|---|
| `tabs/image_tab.py` | 2312 | one class (`ImageViewerTab`, 104 methods) holding ≥5 separable subsystems |
| `tabs/preprocess_tab.py` | 1887 | one main class + loose module-level loaders |
| `tabs/config_tab.py` | 1783 | `ConfigTab` (65 methods) + `SplitButtonStyle` + the "rich" job observer |
| `widgets.py` | 1302 | five modules in one file (mixins / field factory / domain widgets / buttons / image view) |
| `app.py` | 862 | `MainWindow` + `SettingsDialog` + `GuidebookDialog` + MCP helpers + font tweaks |
| `system_dialog.py` | 737 | models dialog + update dialog + 2 QThreads + a private `gui_settings.json` accessor pair |

Cross-cutting smell: `gui_settings.json` is opened/parsed by **four independent
implementations** — `_paths.py:38` (the canonical one, with `get_setting`/`set_setting`),
`system_dialog.py:393-402`, `i18n/__init__.py:24`, `tabs/preprocess_tab.py:220`
(plus a duplicate `SETTINGS_FILE` path constant at `tabs/preprocess_tab.py:103`).

## Non-goals

- **No observer unification.** `DaemonJobMixin`'s tail-and-poll loop vs the
  ConfigTab/EasyControl "richer" observer (progress.jsonl + sample preview +
  preprocess→train auto-chain) is a documented, intentional split
  (`gui/CLAUDE.md` §Architecture). Merging them is real risk for marginal gain;
  revisit only if a bug lands in both.
- **No public-name renames.** `_widget`/`_read` are underscore-named but are
  cross-module API (imported by `gui/__init__.py` and, via the package root, by
  `tabs/distill_tab.py:44`). Renaming (`_widget` → `build_field_widget`, …) is a
  cheap optional follow-up **after** the split, not part of it.
- **No tab-architecture changes.** Mixin composition order, LazyTabMixin
  semantics, and the daemon contract stay exactly as documented.

## Phase 1 — one owner for `gui_settings.json` (~30 min, lowest risk) — DONE

`system_dialog.py`, `i18n/__init__.py`, and `tabs/preprocess_tab.py` now go
through `_paths.get_setting`/`set_setting`/`read_gui_settings` (the latter
promoted from the private `_read_gui_settings`) instead of their own
private file/read/write copies. `grep -rn "gui_settings.json" gui/
--include="*.py"` shows path construction in `_paths.py` only (the remaining
hits are comments/docstrings).

`_paths.py` is already the intended owner: dependency-free foundation module,
`GUI_SETTINGS_FILE` + `_read_gui_settings()` + `get_setting()` + `set_setting()`.
The fix is migration, not new code:

1. `system_dialog.py` — delete `_GUI_SETTINGS_FILE` (line 377) and its private
   `_read_gui_settings`/`_write_gui_settings` (393–407); the update-tag cache
   (`_load_cached_latest_tag`/`_save_cached_latest_tag`) becomes
   `get_setting(_UPDATE_CACHE_KEY)` / `set_setting(...)`. Note the entry is a
   nested dict `{tag, ts}` — `get_setting`/`set_setting` handle opaque values
   fine, keep the TTL logic where it is.
2. `i18n/__init__.py` — replace `_settings_path()` + inline json read/write
   (lines 24–60) with `_paths` imports. No cycle risk: `_paths` imports nothing
   from `gui` (its docstring guarantees it), and i18n already resolves the same
   file path by hand.
3. `tabs/preprocess_tab.py` — delete `SETTINGS_FILE` (line 103) and
   `_load_settings()` (line 220); the two call sites (532, 891) read whole-dict
   today, so either use `_paths._read_gui_settings()` directly (promote it to
   `read_gui_settings`) or fetch the specific keys via `get_setting`.

Acceptance: `grep -rn "gui_settings.json" gui/ --include="*.py"` shows path
construction in `_paths.py` only.

## Phase 2 — explode `widgets.py` into `gui/widgets/` (mechanical, ~1–2 h) — DONE

Landed as planned, with one addition not called out in the original table:
`_no_wheel` couldn't live only in `fields.py` as written, because
`sample_prompts.py`'s `_SamplePromptRow` also calls it and `fields.py` already
imports `target_res.py`/`sample_prompts.py` for the `_widget`/`_read` dispatch
— `sample_prompts.py` importing back from `fields.py` would cycle. Added a
tiny leaf `gui/widgets/_qt_utils.py` holding just `_no_wheel`, imported by both
`fields.py` and `sample_prompts.py`; `SplitButtonStyle` moved into
`buttons.py` as planned, with `tabs/config_tab.py` re-exporting it (`from
gui.widgets import SplitButtonStyle`) since `tabs/preprocess_tab.py` imports
it from there. All acceptance checks passed: every old `widgets.py` public
name resolves via `gui.widgets`, `import gui.app` still shows no torch, and
`tests/test_gui_*.py`/`test_doc_refs.py` pass (the launch-speed budget test
is flaky under machine load — reproduces identically on unmodified `main`,
not a regression from this split).

Split along the five concern boundaries already visible in the file. New
package, **`gui/widgets/__init__.py` re-exports every current name** so all 11
importers (`__init__`, `app`, `tensorboard`, `system_dialog`, and 7 tabs) keep
working unchanged — `from gui.widgets import action_button` is identical
whether `widgets` is a module or a package.

| New module | Contents (current widgets.py lines) |
|---|---|
| `widgets/mixins.py` | `LazyTabMixin` (51), `DirtyTrackingMixin` (1040) |
| `widgets/fields.py` | `_widget` (770), `_read` (834), `_no_wheel` (761), `ClickableLabel` (877), `make_field_label` (957), `hint_label` (1029) + text utils `_display_width`/`_wrap_plain`/`wrap_tooltip` (908–942) |
| `widgets/buttons.py` | `action_button` (997), `apply_variant` (980); **also move `SplitButtonStyle` here from `tabs/config_tab.py:116`** — it's a shared button style, not config-tab logic (theme.py's QSS docs already describe it as a widgets-level concern) |
| `widgets/target_res.py` | `_TargetResWidget` (114), `_BucketMenuPanel` (104), `_target_res_tiers`/`_target_res_buckets` (77–97) |
| `widgets/sample_prompts.py` | `_SamplePromptRow` (269), `_SamplePromptsWidget` (486), `SamplePromptsDialog` (695), `_SamplePromptsLauncher` (715), `_normalize_prompt_lines` (683) |
| `widgets/image_view.py` | `ScaledImageLabel` (1122), `ImageViewerDialog` (1268) |

Invariants to preserve per-module (they're why widgets.py is layered where it
is): Qt-only, **no `gui.daemon` import** (cycle-freedom), no `library/` imports
beyond the torch-free leaves already used. `fields.py` ← `_widget` dispatches
to `target_res.py`/`sample_prompts.py` widgets by key; keep that a one-way
import (fields → domain widgets), never the reverse.

Acceptance: `python -X importtime -c "import gui.app"` still shows no torch;
`tests/test_gui_launch_speed.py` passes; every name in the old file resolves
via `gui.widgets`.

## Phase 3 — carve the god-class tabs (the real lift, do incrementally)

Continue the pattern that already worked: `tabs/_caption_editor.py` and
`tabs/_image_overlays.py` were split out of the image tab as sibling modules.

### 3a. `tabs/image_tab.py` (2312 → target <900)

`ImageViewerTab`'s 104 methods cluster cleanly:

- **`tabs/_autotag.py`** — DONE. The tagger worker lifecycle extracted as
  `_AutotagWorker(QObject)`: owns the `QProcess`, the stdio sentinel protocol,
  and the idle/GPU-watch timers behind four signals (`status`/`busy`/`result`/
  `error`) — no widget references, so it stays UI-free. `ImageViewerTab` keeps
  only the thin reactive slots (`_run_autotag`, `_on_autotag_status`,
  `_on_autotag_result`, `_on_autotag_error`) that translate + apply the
  signals against its own widgets (`autotag_btn`/`autotag_status`/`cap`).
  Status phases are plain strings (`STATUS_LOADING`/`STATUS_RUNNING`/
  `STATUS_READY`/`""`) so `_autotag.py` never imports `gui.i18n` — the tab maps
  phase → translated text. `image_tab.py` 2312 → 2173 lines.
- **`tabs/_caption_kb.py`** — caption correction + knowledge base + tag
  completion: `_caption_correction_options`, `_load_caption_kb`,
  `_describe_kb`, `_on_tag_clicked`, `_start_tag_completion_preload`,
  `reload_tag_knowledge_base`, `_preload_tag_completion`,
  `_correct_current_caption`, `_correct_visible_captions` (~752–992).
- **`tabs/_dataset_tree.py`** — tree build / grouping / sort / filter:
  `_rebuild_tree*`, `_ensure_tree_folder`, `_ensure_group_node`,
  `_float_groups_to_top`, `_apply_filter_and_sort`, `_group_sort_key`,
  `_load_groups`/`_rebuild_groups` (~543–1456).
- **`tabs/_curation.py`** — preprocess-decision persistence:
  `_curation_decisions_path`, `_load/_save_preprocess_decisions`,
  `_set/_clear_current_preprocess_decision`, `_clear_all_decisions`,
  `_toggle_mark_current`, `_mark_current_for_move` (~1458–1901).
- Image display/prefetch (`_cache_pixmap`, `_prefetch_neighbors`, `_show`,
  `_set_image`, resize-preview meta) stays — that *is* the image tab.

Each extraction is its own commit with a GUI smoke-run between (see
Verification). Don't do all four in one pass.

### 3b. `tabs/config_tab.py` (1783 → target <1200)

- `SplitButtonStyle` → `widgets/buttons.py` (done in Phase 2).
- **`tabs/_run_observer.py`** — the rich observer (progress.jsonl reader +
  live sample preview + preprocess→train auto-chain) that ConfigTab owns and
  EasyControlTab inherits. Extracting it does **not** merge it with
  `DaemonJobMixin` (see Non-goals) — it just gives the second observer its own
  file so both are findable.

### 3c. `tabs/preprocess_tab.py` (1887)

Lower priority — it's big but has visible internal structure already
(`_RuleCard`, `_ResizeCropAnchorWidget`, module-level loaders). The
module-level `_load_preprocess_toml`/`_load_sam_yaml`/`_load_rules` +
`_count_*` helpers could move to a Qt-free `tabs/_preprocess_io.py` (making
them unit-testable like `config_io`), but only bother when next touching the
tab.

## Phase 4 — slim `app.py` (~1 h) — DONE

`app.py` now keeps `MainWindow`, `main()`, `_dark()`, `_ensure_source_image_dir`
(862 → 404 lines). Moved out exactly as planned:

- `SettingsDialog` + `_mcp_paths`/`_mcp_add_command`/`_mcp_json_config` →
  `gui/settings_dialog.py` (new module).
- `GuidebookDialog` + `_guidebook_path` → `gui/dialogs.py` (existing module).
- `_prefer_cleartype_font_engine` → `theme.py` next to `_load_bundled_fonts`;
  `app.py::main()` calls it as `gui_theme._prefer_cleartype_font_engine()`
  (must still run before `QApplication` is constructed, so it stays a
  standalone function rather than folding into `apply_theme`).

`tabs/easycontrol_tab.py::_open_adapter_guide` imported `GuidebookDialog` from
`gui.app` (a lazy import to dodge a circular dependency, since `gui.app`
imports the tab); updated to import from `gui.dialogs` instead — and the
circularity concern is now moot since `dialogs.py` doesn't import `tabs/`.
All acceptance checks passed: `import gui.app` still shows no torch,
`tests/test_gui_*.py`/`test_doc_refs.py` pass (launch-speed budget test still
flaky under machine load exactly as on unmodified `main` — not a regression).

## Ordering & verification

Order: 1 → 2 → 4 → 3 (3 is open-ended; everything before it is bounded and
mostly mechanical). Each phase is a standalone commit; stop-anywhere-safe.

Per-phase checks:

1. `uv run pytest tests/test_gui_*.py tests/test_doc_refs.py` — the `test_gui_*`
   suite covers variants/validation-split/preprocess-tab/jsonl-progress/launch
   speed headlessly; `test_doc_refs` catches doc references to moved files.
2. `python -X importtime -c "import gui.app"` — torch must not appear
   (the CLAUDE.md startup contract; also enforced by `test_gui_launch_speed`).
3. `make gui` smoke: open each tab, flip Hardware preset, dirty+Save a variant,
   submit a no-op preprocess job (daemon round-trip), open the image tab on
   `image_dataset/`.
4. `ruff check <touched files> --fix && ruff format <touched files>` —
   touched files only.

Docs debt to settle in the same PR as each phase: `gui/CLAUDE.md` line counts
and module map are already stale (says 29 files / config_tab 1654 /
preprocess_tab 1366; actual 35 / 1783 / 1887) — update its Architecture +
Common-changes sections as modules move, and re-point the `translator` agent
surfaces only if `i18n/` paths change (they don't in this plan).

## Optional follow-ups (explicitly out of scope)

- Rename `_widget`/`_read` to public names once importers are stable.
- `system_dialog.py` split (models vs update dialogs) — same pattern as
  Phase 4, do when next touched.
- Observer unification — only with a concrete motivating bug.
