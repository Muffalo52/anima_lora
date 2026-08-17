from __future__ import annotations


def test_preprocess_te_uses_corrected_resized_captions(monkeypatch):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        values = {
            "source_image_dir": "image_dataset",
            "resized_image_dir": "post_image_dataset/resized",
            "lora_cache_dir": "post_image_dataset/lora",
        }
        return values.get(key, default)

    monkeypatch.setattr(preprocess, "run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(preprocess, "_path", fake_path)

    preprocess.cmd_preprocess_te(
        ["--min_pixels", "500000", "--path_pattern", "group/*"],
        caption_config={
            "correct_order": True,
            "insert_no_artist": True,
            "trigger_word": "@dataset-trigger",
            "trigger_at_front": False,
        },
    )

    assert len(calls) == 2
    caption_cmd, te_cmd = calls

    assert caption_cmd[:2] == [
        preprocess.PY,
        "scripts/preprocess/correct_captions.py",
    ]
    assert caption_cmd[caption_cmd.index("--src") + 1] == "image_dataset"
    assert caption_cmd[caption_cmd.index("--dst") + 1] == "post_image_dataset/resized"
    assert caption_cmd[caption_cmd.index("--path_pattern") + 1] == "group/*"
    assert "--caption_insert_no_artist" in caption_cmd
    assert caption_cmd[caption_cmd.index("--caption_trigger_word") + 1] == (
        "@dataset-trigger"
    )

    assert te_cmd[:2] == [
        preprocess.PY,
        "scripts/preprocess/cache_text_embeddings.py",
    ]
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"
    assert "--match_images_from" not in te_cmd
    assert te_cmd[te_cmd.index("--cache_dir") + 1] == "post_image_dataset/lora"
    assert te_cmd[te_cmd.index("--path_pattern") + 1] == "group/*"
    assert [i for i, arg in enumerate(te_cmd) if arg == "--min_pixels"] == [
        te_cmd.index("--min_pixels")
    ]
    assert te_cmd[te_cmd.index("--min_pixels") + 1] == "0"


def test_caption_correction_enabled_when_only_trigger_or_no_artist_set():
    from scripts.tasks.preprocess import _caption_correction_enabled

    base = {
        "correct_order": False,
        "insert_no_artist": False,
        "trigger_word": "",
        "trigger_at_front": False,
    }
    # Nothing set → no rewrite pass.
    assert not _caption_correction_enabled(base)
    # Trigger word alone must still run the pass (the bug: it was being dropped).
    assert _caption_correction_enabled({**base, "trigger_word": "@trig"})
    # Whitespace-only trigger does not count.
    assert not _caption_correction_enabled({**base, "trigger_word": "   "})
    # Insert-no-artist alone also requires the pass.
    assert _caption_correction_enabled({**base, "insert_no_artist": True})


def test_preprocess_te_runs_correction_for_trigger_word_without_correct_order(
    monkeypatch,
):
    from scripts.tasks import preprocess

    calls: list[list[str]] = []

    def fake_path(key: str, default: str) -> str:
        values = {
            "source_image_dir": "image_dataset",
            "resized_image_dir": "post_image_dataset/resized",
            "lora_cache_dir": "post_image_dataset/lora",
        }
        return values.get(key, default)

    monkeypatch.setattr(preprocess, "run", lambda cmd: calls.append(cmd))
    monkeypatch.setattr(preprocess, "_path", fake_path)

    preprocess.cmd_preprocess_te(
        [],
        caption_config={
            "correct_order": False,
            "insert_no_artist": False,
            "trigger_word": "@dataset-trigger",
            "trigger_at_front": False,
        },
    )

    # Correction pass runs (trigger injected) even though correct_order is off,
    # and the TE cache then reads the corrected captions from the resized dir.
    assert len(calls) == 2
    caption_cmd, te_cmd = calls
    assert caption_cmd[1] == "scripts/preprocess/correct_captions.py"
    assert caption_cmd[caption_cmd.index("--caption_trigger_word") + 1] == (
        "@dataset-trigger"
    )
    assert te_cmd[te_cmd.index("--dir") + 1] == "post_image_dataset/resized"


def _stub_overrides(monkeypatch, overrides: dict) -> None:
    """Pin the merged-config read both builders fall back to when env is unset."""
    from scripts.tasks import _common

    monkeypatch.setattr(_common, "_path_overrides", lambda: dict(overrides))


def test_min_pixels_args_env_drop_false_keeps_every_image(monkeypatch):
    """GUI auto-chain unchecks low-res → DROP_LOWRES_IMAGES=0 forces --min_pixels 0,
    overriding a merged config that still says drop=true (the snapshot strips it)."""
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {"drop_lowres_images": True, "min_pixels": 250_000})
    monkeypatch.setenv("DROP_LOWRES_IMAGES", "0")
    monkeypatch.setenv("MIN_PIXELS", "250000")

    assert _min_pixels_args() == ["--min_pixels", "0"]


def test_min_pixels_args_env_drop_true_uses_env_threshold(monkeypatch):
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {})
    monkeypatch.setenv("DROP_LOWRES_IMAGES", "1")
    monkeypatch.setenv("MIN_PIXELS", "250000")

    assert _min_pixels_args() == ["--min_pixels", "250000"]


def test_min_pixels_args_no_env_falls_back_to_config(monkeypatch):
    from scripts.tasks.preprocess import _min_pixels_args

    _stub_overrides(monkeypatch, {"drop_lowres_images": False, "min_pixels": 250_000})
    monkeypatch.delenv("DROP_LOWRES_IMAGES", raising=False)
    monkeypatch.delenv("MIN_PIXELS", raising=False)

    assert _min_pixels_args() == ["--min_pixels", "0"]


def test_target_res_args_env_wins_over_config(monkeypatch):
    from scripts.tasks.preprocess import _target_res_args

    _stub_overrides(monkeypatch, {"target_res": [1024]})
    monkeypatch.setenv("TARGET_RES", "1024 896")

    assert _target_res_args([]) == ["--target_res", "1024", "896"]
    # An explicit CLI --target_res still wins over both env and config.
    assert _target_res_args(["--target_res", "768"]) == []


def test_caption_correction_config_parses_trigger_cli_args():
    from scripts.tasks.preprocess import _caption_correction_config

    config, cleaned = _caption_correction_config(
        [
            "--caption_trigger_word",
            "@foo",
            "--caption_trigger_at_front",
            "--other",
        ]
    )

    assert config["trigger_word"] == "@foo"
    assert config["trigger_at_front"] is True
    assert cleaned == ["--other"]


def test_caption_position_clauses_is_not_a_correction_flag():
    """It gates its own stage — it must never reach ``correct_captions.py``.

    ``position_clauses`` rides in the caption-correction config dict (same
    family of caption-master rewrites, one GUI box), but enabling it alone must
    not turn the correction pass on or add an argv flag it doesn't know.
    """
    from scripts.tasks.preprocess import (
        _caption_correction_args,
        _caption_correction_config,
        _caption_correction_enabled,
    )

    config, cleaned = _caption_correction_config(
        ["--caption_position_clauses", "--other"]
    )

    assert config["position_clauses"] is True
    assert cleaned == ["--other"]
    assert _caption_correction_enabled(config) is False
    assert _caption_correction_args(config) == []

    off, _ = _caption_correction_config(["--no_caption_position_clauses"])
    assert off["position_clauses"] is False


def test_preprocess_chains_position_clauses_before_the_caption_step(monkeypatch):
    """The stage rewrites the caption master, so it must land before TE.

    Ordering is the whole contract: the caption/TE steps read the master and
    write the variant sidecars, so a position pass that ran after them would be
    encoded only on the *next* preprocess. It also has to run inline (plain
    ``run``) — this process is itself a daemon job on a serial queue.
    """
    from scripts.tasks import preprocess

    order: list[str] = []

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_resize", lambda *_a, **_k: order.append("resize")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_vae", lambda *_a, **_k: order.append("vae")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_te", lambda *_a, **_k: order.append("te")
    )
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        order.append("position")

    monkeypatch.setattr(preprocess, "run", fake_run)
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess([])

    assert order == ["resize", "vae", "position", "te"]
    (cmd,) = calls
    assert cmd[:2] == [preprocess.PY, "scripts/preprocess/position_captions.py"]
    assert cmd[-1] == "--apply"


def test_preprocess_skips_position_clauses_when_unset(monkeypatch):
    from scripts.tasks import preprocess

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    for name in ("cmd_preprocess_resize", "cmd_preprocess_vae", "cmd_preprocess_te"):
        monkeypatch.setattr(preprocess, name, lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []
    monkeypatch.setattr(preprocess, "run", lambda cmd, **_k: calls.append(cmd))
    monkeypatch.delenv("CAPTION_POSITION_CLAUSES", raising=False)
    monkeypatch.delenv("CAPTION_AUTOTAG", raising=False)

    preprocess.cmd_preprocess([])

    assert calls == []


def test_caption_autotag_is_not_a_correction_flag():
    """Like ``position_clauses``: own stage, must never reach correct_captions.py."""
    from scripts.tasks.preprocess import (
        _caption_correction_args,
        _caption_correction_config,
        _caption_correction_enabled,
    )

    config, cleaned = _caption_correction_config(
        ["--caption_autotag", "--caption_autotag_mode", "merge", "--other"]
    )

    assert config["autotag"] is True
    assert config["autotag_mode"] == "merge"
    assert cleaned == ["--other"]
    assert _caption_correction_enabled(config) is False
    assert _caption_correction_args(config) == []

    off, _ = _caption_correction_config(["--no_caption_autotag"])
    assert off["autotag"] is False
    # Mode defaults to the non-destructive one when nothing selects it.
    assert off["autotag_mode"] == "missing"


def test_caption_autotag_rejects_an_unknown_mode():
    """Fail at argv-parse time, not minutes into a GPU job."""
    import pytest

    from scripts.tasks.preprocess import _caption_correction_config

    with pytest.raises(SystemExit):
        _caption_correction_config(["--caption_autotag_mode", "clobber"])


def test_caption_autotag_args_always_apply():
    """In-pipeline the user already opted in; a dry run there writes nothing."""
    from scripts.tasks.preprocess import _caption_autotag_args

    assert _caption_autotag_args({"autotag_mode": "missing"}) == [
        "--mode",
        "missing",
        "--apply",
    ]
    # A zero floor is left off entirely so the tagger's own thresholds rule.
    assert "--min_confidence" not in _caption_autotag_args(
        {"autotag_mode": "merge", "autotag_min_confidence": 0.0}
    )
    assert _caption_autotag_args(
        {"autotag_mode": "merge", "autotag_min_confidence": 0.35}
    ) == ["--mode", "merge", "--min_confidence", "0.35", "--apply"]


def test_preprocess_chains_autotag_first(monkeypatch):
    """Autotag *creates* the captions every later stage reads, so it goes first.

    Ordering is the contract: position clauses append to the master, correction
    and TE read it. An autotag that ran after any of them would leave this run
    encoding the un-tagged caption.
    """
    from scripts.tasks import preprocess

    order: list[str] = []

    monkeypatch.setattr(preprocess, "_path", lambda key, default: default)
    monkeypatch.setattr(preprocess, "_repa_pe_encoder", lambda: None)
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_resize", lambda *_a, **_k: order.append("resize")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_vae", lambda *_a, **_k: order.append("vae")
    )
    monkeypatch.setattr(
        preprocess, "cmd_preprocess_te", lambda *_a, **_k: order.append("te")
    )
    monkeypatch.setattr(preprocess, "cmd_caption_index", lambda *_a, **_k: None)
    monkeypatch.setattr(preprocess.os.path, "exists", lambda _p: True)

    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        order.append("autotag" if "autotag_captions.py" in cmd[1] else "position")

    monkeypatch.setattr(preprocess, "run", fake_run)
    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MODE", "merge")
    monkeypatch.setenv("CAPTION_POSITION_CLAUSES", "1")

    preprocess.cmd_preprocess([])

    assert order == ["resize", "autotag", "vae", "position", "te"]
    autotag_cmd = calls[0]
    assert autotag_cmd[:2] == [
        preprocess.PY,
        "scripts/preprocess/autotag_captions.py",
    ]
    assert autotag_cmd[-3:] == ["--mode", "merge", "--apply"]


def test_preprocess_autotag_blank_env_confidence_is_zero(monkeypatch):
    """The GUI writes ``""`` for an empty field — that must not raise."""
    from scripts.tasks.preprocess import _caption_correction_config

    monkeypatch.setenv("CAPTION_AUTOTAG", "1")
    monkeypatch.setenv("CAPTION_AUTOTAG_MIN_CONFIDENCE", "")

    config, _ = _caption_correction_config([])
    assert config["autotag_min_confidence"] == 0.0


def test_sigma_demote_routes_true_is_the_certified_route(monkeypatch):
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {"sigma_demote": True})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    assert _sigma_demote_routes([]) == ["1024:896"]


def test_sigma_demote_routes_off_by_default(monkeypatch):
    from scripts.tasks.preprocess import _sigma_demote_routes

    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    for value in ({}, {"sigma_demote": False}, {"sigma_demote": "off"}):
        _stub_overrides(monkeypatch, value)
        assert _sigma_demote_routes([]) == []


def test_sigma_demote_routes_comma_list_feeds_the_stacked_router(monkeypatch):
    """The stacked router (--sigma_lowres_route2) needs BOTH routes' sibling
    keys; one comma-listed config value must emit one pass per route, in
    order, deduplicated."""
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {})
    monkeypatch.setenv("SIGMA_DEMOTE", "1024:896, 1024:768 ,1024:896")
    assert _sigma_demote_routes([]) == ["1024:896", "1024:768"]


def test_sigma_demote_routes_skips_malformed_entries(monkeypatch):
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {})
    monkeypatch.setenv("SIGMA_DEMOTE", "1024:896,nonsense,")
    assert _sigma_demote_routes([]) == ["1024:896"]


def test_sigma_demote_routes_never_chains_from_a_demote_run(monkeypatch):
    """An explicit --sigma_demote means this invocation IS the demote pass."""
    from scripts.tasks.preprocess import _sigma_demote_routes

    _stub_overrides(monkeypatch, {"sigma_demote": "1024:896,1024:768"})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    assert _sigma_demote_routes(["--sigma_demote", "1024:768"]) == []


def _demote_passes(monkeypatch, extra) -> list[str]:
    """Run cmd_preprocess_demote with the subprocess stubbed; return the routes
    each `cache_latents.py` invocation actually received."""
    from scripts.tasks import preprocess as pp

    seen: list[str] = []

    def fake_run(cmd):
        assert "--sigma_demote" in cmd
        seen.append(cmd[cmd.index("--sigma_demote") + 1])

    monkeypatch.setattr(pp, "run", fake_run)
    monkeypatch.setattr(pp, "_path", lambda key, default="": default)
    pp.cmd_preprocess_demote(list(extra))
    return seen


def test_preprocess_demote_target_emits_every_configured_route(monkeypatch):
    """`make preprocess-demote` must honor the SAME comma list as the automatic
    chain — otherwise the stacked router's deep route silently degrades because
    only 1024:896 ever landed on disk."""
    _stub_overrides(monkeypatch, {"sigma_demote": "1024:896,1024:768"})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    assert _demote_passes(monkeypatch, []) == ["1024:896", "1024:768"]


def test_preprocess_demote_target_falls_back_to_the_certified_route(monkeypatch):
    """Nothing configured → the target still does its historical single pass."""
    _stub_overrides(monkeypatch, {})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    assert _demote_passes(monkeypatch, []) == ["1024:896"]


def test_preprocess_demote_args_override_and_split(monkeypatch):
    """ARGS wins over the config, and a comma list there is expanded too —
    cache_latents.py parses one route per invocation."""
    _stub_overrides(monkeypatch, {"sigma_demote": "1024:896"})
    monkeypatch.delenv("SIGMA_DEMOTE", raising=False)
    monkeypatch.delenv("PREPROCESS_PATH_PATTERN", raising=False)

    assert _demote_passes(monkeypatch, ["--sigma_demote", "1280:1024, 1024:768"]) == [
        "1280:1024",
        "1024:768",
    ]
