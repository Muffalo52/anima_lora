"""Invariants for the CJK Phase 2a glossary/corpus builders.

The subtle part of `project/cjk_aware_anima/datasets/` is deciding *which
string is Japanese* and *which alternate wording is safe to swap in* — both
silently corrupt the corpus when wrong, and neither shows up as a crash. These
guard the two rules that were established by measurement:

* script detection has to reject Chinese in both directions (simplified via
  Shift-JIS, traditional via a JA-kanji inventory), because a Chinese string
  scored a *perfect* back-translation and would otherwise have been chosen;
* the alternate-wording pool for general tags must come from verified
  candidates, never from the raw wiki list, which carries 女スパイ under
  ``1girl``.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

DATASETS = (
    Path(__file__).resolve().parents[1] / "project" / "cjk_aware_anima" / "datasets"
)


def _load(name: str):
    """Import a `datasets/` script by path — they are entry points, not a package."""
    sys.path.insert(0, str(DATASETS))
    spec = importlib.util.spec_from_file_location(name, DATASETS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve types via sys.modules
    spec.loader.exec_module(module)
    return module


tag_glossary = _load("tag_glossary")
build_pairs = _load("build_pairs")
mt = _load("mt")


def test_translation_cache_survives_a_killed_job(tmp_path):
    """A pass is hours of GPU, so results are appended per batch, not held.

    The last line of a killed job's cache can be torn mid-write; that must cost
    one entry, not the whole file.
    """
    engine = object.__new__(mt.MTEngine)
    engine.model_path, engine.greedy, engine.seed = "m", True, 0
    key = engine._cache_key("hello", "ja", 32)

    cache = tmp_path / "c.jsonl"
    cache.write_text(
        json.dumps({"key": key, "out": "こんにちは"}) + '\n{"torn line\n',
        encoding="utf-8",
    )
    loaded = mt.MTEngine._load_cache(cache)
    assert loaded == {key: "こんにちは"}


def test_cache_key_separates_runs_that_would_differ():
    engine = object.__new__(mt.MTEngine)
    engine.model_path, engine.greedy, engine.seed = "m", True, 0
    base = engine._cache_key("hello", "ja", 32)
    assert engine._cache_key("hello", "ja", 32) == base  # stable
    assert engine._cache_key("hello", "en", 32) != base  # target language
    assert engine._cache_key("other", "ja", 32) != base  # prompt
    engine.greedy = False
    assert engine._cache_key("hello", "ja", 32) != base  # decode mode


@pytest.mark.parametrize(
    "text",
    ["ツインテール", "カメラ目線", "割座", "火宮チナツ", "へそ", "赤面"],
)
def test_japanese_accepted(text):
    assert tag_glossary.is_japanese(text)


@pytest.mark.parametrize(
    "text", ["小鸟游星野", "双马尾", "多色发", "阿黑颜", "看向观众"]
)
def test_simplified_chinese_rejected(text):
    """Shift-JIS cannot encode simplified forms — the zero-dependency filter."""
    assert not tag_glossary.is_japanese(text)


def test_ja_kanji_inventory_rejects_traditional_chinese(tmp_path):
    """Traditional Chinese survives Shift-JIS, so the inventory has to catch it.

    棕毛 and 藍眼睛 both encode fine *and* back-translated perfectly ("brown
    hair" / "blue eyes"); only the fact that their characters never appear in
    kana-bearing Japanese entries separates them from 茶髪.
    """
    dump = tmp_path / "wiki.jsonl"
    dump.write_text(
        "\n".join(
            [
                '{"other_names": ["茶髪ロング", "黒髪ボブ", "目線カメラ"]}',
                '{"other_names": ["髪型のこと", "茶色い髪の毛"]}',
                '{"other_names": ["棕毛", "藍眼睛"]}',
            ]
        ),
        encoding="utf-8",
    )
    inventory = tag_glossary.ja_kanji_inventory(dump, min_count=1)
    assert "髪" in inventory  # seen inside kana-bearing entries
    assert "棕" not in inventory  # only ever seen in a Han-only entry
    assert "睛" not in inventory


def test_back_translation_f1_prefers_exact_recovery():
    assert tag_glossary._f1("blush", "Blush") == pytest.approx(1.0)
    assert tag_glossary._f1("blue eyes", "Blue eyes") == pytest.approx(1.0)
    # extra content is penalised — this is what demotes 黒髪ボブ under `black hair`
    assert tag_glossary._f1("black hair", "Black Bob") < 0.75
    assert tag_glossary._f1("solo", "Alone") == 0.0


def test_alt_pool_uses_verified_candidates_for_general_tags():
    entry = {
        "axis": "general",
        "ja": "貧乳",
        "alts": ["女スパイ", "ガキ巨乳"],  # raw wiki junk, must not be swapped in
        "candidates": [
            {"ja": "貧乳", "f1": 1.0, "kana": False},  # the chosen primary
            {"ja": "ちっぱい", "f1": 1.0, "kana": True},
            {"ja": "極上の貧乳", "f1": 0.4, "kana": False},  # below the gate
        ],
    }
    pool = build_pairs.alt_pool(entry, min_f1=0.75)
    assert pool == ["貧乳", "ちっぱい"]


def test_alt_pool_drops_verified_chinese():
    """Verified is not proof of Japanese — 汗液/短袖 back-translate perfectly.

    They were being swapped into the `tags_alt` register, so the pool demands
    kana unless the candidate is the primary wording itself.
    """
    entry = {
        "axis": "general",
        "ja": "汗",
        "candidates": [
            {"ja": "汗", "f1": 1.0, "kana": False},
            {"ja": "汗水", "f1": 1.0, "kana": False},
            {"ja": "汗液", "f1": 1.0, "kana": False},
        ],
    }
    assert build_pairs.alt_pool(entry, min_f1=0.75) == ["汗"]


def test_alt_pool_keeps_name_variants_for_proper_nouns():
    """Character nicknames are real alternates users type — no F1 gate there."""
    entry = {"axis": "character", "alts": ["ホシノ", "暁のホルス"]}
    assert build_pairs.alt_pool(entry, min_f1=0.75) == ["ホシノ", "暁のホルス"]


def test_compose_passes_unmapped_segments_through_and_counts_them():
    glossary = {
        "1girl": {"axis": "general", "ja": "女の子1人", "candidates": []},
        "@aak": {"axis": "artist", "ja": "@aak"},
    }
    ja, missing = build_pairs.compose(
        ["1girl", "@aak", "unmapped tag"],
        glossary,
        alt=False,
        rng=random.Random(0),
        min_f1=0.75,
    )
    assert ja == ["女の子1人", "@aak", "unmapped tag"]
    assert missing == ["unmapped tag"]
