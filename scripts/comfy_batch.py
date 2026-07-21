#!/usr/bin/env python3
"""Submit a ComfyUI workflow as a batch over the ComfyUI API.

Two modes, auto-selected by the workflow's contents:

* **Image mode** — if the workflow contains a ``LoadImage`` node and
  ``--images_dir`` is given, submit one job per image file in that directory,
  rewriting every ``LoadImage`` node's ``image`` input to point at the current
  file. Jobs are queued sequentially (``--wait``), so the server keeps the model
  + block-compile graph warm across the whole folder. Used by
  ``make comfy-batch W=colorize.json``.

* **Artist×chara mode** — otherwise, read ``__artist__`` / ``__chara__``
  placeholders and submit one job per cartesian-product pair. With
  ``--prompts``, the prompt text itself becomes a third axis: every line of the
  file replaces the positive ``CLIPTextEncode`` text before substitution, so
  prompt × artist × chara pairs are queued.

Usage:
    python scripts/comfy_batch.py workflows/colorize.json \
        --images_dir ../comfy/input/to_colorize
    python scripts/comfy_batch.py workflows/lora-batch.json \
        --artist workflows/artist.txt --chara workflows/chara.txt
    python scripts/comfy_batch.py workflows/modhydra.json \
        --prompts workflows/preferred.txt
"""

import argparse
import json
import itertools
import os
import random
import time
import urllib.request
import urllib.error


COMFY_URL = "http://localhost:8188"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_prompts(path: str) -> list[str]:
    """Read a prompt-per-line file.

    Lines may be bare text or JSON string literals with a trailing comma (the
    format ``workflows/preferred.txt`` is kept in, so it can be pasted straight
    out of a workflow JSON). ``#`` comment lines are skipped.
    """
    prompts = []
    for line in load_lines(path):
        if line.startswith("#"):
            continue
        stripped = line.rstrip(",").strip()
        if stripped.startswith('"'):
            try:
                stripped = json.loads(stripped)
            except json.JSONDecodeError:
                pass
        stripped = stripped.strip()
        if stripped:
            prompts.append(stripped)
    return prompts


def apply_prompt(workflow: dict, prompt: str) -> None:
    """Overwrite the positive ``CLIPTextEncode`` text in-place.

    The positive node is the one carrying a placeholder (``__prompt__`` or the
    artist/chara markers) — negatives never do.
    """
    for node in workflow.values():
        if node.get("class_type") != "CLIPTextEncode":
            continue
        text = node.get("inputs", {}).get("text")
        if isinstance(text, str) and (
            "__prompt__" in text or "__artist__" in text or "__chara__" in text
        ):
            node["inputs"]["text"] = prompt


def substitute(workflow: dict, artist: str, chara: str) -> dict:
    """Deep-copy workflow and replace __artist__ / __chara__ in all string values."""
    raw = json.dumps(workflow)
    # Escape for JSON string context (backslashes must be doubled)
    artist_esc = artist.replace("\\", "\\\\").replace('"', '\\"')
    chara_esc = chara.replace("\\", "\\\\").replace('"', '\\"')
    raw = raw.replace("__artist__", artist_esc).replace("__chara__", chara_esc)
    return json.loads(raw)


def load_image_nodes(workflow: dict) -> list[dict]:
    """Return every LoadImage node's ``inputs`` dict in the workflow."""
    return [
        node["inputs"]
        for node in workflow.values()
        if node.get("class_type") == "LoadImage" and "inputs" in node
    ]


def list_images(images_dir: str) -> list[str]:
    """Sorted list of image filenames (not paths) directly in ``images_dir``."""
    return sorted(
        f
        for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def comfy_relative(images_dir: str) -> str:
    """Path of ``images_dir`` as ComfyUI's LoadImage expects it.

    LoadImage resolves names against the server's ``input/`` dir, so we strip
    everything up to and including ``.../input/`` and keep the remainder as a
    forward-slash subfolder prefix. If no ``input`` segment is present we fall
    back to the basename (assumes files sit directly under ``input/``).
    """
    norm = os.path.normpath(os.path.abspath(images_dir))
    parts = norm.split(os.sep)
    if "input" in parts:
        after = parts[parts.index("input") + 1 :]
        return "/".join(after)
    return os.path.basename(norm)


def queue_prompt(workflow: dict, server: str) -> dict:
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{server}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def wait_until_done(server: str, prompt_id: str, poll_interval: float = 2.0):
    """Poll /history until the prompt_id appears (i.e. execution finished)."""
    while True:
        try:
            with urllib.request.urlopen(f"{server}/history/{prompt_id}") as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.URLError:
            pass
        time.sleep(poll_interval)


def submit(wf: dict, server: str, randomize_seed: bool, wait: bool, label: str):
    """Optionally reseed, queue one job, and (optionally) wait for it."""
    if randomize_seed:
        for node in wf.values():
            if "seed" in node.get("inputs", {}):
                node["inputs"]["seed"] = random.randint(0, 2**53)

    print(f"{label} ... ", end="", flush=True)
    try:
        result = queue_prompt(wf, server)
        prompt_id = result["prompt_id"]
    except (urllib.error.URLError, KeyError) as e:
        print(f"FAILED to queue: {e}")
        return

    if wait:
        wait_until_done(server, prompt_id)
        print("done")
    else:
        print(f"queued ({prompt_id})")


def run_image_mode(workflow, images_dir, args):
    """One job per image file in ``images_dir`` (sequential)."""
    files = list_images(images_dir)
    if not files:
        print(f"No images found in {images_dir}")
        return
    prefix = comfy_relative(images_dir)
    print(
        f"Queuing {len(files)} images from {images_dir} (LoadImage prefix '{prefix}/')"
    )

    for i, fname in enumerate(files, 1):
        wf = json.loads(json.dumps(workflow))
        rel = f"{prefix}/{fname}" if prefix else fname
        for inputs in load_image_nodes(wf):
            inputs["image"] = rel
        submit(
            wf,
            args.server,
            args.randomize_seed,
            args.wait,
            f"[{i}/{len(files)}] {fname}",
        )

    print("All done.")


def run_artist_chara_mode(workflow, args):
    """One job per (prompt, artist, chara) triple."""
    # An empty/missing list file is an inert axis, not zero jobs.
    artists = load_lines(args.artist) or [""]
    charas = load_lines(args.chara) or [""]
    prompts = load_prompts(args.prompts) if args.prompts else [None]

    combos = list(itertools.product(enumerate(prompts, 1), artists, charas))
    print(
        f"Queuing {len(prompts)} prompts × {len(artists)} artists × "
        f"{len(charas)} charas = {len(combos)} jobs"
    )

    for i, ((pidx, prompt), artist, chara) in enumerate(combos, 1):
        wf = json.loads(json.dumps(workflow))
        if prompt is not None:
            apply_prompt(wf, prompt)
        wf = substitute(wf, artist, chara)
        tag = " x ".join(p for p in (artist, chara) if p)
        if prompt is not None:
            tag = f"prompt{pidx}" + (f" | {tag}" if tag else "")
        submit(
            wf,
            args.server,
            args.randomize_seed,
            args.wait,
            f"[{i}/{len(combos)}] {tag}",
        )

    print("All done.")


def main():
    parser = argparse.ArgumentParser(description="ComfyUI batch runner")
    parser.add_argument("workflow", help="Path to workflow JSON")
    parser.add_argument("--artist", default="workflows/artist.txt")
    parser.add_argument("--chara", default="workflows/chara.txt")
    parser.add_argument(
        "--prompts",
        default=None,
        help=(
            "File of prompts (one per line, bare or JSON-quoted); each replaces "
            "the positive CLIPTextEncode text, e.g. workflows/preferred.txt."
        ),
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Directory of images; one job per image (requires a LoadImage node).",
    )
    parser.add_argument("--server", default=COMFY_URL)
    parser.add_argument(
        "--randomize-seed",
        action="store_true",
        default=True,
        help="Randomize seed per job (default: true)",
    )
    parser.add_argument(
        "--no-randomize-seed", dest="randomize_seed", action="store_false"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for each job to finish before queuing next (default: true)",
    )
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    args = parser.parse_args()

    with open(args.workflow) as f:
        workflow = json.load(f)

    # Image mode when the workflow loads images *and* a dir was provided;
    # otherwise fall back to the artist×chara placeholder batch.
    if args.images_dir and load_image_nodes(workflow):
        run_image_mode(workflow, args.images_dir, args)
    else:
        run_artist_chara_mode(workflow, args)


if __name__ == "__main__":
    main()
