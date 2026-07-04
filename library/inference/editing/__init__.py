"""Image-editing and inversion pipelines (not the core generation engine).

DirectEdit inversion + edit-conditioning swap (``directedit``,
``directedit_splice``, ``edit_dispatcher``). These are experimental techniques
driven by ``scripts/edit.py`` and the comfyui-anima-directedit node, kept out
of the top-level engine namespace. (The postfix-tail inversion probe that used
to live here was archived 2026-07-04 → ``_archive/postfix/inversion_probe/``.)
"""
