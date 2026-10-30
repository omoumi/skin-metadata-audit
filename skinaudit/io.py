# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""File helpers: checksums, image-header properties and cheap image statistics."""
from __future__ import annotations

import glob
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(name: str, root: str = "/kaggle/input") -> tuple[str, int]:
    """Locate a file by name under root; fail if several different copies exist."""
    hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True), key=len)
    assert hits, f"{name} not found under {root}"
    assert len({md5(p) for p in hits}) == 1, f"several different copies of {name}: {hits}"
    return hits[0], len(hits)


def index_images(root: str, exts=("jpg", "jpeg", "png")) -> tuple[dict, int]:
    """Map file stem -> path for all images under root. Returns (index, n_duplicate_stems)."""
    paths = [p for e in exts for p in glob.glob(f"{root}/**/*.{e}", recursive=True)]
    index, dup = {}, 0
    for p in sorted(paths, key=len):
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem in index:
            dup += 1
        else:
            index[stem] = p
    return index, dup


def image_props(paths: pd.Series, workers: int = 32) -> pd.DataFrame:
    """Width, height and file size read from image headers (no full decode)."""
    def one(p):
        with Image.open(p) as im:
            w, h = im.size
        return w, h, os.path.getsize(p)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, paths))
    return pd.DataFrame(rows, columns=["width", "height", "file_bytes"], index=paths.index)


def resolution_groups(width: pd.Series, height: pd.Series, min_count: int = 30) -> pd.Series:
    """'LxS' (long side x short side) for sizes with at least min_count images, otherwise 'other'.

    A device or acquisition protocol often writes one fixed resolution, so the groups
    can serve as a rough proxy for the camera when the metadata does not name it.
    Orientation is ignored: 4032x3024 and 3024x4032 are the same camera held two ways.
    """
    w, h = width.astype(int), height.astype(int)
    res = np.maximum(w, h).astype(str) + "x" + np.minimum(w, h).astype(str)
    counts = res.value_counts()
    return res.where(res.map(counts) >= min_count, "other")


IMAGE_STAT_COLUMNS = ["border_mean_r", "border_mean_g", "border_mean_b", "border_lum_std",
                      "dark_corner_share", "center_to_border_lum", "mean_r", "mean_g", "mean_b", "lum_std"]
BORDER_STAT_COLUMNS = ["border_mean_r", "border_mean_g", "border_mean_b", "border_lum_std", "dark_corner_share"]


def image_stats(paths: pd.Series, size: int = 128, workers: int = 16) -> pd.DataFrame:
    """Colour, border and frame statistics from a size x size downscaled copy of each image.

    Border = outer 10% ring of the image; corners = four squares of 1/8 side.
    Border statistics mostly describe the camera, frame and background rather than the
    lesion (lesions are usually centred), so they are the ones used as a shortcut test; a
    large lesion can still reach the border.
    """
    w = max(1, size // 10)
    c = max(1, size // 8)
    ring = np.ones((size, size), bool)
    ring[w:-w, w:-w] = False
    corners = np.zeros((size, size), bool)
    corners[:c, :c] = corners[:c, -c:] = corners[-c:, :c] = corners[-c:, -c:] = True
    centre = np.zeros((size, size), bool)
    centre[size // 4:-(size // 4), size // 4:-(size // 4)] = True

    def one(p):
        with Image.open(p) as im:
            im.draft("RGB", (size * 2, size * 2))
            a = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
        lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        b = a[ring]
        return (*b.mean(0), float(lum[ring].std()), float((lum[corners] < 30).mean()),
                float(lum[centre].mean() / max(lum[ring].mean(), 1.0)), *a.reshape(-1, 3).mean(0), float(lum.std()))
    def key(p):
        # path + size + file stamp: a changed file is never served from the cache
        try:
            st = os.stat(p)
        except (OSError, TypeError, ValueError):
            return None
        return (str(p), size, st.st_mtime_ns, st.st_size)

    keys = [key(p) for p in paths]
    todo = [i for i, k in enumerate(keys) if k is None or k not in _STATS_CACHE]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fresh = list(ex.map(one, [paths.iloc[i] for i in todo]))
    rows = [None] * len(keys)
    for i, r in zip(todo, fresh):
        rows[i] = r
        if keys[i] is not None:
            _STATS_CACHE[keys[i]] = r
    for i, k in enumerate(keys):
        if rows[i] is None:
            rows[i] = _STATS_CACHE[k]
    return pd.DataFrame(rows, columns=IMAGE_STAT_COLUMNS, index=paths.index).astype(float)


# Image statistics already computed in this Python session, so a second run_audit on a
# subset (e.g. dermoscopic only) does not read the images again.
_STATS_CACHE: dict = {}


def clear_image_stats_cache() -> None:
    _STATS_CACHE.clear()
