# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
import numpy as np
import pandas as pd
from PIL import Image

from skinaudit.dedup import load_hashes, near_pairs, save_hashes
from skinaudit.io import image_props, image_stats, resolution_groups


def test_resolution_groups_ignore_orientation():
    w = pd.Series([4032, 3024, 4032, 640, 10])
    h = pd.Series([3024, 4032, 3024, 480, 5])
    out = resolution_groups(w, h, min_count=2).tolist()
    assert out[:3] == ["4032x3024"] * 3
    assert out[3] == "other" and out[4] == "other"


def test_image_props_and_stats(tmp_path):
    paths = []
    for i, (size, colour) in enumerate([((300, 200), (200, 150, 120)), ((200, 300), (20, 20, 20))]):
        p = tmp_path / f"im{i}.jpg"
        Image.new("RGB", size, colour).save(p)
        paths.append(str(p))
    s = pd.Series(paths)
    props = image_props(s)
    assert props[["width", "height"]].values.tolist() == [[300, 200], [200, 300]]
    st = image_stats(s, size=32)
    assert st.shape[0] == 2 and not st.isna().any().any()
    assert st.loc[1, "dark_corner_share"] == 1.0 and st.loc[0, "dark_corner_share"] == 0.0


def test_hash_roundtrip_and_near_pairs(tmp_path):
    h = pd.DataFrame({"image_id": ["a", "b", "c"], "group_id": ["g1", "g2", "g1"],
                      "phash": np.array([2 ** 64 - 1, 2 ** 64 - 2, 0], dtype=np.uint64)})
    save_hashes(h, tmp_path / "h.csv")
    back = load_hashes(tmp_path / "h.csv")
    assert (back["phash"].to_numpy() == h["phash"].to_numpy()).all()
    pairs = near_pairs(back, max_dist=1)
    assert len(pairs) == 1 and bool(pairs.loc[0, "crosses_groups"])
