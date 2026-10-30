# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Human-confirmed duplicates: group merging, review figure and the audit run."""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import make_frame, make_spec
from skinaudit.audit import run_audit
from skinaudit.dedup import load_confirmed, merge_duplicate_groups, review_figure
from skinaudit.schema import AuditDataset


def test_merge_is_transitive_and_keeps_rows():
    df = pd.DataFrame({"image_id": list("abcdef"), "group_id": ["P1", "P2", "P3", "P4", "P5", "P5"]})
    pairs = pd.DataFrame({"image_id_1": ["a", "b", "e"], "image_id_2": ["b", "c", "f"]})
    g, merged = merge_duplicate_groups(df, pairs)
    assert g.tolist() == ["P1+P2+P3"] * 3 + ["P4", "P5", "P5"]
    assert merged == [["P1", "P2", "P3"]]


def test_unknown_image_stops():
    df = pd.DataFrame({"image_id": ["a"], "group_id": ["P1"]})
    with pytest.raises(AssertionError):
        merge_duplicate_groups(df, pd.DataFrame({"image_id_1": ["a"], "image_id_2": ["zz"]}))


def test_load_confirmed_needs_reason(tmp_path):
    pd.DataFrame({"image_id_1": ["a"], "image_id_2": ["b"], "reason": [None]}).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(AssertionError):
        load_confirmed(str(tmp_path / "c.csv"))


def test_run_audit_merges_groups_and_review_figure(tmp_path):
    df = make_frame(n_patients=60, seed=9)
    paths = []
    for i, r in df.iterrows():
        p = tmp_path / f"{r.image_id}.jpg"
        Image.new("RGB", (40, 30), (int(i) % 200, 90, 90)).save(p)
        paths.append(str(p))
    df["image_path"], df["file_bytes"] = paths, [os.path.getsize(p) for p in paths]
    a = df.drop_duplicates("group_id").iloc[0]
    b = df.drop_duplicates("group_id").iloc[1]
    pairs = pd.DataFrame({"image_id_1": [a.image_id], "image_id_2": [b.image_id], "reason": ["same photo"],
                          "hamming": [0]})
    files = review_figure(pairs, df, str(tmp_path / "review.png"))
    assert files == [str(tmp_path / "review_p01.png")] and os.path.getsize(files[0]) > 1000
    ds = AuditDataset(make_spec(), df.drop(columns=["image_path", "file_bytes"]))
    s = run_audit(ds, str(tmp_path / "o"), n_perm=1, n_boot=5, image_stats=False, make_figures=False, n_jobs=1,
                  duplicate_pairs=pairs)
    assert s["duplicates"]["groups_after"] == s["duplicates"]["groups_before"] - 1
    assert s["structure"]["n_groups"] == s["duplicates"]["groups_after"]
    assert (tmp_path / "o" / "tables" / "confirmed_duplicates.csv").exists()


def test_pairs_outside_subset_are_skipped(tmp_path):
    df = make_frame(n_patients=40, seed=10)
    ids = df.drop_duplicates("group_id")["image_id"].tolist()
    pairs = pd.DataFrame({"image_id_1": [ids[0], "ISIC_not_here"], "image_id_2": [ids[1], ids[2]],
                          "reason": ["same photo", "other image type"]})
    ds = AuditDataset(make_spec(), df)
    s = run_audit(ds, str(tmp_path / "o"), n_perm=1, n_boot=5, image_stats=False, make_figures=False, n_jobs=1,
                  duplicate_pairs=pairs)
    assert s["duplicates"]["confirmed_pairs"] == 1 and s["duplicates"]["pairs_not_in_this_data"] == 1


def test_duplicate_lesions_become_one_permutation_unit(tmp_path):
    df = make_frame(n_patients=40, seed=12)
    first = df.drop_duplicates("group_id")
    a, b = first.iloc[0], first.iloc[1]
    df.loc[df["lesion_id"] == b.lesion_id, "label"] = a.label      # a duplicate pair shares the label
    pairs = pd.DataFrame({"image_id_1": [a.image_id], "image_id_2": [b.image_id], "reason": ["same photo"]})
    s = run_audit(AuditDataset(make_spec(), df), str(tmp_path / "o"), n_perm=1, n_boot=5, image_stats=False,
                  make_figures=False, n_jobs=1, duplicate_pairs=pairs)
    n_units = s["shortcut_cv"]["A_missing_indicators"]["n_perm_units"]
    assert n_units == df["lesion_id"].nunique() - 1


def test_review_figure_pages(tmp_path):
    rows = []
    for i in range(14):
        p = tmp_path / f"r{i}.jpg"
        Image.new("RGB", (20, 16), (i * 10, 50, 50)).save(p)
        rows.append({"image_id": f"r{i}", "image_path": str(p), "group_id": f"g{i}", "label": "NV"})
    df = pd.DataFrame(rows)
    pairs = pd.DataFrame({"image_id_1": [f"r{2*k}" for k in range(7)], "image_id_2": [f"r{2*k+1}" for k in range(7)],
                          "hamming": 1})
    files = review_figure(pairs, df, str(tmp_path / "rev.jpg"), per_page=3)
    assert [os.path.basename(f) for f in files] == ["rev_p01.jpg", "rev_p02.jpg", "rev_p03.jpg"]


def test_describe_pairs_flags_identical_files(tmp_path):
    import shutil
    from skinaudit.dedup import describe_pairs, identical_pairs
    rows = []
    for i, col in enumerate([(10, 20, 30), (200, 20, 30), (10, 20, 30)]):
        p = tmp_path / f"x{i}.png"
        Image.new("RGB", (20, 16), col).save(p)
        rows.append({"image_id": f"x{i}", "image_path": str(p), "group_id": f"g{i}", "label": ["MEL", "NV", "NV"][i]})
    shutil.copy(rows[0]["image_path"], tmp_path / "x3.png")
    rows.append({"image_id": "x3", "image_path": str(tmp_path / "x3.png"), "group_id": "g3", "label": "NV"})
    df = pd.DataFrame(rows)
    pairs = pd.DataFrame({"image_id_1": ["x0", "x0"], "image_id_2": ["x1", "x3"], "hamming": [0, 2]})
    d = describe_pairs(pairs, df)
    assert d.loc[0, "image_id_2"] == "x3" and bool(d.loc[0, "identical_file"])      # exact copy first
    assert not bool(d.loc[1, "identical_file"])
    ip = identical_pairs(d)
    assert len(ip) == 1 and "conflicting labels MEL vs NV" in ip.loc[0, "reason"]


def _photo(path, seed, size=(120, 90), quality=None):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (9, 12, 3)).astype(np.uint8)
    im = Image.fromarray(base).resize(size, Image.BILINEAR)
    im.save(path, quality=quality) if quality else im.save(path)
    return str(path)


def test_pixel_corr_separates_resaved_photo_from_other_photo(tmp_path):
    from skinaudit.dedup import describe_pairs, likely_same_photo
    a = _photo(tmp_path / "a.png", 1)
    with Image.open(a) as im:                       # the same photo saved again, smaller and as JPEG
        im.resize((80, 60)).save(tmp_path / "b.jpg", quality=70)
    c = _photo(tmp_path / "c.png", 2)
    df = pd.DataFrame({"image_id": ["a", "b", "c"], "image_path": [a, str(tmp_path / "b.jpg"), c],
                       "group_id": ["g1", "g2", "g3"], "label": ["NV", "NV", "NV"]})
    pairs = pd.DataFrame({"image_id_1": ["a", "a"], "image_id_2": ["b", "c"], "hamming": [0, 0]})
    d = describe_pairs(pairs, df).set_index("image_id_2")
    assert d.loc["b", "pixel_corr"] > 0.97 > d.loc["c", "pixel_corr"]
    lk = likely_same_photo(d.reset_index())
    assert lk["image_id_2"].tolist() == ["b"] and lk.loc[0, "reason"].startswith("same photo")


def test_duplicate_drops_keeps_most_complete_copy():
    from skinaudit.dedup import duplicate_drops
    df = pd.DataFrame({"image_id": ["a", "b", "c", "d", "e"], "group_id": ["image:a", "P1", "P2", "P3", "P4"],
                       "label": ["MEL", "MEL", "MEL", "NV", "MEL"], "age": [np.nan, 50, 60, 40, 30]})
    pairs = pd.DataFrame({"image_id_1": ["a", "b", "d"], "image_id_2": ["b", "c", "e"], "reason": ["x"] * 3})
    drops, clusters = duplicate_drops(df, pairs, ["age"])
    # cluster a-b-c: a lacks age, b and c complete -> keep b (smaller id); cluster d-e: labels differ -> both go
    assert drops == {"a", "c", "d", "e"}
    assert sorted(c["kept"] or "-" for c in clusters) == ["-", "b"]


def test_run_audit_drop_mode(tmp_path):
    from conftest import make_frame, make_spec
    df = make_frame(n_patients=60, seed=5)
    a, b = df["image_id"].iloc[0], df[df["patient_id"] != df["patient_id"].iloc[0]]["image_id"].iloc[0]
    df.loc[df["image_id"] == b, "label"] = df.loc[df["image_id"] == a, "label"].iloc[0]
    df.loc[df["image_id"] == b, "lesion_id"] = df.loc[df["image_id"] == b, "lesion_id"] + "_dup"
    pairs = pd.DataFrame({"image_id_1": [a], "image_id_2": [b], "reason": ["same photo"]})
    s = run_audit(AuditDataset(make_spec(), df, {}), str(tmp_path), n_perm=2, n_boot=0, make_figures=False,
                  image_stats=False, n_jobs=1, mechanism=False, n_repeats=1, duplicate_pairs=pairs,
                  duplicate_action="drop")
    assert s["duplicates"]["action"] == "drop" and s["duplicates"]["images_dropped"] == 1
    assert s["structure"]["n_images"] == len(df) - 1


def _circle_photo(path, seed, size=(300, 225)):
    """A lesion-like texture inside a black dermoscope circle, as in many ISIC uploads."""
    rng = np.random.default_rng(seed)
    w, h = size
    tex = Image.fromarray(rng.integers(60, 255, (12, 16, 3)).astype(np.uint8)).resize(size, Image.BILINEAR)
    a = np.asarray(tex).copy()
    yy, xx = np.mgrid[:h, :w]
    a[(xx - w / 2) ** 2 + (yy - h / 2) ** 2 > (0.45 * h) ** 2] = 0
    Image.fromarray(a).save(path)
    return str(path)


def test_pixel_corr_is_not_fooled_by_a_dermoscope_circle(tmp_path):
    from skinaudit.dedup import describe_pairs
    a = _circle_photo(tmp_path / "a.png", 1)
    b = _circle_photo(tmp_path / "b.png", 2)                       # another lesion, same frame
    with Image.open(a) as im:                                       # the same photo, smaller, JPEG, brighter
        Image.eval(im.resize((200, 150)), lambda v: min(255, int(v * 1.15) + 5)).save(tmp_path / "c.jpg", quality=70)
    df = pd.DataFrame({"image_id": ["a", "b", "c"], "image_path": [a, b, str(tmp_path / "c.jpg")],
                       "group_id": ["g1", "g2", "g3"], "label": ["NV", "MEL", "NV"]})
    pairs = pd.DataFrame({"image_id_1": ["a", "a"], "image_id_2": ["b", "c"], "hamming": [4, 2]})
    d = describe_pairs(pairs, df).set_index("image_id_2")
    assert d.loc["c", "pixel_corr"] > 0.97 and d.loc["b", "pixel_corr"] < 0.8


def test_cross_pairs_match_brute_force():
    """The pigeonhole search finds exactly the pairs the all-pairs search finds, including
    near-copies (a few flipped bits) and many identical low-entropy hashes."""
    import numpy as np
    import pandas as pd
    from skinaudit.dedup import cross_pairs, near_pairs, overlap_table
    rng = np.random.default_rng(3)
    base = rng.integers(0, 2 ** 63, 300, dtype=np.uint64)
    flips = [np.uint64(sum(1 << int(b) for b in rng.choice(64, k, replace=False))) for k in rng.integers(0, 7, 200)]
    other = np.r_[base[:200] ^ np.array(flips, dtype=np.uint64), rng.integers(0, 2 ** 63, 200, dtype=np.uint64),
                  np.zeros(20, np.uint64)]
    h1 = pd.DataFrame({"image_id": [f"a{i}" for i in range(300)], "group_id": "g", "phash": np.r_[base[:280], np.zeros(20, np.uint64)]})
    h2 = pd.DataFrame({"image_id": [f"b{i}" for i in range(len(other))], "group_id": "g", "phash": other})
    for d in (0, 2, 4, 6):
        fast = cross_pairs(h1, h2, max_dist=d)
        slow = near_pairs(h1, h2, max_dist=d)
        assert set(zip(fast.image_id_1, fast.image_id_2, fast.hamming)) == set(zip(slow.image_id_1, slow.image_id_2, slow.hamming))
    h3 = h1.assign(image_id=[f"a{i}" if i < 10 else f"c{i}" for i in range(300)])
    summary, pairs = overlap_table({"one": h1, "two": h2, "three": h3})
    row = summary[(summary.dataset_1 == "one") & (summary.dataset_2 == "three")].iloc[0]
    assert row["shared_ids"] == 10 and row["phash_pairs"] >= 300 and set(pairs.columns) >= {"same_id", "hamming"}


def test_heatmap_marks_tiny_shares(tmp_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    from skinaudit import plots
    texts = []
    orig = plt.Axes.text
    def spy(self, x, y, s, *a, **k):
        texts.append(s)
        return orig(self, x, y, s, *a, **k)
    plt.Axes.text = spy
    try:
        plots.heatmap(pd.DataFrame({"a": [0.001, 0.997], "b": [0.0, 0.5]}), tmp_path / "h.png", "t", "s", tiny_text="<1%")
    finally:
        plt.Axes.text = orig
    assert "<1%" in texts and ">99%" in texts and "0%" in texts and "50%" in texts
