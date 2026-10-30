# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
import numpy as np
import pandas as pd

from conftest import make_frame
from skinaudit.shortcut import (category_codes, label_constant_within, make_group_permuter,
                                make_unit_permuter, run_feature_set)


def test_unit_permuter_keeps_clusters_and_label_counts():
    df = make_frame(n_patients=60)
    units, y = df["lesion_id"].to_numpy(), df["label"].to_numpy()
    permute, share = make_unit_permuter(units, y)
    rng = np.random.default_rng(0)
    for _ in range(5):
        yp = permute(y, rng)
        assert label_constant_within(units, yp)
        before = pd.Series(y).groupby(units).first().value_counts().sort_index()
        after = pd.Series(yp).groupby(units).first().value_counts().sort_index()
        assert before.equals(after)
    assert share == 1.0


def test_group_permuter_only_swaps_equal_sizes():
    df = make_frame(n_patients=60)
    g, y = df["group_id"].to_numpy(), df["label"].to_numpy()
    permute, share = make_group_permuter(g)
    yp = permute(y, np.random.default_rng(0))
    sizes = pd.Series(g).map(pd.Series(g).value_counts())
    # rows of groups with a unique size never move
    unique_size = sizes.map(sizes.drop_duplicates().value_counts()).to_numpy() == 1
    grp_sizes = pd.Series(g).drop_duplicates().map(pd.Series(g).value_counts())
    lonely = set(grp_sizes[grp_sizes.map(grp_sizes.value_counts()) == 1])
    fixed = sizes.isin(lonely).to_numpy()
    assert (yp[fixed] == y[fixed]).all()
    assert 0 < share <= 1
    del unique_size


def test_noise_feature_stays_inside_null():
    df = make_frame(n_patients=150, seed=3)
    X = np.random.default_rng(0).normal(size=(len(df), 2))
    r = run_feature_set("noise", X, [False, False], df["label"].to_numpy(), df["group_id"].to_numpy(),
                        n_perm=8, units=df["lesion_id"].to_numpy(), n_perm_reference=8)
    assert r["perm_level"] == "unit"
    assert r["bacc_mean"] - r["null_mean"] < 0.08
    # lesions sit inside patients: the p-value comes from the size-matched group null (its
    # false-positive rate is checked by simulation, not by one draw here)
    assert r["test_null"] == "group" and r["p_perm"] == r["p_perm_group"]


def test_planted_shortcut_is_found():
    df = make_frame(n_patients=150, seed=4)
    y = df["label"].to_numpy()
    leak = category_codes(df["label"]).copy()
    flip = np.random.default_rng(0).random(len(df)) < 0.3
    leak[flip] = np.random.default_rng(1).integers(0, 3, flip.sum())
    r = run_feature_set("leak", leak[:, None], [True], y, df["group_id"].to_numpy(), n_perm=8,
                        units=df["lesion_id"].to_numpy(), n_perm_reference=2)
    assert r["bacc_mean"] - r["null_mean"] > 0.3
    assert r["p_perm"] == 1 / 9


def test_falls_back_to_group_when_label_varies_within_unit():
    df = make_frame(n_patients=80, seed=5)
    r = run_feature_set("x", np.zeros((len(df), 1)), [False], df["label"].to_numpy(), df["group_id"].to_numpy(),
                        n_perm=2, units=df["patient_id"].to_numpy())
    assert r["perm_level"] == "group"


def test_test_null_gets_n_perm_and_reference_gets_the_rest():
    df = make_frame(n_patients=80, seed=6)
    x = np.random.default_rng(0).normal(size=(len(df), 1))
    # lesions inside patients: the size-matched group null is the test
    r = run_feature_set("x", x, [False], df["label"].to_numpy(), df["group_id"].to_numpy(), n_perm=4,
                        units=df["lesion_id"].to_numpy(), n_perm_reference=3)
    assert r["test_null"] == "group" and r["n_perm_group"] == 4 and r["n_perm_unit"] == 3
    # lesions = groups: the unit null is the test (exact there)
    r2 = run_feature_set("x", x, [False], df["label"].to_numpy(), df["lesion_id"].to_numpy(), n_perm=4,
                         units=df["lesion_id"].to_numpy(), n_perm_reference=3)
    assert r2["test_null"] == "unit" and r2["n_perm_unit"] == 4 and r2["n_perm_group"] == 3
    assert r2["p_perm"] == r2["p_perm_unit"]
    for k in ["null_group_mean", "null_group_sd", "null_group_q025", "null_group_q975", "p_perm_group"]:
        assert not np.isnan(r[k])


def test_group_size_link_inflates_size_matched_null():
    """The reason the unit null is the default: when a feature tracks how many images a
    patient has and patient size tracks the label, the size-matched group null keeps
    that link and rises above the unit null."""
    df = make_frame(n_patients=200, seed=7, size_label_link=True)
    size = df["group_id"].map(df["group_id"].value_counts()).to_numpy(float)
    r = run_feature_set("size", size[:, None], [False], df["label"].to_numpy(), df["group_id"].to_numpy(),
                        n_perm=10, units=df["lesion_id"].to_numpy(), n_perm_reference=10)
    gap = r["null_group_mean"] - r["null_mean"]
    se = np.sqrt(r["null_sd"] ** 2 / 10 + r["null_group_sd"] ** 2 / 10)
    assert gap > 2 * se


def test_sensitivity_skipped_when_redundant():
    """Groups that are single units of equal size (MILK10k: one lesion, two images) make the
    size-matched null the same process as the unit null, so it is not run."""
    n = 120
    groups = np.repeat([f"L{i}" for i in range(n)], 2)
    y = np.repeat(np.random.default_rng(0).choice(list("ABC"), n), 2)
    X = np.random.default_rng(1).normal(size=(2 * n, 1))
    r = run_feature_set("x", X, [False], y, groups, n_perm=2, units=groups, n_perm_reference=2)
    assert r["n_perm_group"] == 0 and r["sensitivity_skipped"].startswith("redundant")
    # unequal sizes: still run
    groups2 = groups.copy()
    groups2[-4:] = "L_big"
    y2 = y.copy()
    y2[-4:] = y[-1]
    r2 = run_feature_set("x", X, [False], y2, groups2, n_perm=2, units=groups2, n_perm_reference=2)
    assert r2["n_perm_group"] == 2 and r2["sensitivity_skipped"] is None


def test_clustered_labels_widen_the_null_that_keeps_patients_together():
    """Labels alike within a patient and a feature constant within a patient: the lesion null
    (which breaks the clustering) is narrower than the size-matched group null, which is why
    the group null gives the p-value when lesions sit inside patients."""
    rng = np.random.default_rng(3)
    rows = []
    for p in range(150):
        base, cam = rng.random() < 0.3, rng.integers(0, 4)
        for i in range(int(rng.integers(1, 7))):
            rows.append((p, cam, int(base if rng.random() < 0.9 else not base)))
    a = np.array(rows)
    r = run_feature_set("cam", a[:, [1]].astype(float), [True], a[:, 2], a[:, 0], n_perm=30,
                        units=np.arange(len(a)), n_perm_reference=30, n_repeats=1)
    assert r["test_null"] == "group"
    assert r["null_group_sd"] > 1.3 * r["null_sd"]


def test_reference_sets_stay_out_of_holm(tmp_path):
    from conftest import make_spec
    from skinaudit.audit import run_audit
    from skinaudit.schema import AuditDataset
    df = make_frame(n_patients=60, seed=2)
    extra = {"Z_noise_reference": (np.random.default_rng(0).normal(size=(len(df), 1)), [False],
                                   np.ones(len(df), bool))}
    s = run_audit(AuditDataset(make_spec(), df, {}), str(tmp_path), n_perm=2, n_boot=0, make_figures=False,
                  image_stats=False, n_jobs=1, mechanism=False, n_repeats=1, extra_feature_sets=extra)
    assert "Z_noise_reference" not in s["permutation"]["holm_family"]
    assert s["shortcut_cv"]["Z_noise_reference"]["p_perm_holm"] is None
