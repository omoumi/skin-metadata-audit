# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Missingness-mechanism models and their label-dependence tests, on small synthetic data."""
import numpy as np
import pandas as pd

from conftest import make_frame, make_spec
from skinaudit.mechanism import (_blocks, draw_crt_labels, label_dependence, missingness_mechanism,
                                 stratified_group_swaps)
from skinaudit.shortcut import label_constant_within


def _frame(n=400, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({"group_id": [f"g{i // 2}" for i in range(n)], "a": r.normal(size=n),
                         "b": r.normal(size=n), "c": r.normal(size=n), "d": r.normal(size=n)})


def test_blocks_use_jaccard_and_thresholds():
    df = _frame()
    df.loc[:59, ["a", "b"]] = np.nan          # a and b always missing together
    df.loc[200:259, "c"] = np.nan             # c missing elsewhere
    df.loc[[5, 300], "d"] = np.nan            # d too rare to model
    blocks, skipped, jac = _blocks(df, ["a", "b", "c", "d"], min_missing=20, min_groups=10, jaccard=0.9)
    assert sorted(map(sorted, blocks)) == [["a", "b"], ["c"]]
    assert "d" in skipped and jac[("a", "b")] == 1.0 and jac[("a", "c")] == 0.0


def test_rare_fields_are_not_blocked_by_agreement():
    """Plain agreement of two rarely missing fields is ~100 % even if they never go missing
    together; Jaccard on the missing rows is 0, and the thresholds skip them anyway."""
    df = _frame()
    df.loc[[1, 2], "a"] = np.nan
    df.loc[[3, 4], "b"] = np.nan
    blocks, skipped, jac = _blocks(df, ["a", "b"], 20, 10, 0.9)
    assert blocks == [] and set(skipped) == {"a", "b"} and jac == {}


def _dep_data(seed, dependent, n=400):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, 3))
    y = np.where(r.random(n) < 0.5, "A", "B")
    p = 0.2 + 0.6 * (y == "A") if dependent else np.full(n, 0.4)
    t = (r.random(n) < p).astype(int)
    units = np.array([f"u{i}" for i in range(n)])
    return X, y, t, units


def test_label_dependence_found_and_absent():
    X, y, t, u = _dep_data(1, dependent=True)
    r = label_dependence(X, [False] * 3, t, y, u, u, strata_fine=np.zeros(len(y)), strata_coarse=np.zeros(len(y)),
                         n_perm=4, seed=1, n_jobs=1)
    assert r["auc_gain"] > 0.2 and r["crt_p"] == 0.2 and r["group_swap_p"] == 0.2
    assert r["label_r2"] < 0.1        # label unrelated to the inputs: the gain cannot be substitution
    X, y, t, u = _dep_data(2, dependent=False)
    r = label_dependence(X, [False] * 3, t, y, u, u, n_perm=4, seed=2, n_jobs=1)
    assert abs(r["auc_gain"]) < 0.08


def test_draws_keep_one_label_per_unit():
    units = np.repeat([f"u{i}" for i in range(50)], 3)
    y = np.repeat(np.random.default_rng(0).choice(list("ABC"), 50), 3)
    rng = np.random.default_rng(1)
    proba = np.full((len(y), 3), 1 / 3)
    for d in draw_crt_labels(np.array(list("ABC")), proba, units, 3, rng):
        assert label_constant_within(units, d)


def test_singleton_strata_fall_back_and_movable_share():
    groups = np.array([f"g{i}" for i in range(6)])
    y = np.array(list("ABABAB"))
    fine = np.array(["x1", "x2", "x3", "y1", "y1", "z1"])
    coarse = np.array(["x", "x", "x", "y", "y", "z"])
    draws, movable = stratified_group_swaps(y, groups, fine, coarse, 5, np.random.default_rng(0))
    assert movable == 5 / 6          # z1 is alone in both levels; x1-x3 move within x
    for d in draws:
        assert d[5] == y[5]           # the lone group keeps its label
        assert sorted(d[:3]) == sorted(y[:3]) and sorted(d[3:5]) == sorted(y[3:5])


def test_missingness_mechanism_table():
    df = make_frame(n_patients=150, seed=4)
    r = np.random.default_rng(4)
    df.loc[r.random(len(df)) < 0.2, "site"] = np.nan
    df.loc[r.random(len(df)) < 0.01, "age_approx"] = np.nan
    t = missingness_mechanism(df, make_spec(), df["lesion_id"].to_numpy(), n_perm=2, seed=4, n_jobs=1)
    done = t[t["skipped"].isna()]
    assert set(done["fields"]) == {"site"} and set(done["variant"]) == {"with_indicators", "values_only"}
    assert t.loc[t["fields"] == "age_approx", "skipped"].str.startswith("too few").all()
    assert done[["auc_a", "auc_b", "crt_p", "group_swap_p", "label_r2", "group_swap_movable_share",
                 "crt_p_holm", "group_swap_p_holm"]].notna().all().all()
    assert (done["crt_p_holm"] >= done["crt_p"]).all()
    assert done["label_dependence_supported"].map(lambda v: v in (True, False)).all()
    # values_only drops rows with a missing input (age) and never feeds NaN
    vo = done[done["variant"] == "values_only"].iloc[0]
    assert vo["n_rows"] == int(df["age_approx"].notna().sum()) and "miss_" not in vo["inputs"]


def test_label_model_is_calibrated_not_overconfident():
    """P^ feeds the CRT draws, so it must beat the class frequencies in log-loss (the
    shortcut model's settings can fail this) and keep the class mix."""
    from skinaudit.mechanism import _label_model
    r = np.random.default_rng(1)
    n = 1500
    groups = np.repeat(np.arange(500), 3)
    x = np.column_stack([r.normal(size=n), r.integers(0, 4, n).astype(float), r.normal(size=n)])
    logits = np.column_stack([0.8 * x[:, 0], 0.4 * (x[:, 1] == 2), np.zeros(n), -1.5 + 0 * x[:, 0]])
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([r.choice(list("abcd"), p=row) for row in p])
    classes, proba, ll, ll0, rounds = _label_model(x, y, [False, True, False], groups, seed=3)
    assert ll < ll0                                   # better than predicting the class frequencies
    freq = pd.Series(y).value_counts(normalize=True).reindex(classes).to_numpy()
    assert np.abs(proba.mean(0) - freq).max() < 0.03  # draws keep the class mix
    assert len(rounds) == 5 and all(1 <= k <= 300 for k in rounds)
    # pure noise inputs: P^ falls back to roughly the class frequencies (R2 near 0, not very negative)
    _, _, ll_n, ll0_n, _ = _label_model(r.normal(size=(n, 3)), y, [False] * 3, groups, seed=3)
    assert 1 - ll_n / ll0_n > -0.05


def test_rubric_picks_row_where_both_tests_are_strongest():
    from skinaudit.audit import _strongest_label_dependence
    # b has the smallest CRT p but a weak group-swap p; the larger of the two Holm p decides
    m = pd.DataFrame({"fields": ["a", "b", "c", "d"], "variant": ["v"] * 4, "auc_gain": [0.12, 0.03, 0.05, 0.02],
                      "crt_p_holm": [0.4, 0.01, 0.02, 0.02], "group_swap_p_holm": [0.5, 0.6, 0.02, 0.02],
                      "crt_p": [0.1, 0.005, 0.01, 0.01], "group_swap_p": [0.2, 0.3, 0.01, 0.01],
                      "label_dependence_supported": [False, False, True, True], "label_r2": [0.2, 0.1, 0.1, 0.1]})
    r = _strongest_label_dependence(m)
    assert r["fields"] == "c" and r["supported"] is True


def test_group_swaps_keep_whole_label_vectors():
    """Groups only swap with groups of the same size and stratum, and keep their label vector
    (so labels stay clustered within a patient)."""
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p5"])
    y = np.array(list("AABBABAB"))
    fine = np.array(["x", "x", "x", "x", "y", "y", "x", "x"])
    draws, movable = stratified_group_swaps(y, groups, fine, fine, 30, np.random.default_rng(0))
    seen = set()
    for d in draws:
        # p1 and p2 (size 2, stratum x) swap; p3 (size 2, y) has no partner: unchanged
        assert {"".join(d[:2]), "".join(d[2:4])} == {"AA", "BB"} and "".join(d[4:6]) == "AB"
        assert sorted(d[6:]) == ["A", "B"]           # p4 and p5: size 1, stratum x, swap with each other
        seen.add("".join(d))
    assert len(seen) > 1 and movable == 6 / 8


def test_group_swap_works_when_age_is_the_target():
    """Age missing for whole groups, more often in class A. The age band of such a group is
    'na', so with the band as a stratum the swap could only move labels among groups without
    age and could never break the link; it must be left out when age is the target."""
    r = np.random.default_rng(7)
    n = 300
    lab = r.choice(["A", "B"], n)
    rows = []
    for g in range(n):
        for i in range(2):
            rows.append({"image_id": f"i{g}_{i}", "group_id": f"g{g}", "lesion_id": f"g{g}", "label": lab[g]})
    df = pd.DataFrame(rows)
    df["acquisition"] = np.where(np.arange(len(df)) % 2, "derm", "clin")
    df["age_approx"] = np.repeat(r.integers(20, 85, n), 2).astype(float)
    df["sex"] = np.repeat(r.choice(["male", "female"], n), 2)
    df["site"] = np.repeat(r.choice(["trunk", "head", "leg"], n), 2)
    miss = np.repeat(r.random(n) < np.where(lab == "A", 0.35, 0.03), 2)
    df.loc[miss, "age_approx"] = np.nan
    t = missingness_mechanism(df, make_spec(classes=["A", "B"]), df["lesion_id"].to_numpy(), n_perm=19, seed=7,
                              n_jobs=1)
    row = t[(t["fields"] == "age_approx") & (t["variant"] == "with_indicators")].iloc[0]
    assert row["group_swap_strata"] == "images per group x acquisition"
    assert row["auc_gain"] > 0.2 and row["group_swap_p"] <= 0.05 and row["crt_p"] <= 0.05
    other = t[(t["fields"] == "site") | (t["fields"] == "sex")]
    assert other["skipped"].notna().all()     # never missing: not modelled
