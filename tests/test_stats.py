# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
import numpy as np
import pandas as pd

from skinaudit.stats import cramers_v, grouped_bootstrap_ci, kruskal_epsilon2, normalize_missing


def test_cramers_v_independent_is_near_zero():
    rng = np.random.default_rng(0)
    a, b = rng.choice(list("xyz"), 5000), rng.choice(list("pq"), 5000)
    r = cramers_v(a, b)
    assert r["v_corrected"] < 0.03
    assert r["p"] > 0.001


def test_cramers_v_perfect_association_is_one():
    a = np.repeat(list("xyz"), 200)
    r = cramers_v(a, a)
    assert abs(r["v"] - 1) < 1e-9 and r["v_corrected"] > 0.99


def test_cramers_v_single_level_returns_nan():
    assert np.isnan(cramers_v(["x"] * 10, list("ab") * 5)["v"])


def test_normalize_missing_tokens():
    s, n = normalize_missing(pd.Series(["male", " unknown ", "", "NA", None, "female"]))
    assert n == 3
    assert s.isna().sum() == 4 and s.iloc[0] == "male"


def test_kruskal_epsilon2_separates_groups():
    v = np.r_[np.zeros(100), np.ones(100) * 10] + np.random.default_rng(0).normal(0, 0.1, 200)
    # two fully separated groups of 100: H = 149.25, epsilon2 = H / (n - 1) = 0.75
    assert kruskal_epsilon2(v, np.repeat(["a", "b"], 100))["epsilon2"] > 0.7


def test_grouped_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(200), 3)
    lab = np.repeat(rng.choice(list("ab"), 200), 3)
    m = (lab == "a") & (rng.random(600) < 0.6)
    lo, hi = grouped_bootstrap_ci(m.astype(int), lab, g, n_boot=100)
    v = cramers_v(m.astype(int), lab)["v_corrected"]
    assert lo <= v + 0.05 and hi >= v - 0.05 and lo < hi


def test_smd_numeric_and_categorical_agree_on_binary():
    from skinaudit.stats import smd_categorical, smd_numeric
    r = np.random.default_rng(0)
    a = r.choice(list("xy"), 3000, p=[.6, .4])
    b = r.choice(list("xy"), 3000, p=[.5, .5])
    # equal up to the variance estimator (p(1-p) vs the ddof=1 sample variance)
    assert abs(smd_categorical(a, b) - abs(smd_numeric(a == "x", b == "x"))) < 1e-3
    assert abs(smd_numeric(r.normal(0.5, 1, 20000), r.normal(0, 1, 20000)) - 0.5) < 0.05


def test_smd_categorical_is_stable_with_empty_and_rare_categories():
    from skinaudit.stats import smd_categorical
    v = smd_categorical(["a"] * 99 + ["z"], ["a"] * 100)
    assert np.isfinite(v) and 0 < v < 0.5
    assert smd_categorical(["a", "b"], ["a", "b"]) == 0.0 and np.isnan(smd_categorical([], ["a"]))


def test_complete_case_smd_table():
    from conftest import make_frame, make_spec
    from skinaudit.cc_bias import complete_case_smd, max_abs_smd
    df = make_frame(n_patients=80, seed=5)
    r = np.random.default_rng(5)
    old = df["age_approx"] > 60
    df.loc[old & (r.random(len(df)) < 0.6), "site"] = np.nan      # older patients lose site: biased subset
    miss = [f"miss_{f}" for f in ["age_approx", "sex", "site"]]
    for f in ["age_approx", "sex", "site"]:
        df[f"miss_{f}"] = df[f].isna().astype(int)
    df["complete"] = (df[miss].sum(axis=1) == 0).astype(int)
    df["complete_core"] = df[["age_approx", "sex"]].notna().all(axis=1).astype(int)
    t = complete_case_smd(df, make_spec(), n_boot=30, seed=1)
    assert set(t["subset"]) == {"complete", "complete_core"} and set(t["comparison"]) == {"vs_all", "vs_rest"}
    age = t[(t["subset"] == "complete") & (t["comparison"] == "vs_all") & (t["variable"] == "age_approx")].iloc[0]
    assert age["smd"] < -0.1 and age["ci_low"] <= age["smd"] <= age["ci_high"]
    assert max_abs_smd(t, "complete")["max_abs_smd"] >= round(abs(age["smd_corrected"]), 4)
    cat = t[t["kind"] == "categorical"].dropna(subset=["smd"])
    # the bias-corrected estimate sits inside its basic-bootstrap interval, and both are >= 0
    eps = 1e-12
    assert ((cat["ci_low"] <= cat["smd_corrected"] + eps) & (cat["smd_corrected"] <= cat["ci_high"] + eps)).all()
    assert (cat["ci_low"] >= 0).all() and (cat["smd_corrected"] >= 0).all()


def test_kruskal_epsilon2_constant_values_give_nan():
    from skinaudit.stats import kruskal_epsilon2
    r = kruskal_epsilon2([2.0] * 10, list("ab") * 5)
    assert np.isnan(r["epsilon2"]) and np.isnan(r["p"]) and r["n"] == 10
    assert np.isnan(kruskal_epsilon2([1.0, 2.0, 3.0], ["a"] * 3)["epsilon2"])


def test_kruskal_epsilon2_is_h_over_n_minus_1():
    from scipy import stats as st
    rng = np.random.default_rng(0)
    v = rng.normal(size=150) + np.repeat([0, 0.3, 0.6, 0, 0.2], 30)
    lab = np.repeat(list("abcde"), 30)
    h = st.kruskal(*[v[lab == c] for c in "abcde"]).statistic
    assert abs(kruskal_epsilon2(v, lab)["epsilon2"] - h / 149) < 1e-12


def test_holm():
    from skinaudit.stats import holm
    adj = holm([0.01, 0.04, np.nan, 0.03, 0.5])
    # sorted 0.01, 0.03, 0.04, 0.5 with m = 4: 0.04, 0.09, 0.09 (monotone), 0.5
    assert np.allclose(adj[[0, 3, 1, 4]], [0.04, 0.09, 0.09, 0.5]) and np.isnan(adj[2])
