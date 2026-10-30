# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Coverage of the group-bootstrap intervals (Cramer's V, complete-case SMD) on clustered data."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests")
from skinaudit.stats import cramers_v, grouped_bootstrap_ci
from skinaudit.cc_bias import complete_case_smd
from conftest import make_spec
nsim, out = int(sys.argv[2]), sys.argv[3]


def sample(rng, n_pat):
    g = np.repeat(np.arange(n_pat), rng.integers(1, 7, n_pat))
    k = len(g)
    base = (rng.random(n_pat) < 0.35)[g]
    lab = np.where(rng.random(k) < 0.85, base, ~base)
    label = np.where(lab, "B", "A")
    age = rng.normal(50, 12, n_pat)[g] + 5 * lab
    # site missing per patient, more often for class B and for older patients
    pm = 1 / (1 + np.exp(-(-1.6 + 0.8 * lab + 0.03 * (age - 50))))
    miss = rng.random(n_pat)[g] < pm
    site = np.where(miss, None, rng.choice(["trunk", "head", "leg"], k, p=[0.5, 0.2, 0.3]))
    sex = rng.choice(["male", "female"], k)
    df = pd.DataFrame({"image_id": np.arange(k).astype(str), "group_id": g.astype(str), "label": label,
                       "acquisition": rng.choice(["c1", "c2"], n_pat)[g], "age_approx": age, "sex": sex, "site": site})
    for f in ["age_approx", "sex", "site"]:
        df[f"miss_{f}"] = df[f].isna().astype(int)
    df["complete"] = df[["miss_age_approx", "miss_sex", "miss_site"]].sum(axis=1).eq(0).astype(int)
    df["complete_core"] = df[["age_approx", "sex"]].notna().all(axis=1).astype(int)
    return df


spec = make_spec(classes=["A", "B"])
big = sample(np.random.default_rng(1), 200_000)
v_true = cramers_v(big["miss_site"], big["label"])["v"]
smd_true = complete_case_smd(big, spec, n_boot=0).set_index(["subset", "comparison", "variable"])["smd"]
keys = [("complete", "vs_all", "label"), ("complete", "vs_all", "age_approx"), ("complete", "vs_rest", "age_approx"),
        ("complete", "vs_rest", "label"), ("complete", "vs_all", "acquisition")]
res = []
for sim in range(nsim):
    d = sample(np.random.default_rng(100 + sim), 300)
    lo, hi = grouped_bootstrap_ci(d["miss_site"], d["label"], d["group_id"], n_boot=1000, seed=sim)
    t = complete_case_smd(d, spec, n_boot=1000, seed=sim).set_index(["subset", "comparison", "variable"])
    row = {"v_cover": bool(lo <= v_true <= hi)}
    for k in keys:
        row["smd_cover_" + "|".join(k)] = bool(t.loc[k, "ci_low"] <= abs(smd_true[k]) <= t.loc[k, "ci_high"]) \
            if t.loc[k, "kind"] == "categorical" else bool(t.loc[k, "ci_low"] <= smd_true[k] <= t.loc[k, "ci_high"])
    res.append(row)
    json.dump({"v_true": v_true, "smd_true": {"|".join(k): float(smd_true[k]) for k in keys}, "runs": res}, open(out, "w"))
