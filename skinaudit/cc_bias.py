# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Is the complete-case subset like the whole dataset? Standardized mean differences.

Missingness-injection experiments start from rows where metadata is complete; any
difference between those rows and the whole dataset is a selection bias the
experiments inherit. Every variable is compared on one scale: the standardized mean
difference (numeric) or its multi-category version (Yang and Dalton 2012; class mix,
acquisition and categorical fields); |SMD| > 0.1 is read as an imbalance, a common
convention (Austin 2009) rather than a test. Each field is compared on its observed values only (its own missing rate is
in the missing-rate tables). Intervals come from a bootstrap over whole groups
(patients or lesions), like every other interval in the audit: percentile intervals for
numeric SMDs, basic-bootstrap intervals for the multi-category SMD, which is >= 0 and
biased upward by sampling noise (a percentile interval would drift away from a small
estimate). The multi-category SMD is also reported bias-corrected
(2 * SMD - bootstrap median, clipped at 0); the corrected value is the one plotted,
flagged as imbalanced and used in the rubric.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .stats import _group_members, smd_from_shares, smd_numeric

THRESHOLD = 0.1
SUBSETS = ["complete", "complete_core"]
COMPARISONS = ["vs_all", "vs_rest"]


def _encode(df, spec, variables):
    """Numeric fields as float arrays; categorical ones as integer codes in sorted order of
    their string values (missing = -1), so shares come out in the same category order as
    smd_categorical uses. Done once, so the bootstrap only indexes arrays."""
    enc = {}
    for v in variables:
        s = df[v]
        if v in spec.num_fields:
            enc[v] = ("num", s.to_numpy(dtype=float))
        else:
            c = np.full(len(s), -1, dtype=np.int64)
            obs = s.notna().to_numpy()
            codes, uniq = pd.factorize(s[obs].astype(str), sort=True)
            c[obs] = codes
            enc[v] = ("cat", c, len(uniq))
    return enc


def _smd_cat_codes(ca, cb, k):
    na, nb = np.bincount(ca[ca >= 0], minlength=k), np.bincount(cb[cb >= 0], minlength=k)
    if na.sum() == 0 or nb.sum() == 0:
        return np.nan
    keep = (na + nb) > 0                 # categories present in at least one group
    return smd_from_shares(na[keep] / na.sum(), nb[keep] / nb.sum())


def _smd_all(enc, flags, variables, idx=None):
    """SMD of every (subset, comparison, variable); idx = bootstrap rows (None = all rows)."""
    rows = []
    for subset in SUBSETS:
        inside = flags[subset] if idx is None else flags[subset][idx]
        for comparison in COMPARISONS:
            ref = np.ones(len(inside), bool) if comparison == "vs_all" else ~inside
            for v in variables:
                e = enc[v]
                col = e[1] if idx is None else e[1][idx]
                if e[0] == "num":
                    val = smd_numeric(col[inside], col[ref])
                else:
                    val = _smd_cat_codes(col[inside], col[ref], e[2])
                rows.append((subset, comparison, v, val, int(inside.sum()), int(ref.sum())))
    return rows


def complete_case_smd(df: pd.DataFrame, spec, n_boot: int = 1000, seed: int = 2026) -> pd.DataFrame:
    variables = ["label", "acquisition", *spec.fields]
    enc = _encode(df, spec, variables)
    flags = {s: df[s].to_numpy() == 1 for s in SUBSETS}
    point = _smd_all(enc, flags, variables)
    out = pd.DataFrame(point, columns=["subset", "comparison", "variable", "smd", "n_subset", "n_reference"])
    out["kind"] = ["numeric" if v in spec.num_fields else "categorical" for v in out["variable"]]
    if n_boot:
        members = _group_members(df["group_id"].astype(str).to_numpy())
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            boots.append([r[3] for r in _smd_all(enc, flags, variables, idx)])
        boots = np.asarray(boots, float)
        # a comparison with an empty side (e.g. vs_rest when every row is complete) is NaN in
        # every resample; its interval stays NaN without a warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q_lo, q_hi = np.nanquantile(boots, 0.025, axis=0), np.nanquantile(boots, 0.975, axis=0)
            med = np.nanmedian(boots, axis=0)
        cat = (out["kind"] == "categorical").to_numpy()
        est = out["smd"].to_numpy(float)
        # numeric SMD: percentile interval. Multi-category SMD is >= 0 and biased upward by
        # sampling noise, so a percentile interval drifts away from a small estimate; the basic
        # bootstrap interval [2*est - q975, 2*est - q025] corrects that bias (clipped at 0).
        out["ci_low"] = np.where(cat, np.clip(2 * est - q_hi, 0, None), q_lo)
        out["ci_high"] = np.where(cat, np.clip(2 * est - q_lo, 0, None), q_hi)
        out["ci_method"] = np.where(cat, "basic bootstrap", "percentile bootstrap")
        # bias-corrected point for the multi-category SMD (inside its basic interval by construction)
        out["smd_corrected"] = np.where(cat, np.clip(2 * est - med, 0, None), est)
    else:
        out["smd_corrected"] = out["smd"]
    out["imbalanced"] = out["smd_corrected"].abs() > THRESHOLD
    return out[["subset", "comparison", "variable", "kind", "smd", "smd_corrected",
                *(["ci_low", "ci_high", "ci_method"] if n_boot else []),
                "imbalanced", "n_subset", "n_reference"]]


def max_abs_smd(table: pd.DataFrame, subset: str, comparison: str = "vs_all") -> dict:
    t = table[(table["subset"] == subset) & (table["comparison"] == comparison)].dropna(subset=["smd_corrected"])
    if t.empty:
        return {"max_abs_smd": None, "variable": None}
    r = t.loc[t["smd_corrected"].abs().idxmax()]
    return {"max_abs_smd": round(float(abs(r["smd_corrected"])), 4), "variable": r["variable"]}
