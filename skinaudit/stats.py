# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Association statistics used by the audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MISSING_TOKENS = {"", "unknown", "unk", "nan", "none", "na", "n/a"}


def normalize_missing(s: pd.Series) -> tuple[pd.Series, int]:
    """Strip strings and turn missing-like tokens into NaN. Returns (series, n_converted)."""
    s = s.astype("string").str.strip()
    is_token = s.str.lower().isin(MISSING_TOKENS) & s.notna()
    return s.mask(is_token), int(is_token.sum())


def _v_from_table(tab: np.ndarray) -> dict:
    """Cramer's V, bias-corrected V (Bergsma 2013) and chi-square p from a contingency
    table whose rows and columns are all non-empty."""
    r, k = tab.shape
    n = int(tab.sum())
    if r < 2 or k < 2:
        return {"v": np.nan, "v_corrected": np.nan, "p": np.nan, "n": n}
    chi2, p, _, _ = stats.chi2_contingency(tab, correction=False)
    phi2 = chi2 / n
    v = np.sqrt(phi2 / (min(r, k) - 1))
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    vc = np.sqrt(phi2c / min(kc - 1, rc - 1)) if min(kc, rc) > 1 else np.nan
    return {"v": float(v), "v_corrected": float(vc), "p": float(p), "n": n}


def cramers_v(a, b) -> dict:
    """Cramer's V, bias-corrected V (Bergsma 2013) and chi-square p-value.

    Yates continuity correction is switched off so that 2x2 tables are treated
    the same way as larger ones. Rows where either value is missing are dropped.
    """
    return _v_from_table(pd.crosstab(a, b).to_numpy())


def codes(values) -> tuple[np.ndarray, int]:
    """Integer codes (missing = -1) and the number of categories, for fast tables."""
    c, uniq = pd.factorize(pd.Series(np.asarray(values, dtype=object)), sort=False)
    return c.astype(np.int64), len(uniq)


def _table_from_codes(ca, cb, ka, kb) -> np.ndarray:
    """Contingency table of two code arrays, missing dropped and empty rows / columns removed
    (the same table pd.crosstab builds, up to row and column order, which V does not use)."""
    ok = (ca >= 0) & (cb >= 0)
    tab = np.bincount(ca[ok] * kb + cb[ok], minlength=ka * kb).reshape(ka, kb)
    return tab[tab.sum(axis=1) > 0][:, tab.sum(axis=0) > 0]


def _group_members(groups) -> list[np.ndarray]:
    """Row indices of every group, ascending within each group, groups in sorted order."""
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)


def grouped_bootstrap_ci(a, b, groups, stat=None,
                         n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of stat(a, b), resampling whole groups (patients or lesions).

    stat=None (default) is the bias-corrected Cramer's V, computed from integer codes with
    np.bincount: the same values as cramers_v(a[idx], b[idx])["v_corrected"], much faster
    on large datasets. Any other stat(x, y) is called on the resampled arrays."""
    a, b = np.asarray(a), np.asarray(b)
    members = _group_members(groups)
    rng = np.random.default_rng(seed)
    if stat is None:
        ca, ka = codes(a)
        cb, kb = codes(b)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[i] for i in pick])
        if stat is None:
            vals.append(_v_from_table(_table_from_codes(ca[idx], cb[idx], ka, kb))["v_corrected"])
        else:
            vals.append(stat(a[idx], b[idx]))
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def kruskal_epsilon2(values, labels) -> dict:
    """Kruskal-Wallis test with the epsilon-squared effect size, epsilon^2 = H / (n - 1)
    (Tomczak and Tomczak 2014; between 0 and 1). The p-value treats rows as independent."""
    d = pd.DataFrame({"v": values, "y": labels}).dropna()
    groups = [g["v"].to_numpy() for _, g in d.groupby("y")]
    n, k = len(d), len(groups)
    if k < 2 or d["v"].nunique() < 2:
        # one class, or the same value everywhere (e.g. every image has one resolution):
        # nothing to rank, so no test and no effect size
        return {"statistic": np.nan, "p": np.nan, "epsilon2": np.nan, "n": int(n)}
    kw = stats.kruskal(*groups)
    return {"statistic": float(kw.statistic), "p": float(kw.pvalue),
            "epsilon2": float(kw.statistic / (n - 1)), "n": int(n)}


def smd_numeric(a, b) -> float:
    """Standardized mean difference (mean_a - mean_b) / sqrt((var_a + var_b) / 2), NaN dropped."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if s == 0:
        return 0.0 if a.mean() == b.mean() else np.nan
    return float((a.mean() - b.mean()) / s)


def smd_from_shares(p1: np.ndarray, p2: np.ndarray) -> float:
    """Multi-category SMD (Yang and Dalton 2012) from the category shares of two groups,
    all categories present in at least one group, in a fixed order; the last is dropped
    as reference. S^+ is the pseudo-inverse, so rare categories do not make it unstable."""
    if len(p1) < 2:
        return 0.0
    p1, p2 = p1[:-1], p2[:-1]
    s = -(np.outer(p1, p1) + np.outer(p2, p2)) / 2
    np.fill_diagonal(s, (p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    d = p1 - p2
    return float(np.sqrt(max(0.0, d @ np.linalg.pinv(s) @ d)))


def smd_categorical(a, b) -> float:
    """Multi-category SMD (Yang and Dalton 2012): sqrt(d' S^+ d), d = difference of the
    category shares (one category dropped as reference), S = pooled covariance of the
    shares. Categories empty in both groups are dropped; S^+ is the pseudo-inverse, so
    rare categories do not make the result unstable. NaN values are ignored."""
    a = pd.Series(a).dropna().astype(str)
    b = pd.Series(b).dropna().astype(str)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    cats = sorted(set(a) | set(b))
    p1 = a.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    p2 = b.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return smd_from_shares(p1, p2)


def holm(p) -> np.ndarray:
    """Holm step-down adjustment of a set of p-values (NaN stays NaN and is not counted)."""
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out[order] = adj
    return out
