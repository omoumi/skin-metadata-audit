# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Missingness mechanism: what explains that a field is empty?

For each field (or block of fields that go missing together) two models predict the
field's missing indicator with grouped cross-validation:

  (a) from the other fields, the acquisition variable and the group size;
  (b) the same plus the label.

An AUC of (a) clearly above 0.5 is evidence against MCAR; observed data cannot tell MAR
from MNAR, so it says nothing about which of the two holds. The AUC gain of (b) over (a)
estimates how much of the missingness depends on the label in a way the other fields do
not explain, i.e. how much a model that sees the missing indicators could use them as a
shortcut. It may overstate that dependence (the label can stand in for the part of the
other fields the model did not learn) and overfitting can understate it, so the gain is
reported with a test rather than as a bound.

Test of "the label adds nothing beyond the inputs of (a)":
- main: conditional randomization test (CRT). New labels are drawn per permutation
  unit (lesion) from P^(label | inputs of (a)), an out-of-fold model, so they keep
  their relation to all inputs; model (b) is refitted on each draw. P^ is a shallow,
  regularised model whose number of boosting rounds is chosen on held-out groups, to keep
  its probabilities close to calibrated (see LABEL_MODEL_PARAMS). Exact only if
  P^ were the true conditional distribution; with an estimated P^ that under-fits, the
  test stays somewhat anti-conservative. P^'s out-of-fold log-loss is reported next to
  a constant (class-frequency) model so its quality can be judged.
  Draws are independent between the lesions of one patient, while real labels of one
  patient are often alike; with missingness decided per patient this makes the CRT
  anti-conservative (a simulation gave about 16% false positives at the 5% level; see
  validation/README.md).
- group swap: whole patients swap their label vectors with patients that have the same
  number of images, the same acquisition (most common value of the patient) and the same
  age band (patient median); a patient alone in such a stratum uses size x acquisition.
  When the age field is itself the target, the age band is left out (it would equal the
  target for patients without any age); the strata used are in group_swap_strata.
  The clustering of labels is kept exactly (the same simulation: 5%), but only these
  patient-level variables are held fixed, not every other field, so this test alone does not
  show that the label adds information beyond all other fields. The share of images
  whose patient can move is reported.
A field's missingness is reported as label-dependent only when both tests reject after
Holm adjustment over the rows of the table (label_dependence_supported). This rule has
limited power (see validation/README.md), so a field that is not reported may still be
label-dependent.

Limit: when the missingness depends
on a function of the inputs that the model cannot learn from the available rows, and the
label is a summary of that function, the label "substitutes" for it. In that regime a
plain permutation, the CRT and a residual-covariance test (GCM) are all expected to report
label dependence, and adding P^ as an input need not remove it: from observed data alone the
two situations can look the same (Shah and Peters 2020: conditional independence cannot be
tested without assumptions on how complex the relation is). Substitution needs a label
that is predictable from the inputs, so every row reports label_r2 = 1 - log-loss(P^) /
log-loss(class frequencies): near 0, the label carries little about the inputs and
substitution is unlikely; the higher it is, the more of the gain may be substitution. It is
an indication, not a correction.

Two input variants:
- with_indicators: every other field (values with NaN) plus their missing indicators;
  all rows. Co-missingness of other fields is part of what explains the target here.
- values_only: other fields with at most `max_input_missing` missing, on the rows where
  all of them are observed; no NaN enters the model (gradient boosting treats NaN, and
  any constant fill value, as a value of its own).

Fields with fewer than `min_missing` missing images or `min_groups` groups with a
missing image are skipped. Fields whose missing sets overlap with Jaccard >= `jaccard`
(both above the thresholds) form a block: one model per block, block members are not
inputs, and the target is "any block field missing".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .parallel import parallel_map, resolve_n_jobs
from . import shortcut
from .shortcut import category_codes, label_constant_within

FIGURE_TEXT = ("AUC of predicting that a field is missing, from the other fields, acquisition and group size (a), "
               "and with the label added (b). AUC(a) clearly above 0.5 is evidence against MCAR; it cannot tell MAR "
               "from MNAR. The gain of (b) over (a) may overstate the dependence on the label, because the label can "
               "stand in for the part of the other fields the model did not learn; label R2 shows how much of the "
               "label the other fields predict, and the higher it is, the more of the gain may be that substitution. "
               "p-values: conditional randomization test, and whole patients swapping labels within strata of images "
               "per patient x acquisition x age band (no age band when age itself is the target); both are "
               "Holm-adjusted, and a claim needs both. A field without a claim may still depend on the label: the "
               "power of this rule is limited.")


def _blocks(df, fields, min_missing, min_groups, jaccard):
    """Eligible fields grouped into blocks by Jaccard overlap of their missing rows."""
    miss = {f: df[f].isna().to_numpy() for f in fields}
    skipped, ok = {}, []
    for f in fields:
        n = int(miss[f].sum())
        ng = int(df.loc[miss[f], "group_id"].nunique()) if n else 0
        if n < min_missing or ng < min_groups:
            skipped[f] = f"too few missing ({n} images, {ng} groups; need {min_missing} and {min_groups})"
        else:
            ok.append(f)
    parent = {f: f for f in ok}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    jac = {}
    for i, a in enumerate(ok):
        for b in ok[i + 1:]:
            inter = np.logical_and(miss[a], miss[b]).sum()
            union = np.logical_or(miss[a], miss[b]).sum()
            jac[(a, b)] = inter / union if union else 0.0
            if jac[(a, b)] >= jaccard:
                parent[find(b)] = find(a)
    blocks = {}
    for f in ok:
        blocks.setdefault(find(f), []).append(f)
    return list(blocks.values()), skipped, jac


def _folds(target, groups, seed, n_splits=5):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in cv.split(np.zeros(len(target)), target, groups)]


def _binary_oof(X, target, cat_mask, folds, seed, params):
    """Out-of-fold P(missing) and per-fold AUC."""
    oof = np.zeros(len(target))
    per_fold = []
    for tr, te in folds:
        clf = HistGradientBoostingClassifier(categorical_features=np.asarray(cat_mask), random_state=seed, **params)
        clf.fit(X[tr], target[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        per_fold.append(roc_auc_score(target[te], oof[te]) if len(np.unique(target[te])) == 2 else np.nan)
    return oof, np.asarray(per_fold)


def _auc_with_label(X, ycode, target, cat_mask, folds, seed, params):
    """AUC of model (b) for one label vector (used for the real labels and for every draw)."""
    oof, _ = _binary_oof(np.c_[X, ycode], target, list(cat_mask) + [True], folds, seed, params)
    return float(roc_auc_score(target, oof))


# P^(label | X) must give calibrated probabilities, not just a good ranking: the CRT draws
# labels from it. The shortcut model (100 boosting rounds, no early stopping, balanced
# class weights) suits balanced accuracy but gives over-confident probabilities, whose
# out-of-fold log-loss can be worse than predicting the class frequencies. So P^ is a
# shallow, regularised model whose number of rounds is chosen on a held-out set of
# groups inside each training fold (sklearn's own early stopping splits by row, which
# would put images of one patient on both sides).
LABEL_MODEL_PARAMS = dict(learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0,
                          early_stopping=False)
LABEL_MODEL_MAX_ITER = 300


def _class_logloss(y_true, proba, classes):
    """Mean -log p(true class); a class the model never saw gets probability 1e-6."""
    idx = np.searchsorted(classes, y_true)
    ok = (idx < len(classes)) & (classes[np.minimum(idx, len(classes) - 1)] == y_true)
    p = np.full(len(y_true), 1e-6)
    p[ok] = proba[np.flatnonzero(ok), idx[ok]]
    return float(-np.mean(np.log(np.clip(p, 1e-6, None))))


def _fit_label_model(X, y, cat_mask, groups, seed):
    """Regularised classifier with the number of rounds chosen on held-out groups."""
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    n_iter = LABEL_MODEL_MAX_ITER
    try:
        itr, iva = next(inner.split(X, y, groups))
    except ValueError:              # too few groups for an inner split: keep the cap
        itr = iva = None
    if itr is not None and len(iva):
        clf = HistGradientBoostingClassifier(categorical_features=np.asarray(cat_mask), random_state=seed,
                                             max_iter=LABEL_MODEL_MAX_ITER, **LABEL_MODEL_PARAMS)
        clf.fit(X[itr], y[itr])
        losses = [_class_logloss(y[iva], p, clf.classes_) for p in clf.staged_predict_proba(X[iva])]
        n_iter = int(np.argmin(losses)) + 1
    clf = HistGradientBoostingClassifier(categorical_features=np.asarray(cat_mask), random_state=seed,
                                         max_iter=n_iter, **LABEL_MODEL_PARAMS)
    return clf.fit(X, y), n_iter


def _label_model(X, y, cat_mask, groups, seed, n_splits=5):
    """Out-of-fold P^(label | X), grouped folds stratified on the label.
    Returns classes, probabilities, log-loss of P^, log-loss of the class frequencies and
    the boosting rounds chosen per fold. P^ uses LABEL_MODEL_PARAMS."""
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    rounds = []
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        clf, n_iter = _fit_label_model(X[tr], y[tr], cat_mask, groups[tr], seed)
        rounds.append(n_iter)
        p = np.zeros((len(te), len(classes)))
        p[:, np.searchsorted(classes, clf.classes_)] = clf.predict_proba(X[te])
        proba[te] = p
    proba = np.clip(proba, 1e-6, None)
    proba /= proba.sum(axis=1, keepdims=True)
    freq = pd.Series(y).value_counts(normalize=True).reindex(classes).to_numpy()
    return (classes, proba, float(log_loss(y, proba, labels=classes)),
            float(log_loss(y, np.tile(freq, (len(y), 1)), labels=classes)), rounds)


def _unit_index(units):
    uniq, inv = np.unique(np.asarray(units).astype(str), return_inverse=True)
    return uniq, inv


def draw_crt_labels(classes, proba, units, n_draws, rng):
    """One label per unit drawn from the unit's mean P^; every image of a unit gets it."""
    _, inv = _unit_index(units)
    n_units = inv.max() + 1
    pu = np.zeros((n_units, proba.shape[1]))
    np.add.at(pu, inv, proba)
    pu /= pu.sum(axis=1, keepdims=True)
    cum = pu.cumsum(axis=1)
    draws = []
    for _ in range(n_draws):
        u = rng.random(n_units)[:, None]
        idx = (u > cum).sum(axis=1).clip(max=len(classes) - 1)
        draws.append(classes[idx][inv])
    return draws


def stratified_group_swaps(y, groups, strata_fine, strata_coarse, n_draws, rng):
    """Whole groups (patients) swap their label vectors inside strata of exact group size x
    fine stratum; a group alone in its fine stratum uses group size x coarse stratum. The
    strata are taken from each group's first row, so pass values that are constant within a
    group. The received vector is put in random order over the group's images. Returns the
    draws and the share of rows whose group can move."""
    from .stats import _group_members
    y = np.asarray(y)
    members = _group_members(groups)
    first = np.array([m[0] for m in members])
    size = np.array([len(m) for m in members])
    fine = pd.Series(size.astype(str)) + "|" + pd.Series(np.asarray(strata_fine).astype(str)[first])
    coarse = pd.Series(size.astype(str)) + "|" + pd.Series(np.asarray(strata_coarse).astype(str)[first])
    key = np.where(fine.map(fine.value_counts()) >= 2, "f:" + fine, "c:" + coarse)
    bucket = pd.Series(np.arange(len(members))).groupby(key).apply(lambda s: s.to_numpy()).tolist()
    movable = float(sum(size[b].sum() for b in bucket if len(b) > 1) / len(y))
    draws = []
    for _ in range(n_draws):
        new = y.copy()
        for b in bucket:
            if len(b) > 1:
                for dst, src in zip(b, b[rng.permutation(len(b))]):
                    new[members[dst]] = rng.permutation(y[members[src]])
        draws.append(new)
    return draws, movable


def _p(null, obs):
    null = np.asarray(null, float)
    return float((1 + (null >= obs).sum()) / (len(null) + 1)) if len(null) else np.nan


def _gain_on_rows(X, ycode, target, groups, cat_mask, seed, params):
    """AUC gain of model (b) over model (a) on a row set of its own (folds built for these
    rows). With a re-applied subsample every draw has other rows, and AUC(a) changes with
    them, so the gain, not AUC(b) alone, is the statistic."""
    folds = _folds(target, groups, seed)
    oof_a, _ = _binary_oof(X, target, cat_mask, folds, seed, params)
    return _auc_with_label(X, ycode, target, cat_mask, folds, seed, params) - float(roc_auc_score(target, oof_a))


def label_dependence(X, cat_mask, target, y, groups, units, strata_fine=None, strata_coarse=None,
                     n_perm=20, seed=2026, n_jobs=None, method=("crt", "group_swap"), resample=None) -> dict:
    """AUC of models (a) and (b) and the label-dependence tests for one target.

    resample: for a label-dependent subsample (SLICE-3D). A dict with the eligible rows before
    subsampling ("X", "target", "y", "groups", "strata_fine", "strata_coarse") and "rule"
    (labels over those rows -> subsample mask). The group swap then permutes labels on all
    eligible rows (sizes matched on the full patients), re-applies the rule and refits model
    (b) on the rows it selects; on the subsample itself patients of equal size would also
    have equal numbers of malignant tiles, and the swap could not change anything. The CRT
    stays on the subsample rows (it draws from P^(label | inputs) fitted there)."""
    params = dict(shortcut.MODEL_PARAMS)      # read at call time, so a changed MODEL_PARAMS is used
    n_jobs = resolve_n_jobs(n_jobs)
    target = np.asarray(target).astype(int)
    folds = _folds(target, groups, seed)
    oof_a, fa = _binary_oof(X, target, cat_mask, folds, seed, params)
    ycode = category_codes(pd.Series(y))
    oof_b, fb = _binary_oof(np.c_[X, ycode], target, list(cat_mask) + [True], folds, seed, params)
    auc_a, auc_b = float(roc_auc_score(target, oof_a)), float(roc_auc_score(target, oof_b))
    out = {"auc_a": auc_a, "auc_b": auc_b, "auc_gain": auc_b - auc_a,
           "fold_gain_mean": float(np.nanmean(fb - fa)), "fold_gain_sd": float(np.nanstd(fb - fa)),
           "n_perm": int(n_perm)}
    if not n_perm:
        return out
    rng = np.random.default_rng(seed)
    codes_of = {c: i for i, c in enumerate(pd.Series(y).astype("category").cat.categories)}

    def to_codes(lab):
        return np.array([codes_of.get(v, np.nan) for v in lab], dtype=float)
    tasks, kinds = [], []
    if "crt" in method:
        classes, proba, ll, ll0, rounds = _label_model(X, np.asarray(y), cat_mask, np.asarray(groups), seed)
        out.update({"phat_logloss": ll, "const_logloss": ll0, "label_r2": 1 - ll / ll0 if ll0 > 0 else np.nan,
                    "phat_rounds": ",".join(map(str, rounds))})
        for d in draw_crt_labels(classes, proba, units, n_perm, rng):
            tasks.append((X, to_codes(d), target, cat_mask, folds, seed, params))
            kinds.append("crt")
    swap_tasks = []
    if "group_swap" in method and resample is not None:
        r = resample
        draws, movable = stratified_group_swaps(r["y"], r["groups"], r["strata_fine"], r["strata_coarse"], n_perm, rng)
        out["group_swap_movable_share"] = movable
        g_r, t_r = np.asarray(r["groups"]), np.asarray(r["target"]).astype(int)
        for d in draws:
            m = np.asarray(r["rule"](d), bool)
            swap_tasks.append((r["X"][m], to_codes(d[m]), t_r[m], g_r[m], cat_mask, seed, params))
        out["group_swap_statistic"] = "auc_gain"
    elif "group_swap" in method and strata_fine is not None:
        draws, movable = stratified_group_swaps(y, groups, strata_fine, strata_coarse, n_perm, rng)
        out["group_swap_movable_share"] = movable
        for d in draws:
            tasks.append((X, to_codes(d), target, cat_mask, folds, seed, params))
            kinds.append("group_swap")
    scores = list(parallel_map(_auc_with_label, tasks, n_jobs))
    if swap_tasks:
        scores += list(parallel_map(_gain_on_rows, swap_tasks, n_jobs))
        kinds += ["group_swap"] * len(swap_tasks)
    scores = np.asarray(scores, float)
    kinds = np.asarray(kinds)
    for k in ["crt", "group_swap"]:
        s = scores[kinds == k]
        if len(s):
            # the statistic is AUC(b) when the rows are fixed (AUC(a) is then a constant), and the
            # gain AUC(b) - AUC(a) when each draw has its own rows (re-applied subsample)
            obs = out["auc_gain"] if (k == "group_swap" and swap_tasks) else auc_b
            out.update({f"{k}_p": _p(s, obs), f"{k}_null_mean": float(s.mean()),
                        f"{k}_null_q975": float(np.quantile(s, 0.975)),
                        # a null that never moves cannot test anything: either the permutation
                        # cannot change what model (b) sees, or model (b) never uses the label
                        f"{k}_degenerate": bool(len(s) > 1 and np.ptp(s) == 0)})
    return out


def _encode(df, cols, num_fields):
    blocks, cat = [], []
    for c in cols:
        if c in num_fields:
            blocks.append(df[c].to_numpy(dtype=float))
            cat.append(False)
        else:
            blocks.append(category_codes(df[c]))
            cat.append(True)
    return blocks, cat


def auto_n_perm(n_tests: int) -> int:
    """Label draws per test so that Holm over n_tests rows can reach 0.05: 25 draws per tested
    row, at least 100 and at most 300 (the cost grows with every draw). The smallest attainable
    adjusted p is n_tests / (draws + 1): about 0.04 or less for up to 12 rows, up to 0.05 for 13
    to 15 rows, and above 0.05 from 16 rows on, where the cap applies and run_audit warns."""
    return int(min(300, max(100, 25 * n_tests)))


def missingness_mechanism(df, spec, units, n_perm=20, seed=2026, n_jobs=None, rows_mask=None,
                          min_missing=20, min_groups=10, jaccard=0.9, max_input_missing=0.2,
                          base_mask=None, rule=None) -> pd.DataFrame:
    """One row per (field block, input variant), plus one row per skipped field.

    n_perm: label draws per test, or "auto" (auto_n_perm of the number of tested rows).
    rows_mask: the rows the models use. With a label-dependent subsample, base_mask marks the
    eligible rows before subsampling and rule(labels over those rows) gives the subsample
    within them; the group-swap test then goes through the subsampling (see label_dependence)."""
    rows_mask = np.ones(len(df), bool) if rows_mask is None else np.asarray(rows_mask, bool)
    if rule is not None:
        base_mask = np.asarray(base_mask, bool)
        assert not (rows_mask & ~base_mask).any(), "rows_mask must lie inside base_mask"
    else:
        base_mask = rows_mask
    src = df[base_mask]                              # rows the permutations work on
    obs = rows_mask[base_mask]                       # of those, the rows the models use
    d = src[obs]
    units_src = np.asarray(units)[base_mask]
    y_src = src["label"].to_numpy()
    if not label_constant_within(units_src[obs], y_src[obs]):
        units_src = src["image_id"].to_numpy()       # label varies within units: images are the units
    # group size counted on all rows, not on the selected ones: a subsample that keeps every
    # malignant image would otherwise turn group size into a proxy for the label
    gsize_src = df.groupby("group_id")["image_id"].transform("size").to_numpy(float)[base_mask]
    acq_src = category_codes(src["acquisition"])
    fields = list(spec.fields)
    blocks, skipped, _ = _blocks(d, fields, min_missing, min_groups, jaccard)

    # patient-level strata for the group-swap test (constant within a group): the group's most
    # common acquisition, and the age band of its median age; the exact group size (on the rows
    # the swap works on) is added by stratified_group_swaps
    gid = src["group_id"].to_numpy()
    acq_mode = src.groupby("group_id")["acquisition"].agg(lambda s: s.astype(str).mode().iloc[0])
    coarse_src = pd.Series(gid).map(acq_mode).to_numpy().astype(str)
    age_col = next((f for f in spec.core_fields if f in spec.num_fields), None)
    if age_col:
        med = src.groupby("group_id")[age_col].median()
        band = pd.qcut(med, 5, labels=False, duplicates="drop").astype("string").fillna("na")
        age_band = pd.Series(gid).map(band).astype(str).to_numpy()
    else:
        age_band = np.full(len(src), "na")
    fine_src = np.char.add(np.char.add(coarse_src, "|"), age_band.astype(str))

    rows = [{"fields": f, "variant": "-", "skipped": reason} for f, reason in skipped.items()]
    jobs = []
    for block in blocks:
        target_src = src[block].isna().any(axis=1).to_numpy()
        others = [f for f in fields if f not in block]
        # when the age field is itself the target, its band cannot be a stratum: a patient whose
        # images all lack age has band "na", so the band would equal the target and the swap
        # could never move the link between label and missingness; size x acquisition only
        age_in_target = age_col is not None and age_col in block
        fine_blk = coarse_src if age_in_target else fine_src
        strata_name = "images per group x acquisition" + ("" if age_in_target or not age_col else " x age band")
        for variant in ["with_indicators", "values_only"]:
            if variant == "with_indicators":
                vals, cat = _encode(src, others, spec.num_fields)
                ind = [src[f].isna().to_numpy(float) for f in others]
                X = np.column_stack(vals + ind + [acq_src, gsize_src]) if others else np.c_[acq_src, gsize_src]
                cat_mask = cat + [False] * len(ind) + [True, False]
                inputs = others + [f"miss_{f}" for f in others] + ["acquisition", "group_size"]
                rmask = np.ones(len(src), bool)
            else:
                keep = [f for f in others if d[f].isna().mean() <= max_input_missing]
                rmask = src[keep].notna().all(axis=1).to_numpy() if keep else np.ones(len(src), bool)
                vals, cat = _encode(src, keep, spec.num_fields)
                X = np.column_stack(vals + [acq_src, gsize_src]) if keep else np.c_[acq_src, gsize_src]
                cat_mask = cat + [True, False]
                inputs = keep + ["acquisition", "group_size"]
            use = rmask & obs
            tgt = target_src[use]
            n_miss, n_grp = int(tgt.sum()), int(pd.Series(gid[use])[tgt].nunique())
            base = {"fields": "+".join(block), "variant": variant, "n_rows": int(use.sum()), "n_missing": n_miss,
                    "n_groups_missing": n_grp, "inputs": ", ".join(inputs)}
            if n_miss < min_missing or n_grp < min_groups or n_miss == use.sum():
                rows.append({**base, "skipped": "too few missing (or no observed) rows after selecting inputs"})
                continue
            resample = None
            if rule is not None:
                def rule_r(y_r, rmask=rmask):
                    y_all = y_src.copy()
                    y_all[rmask] = y_r
                    return np.asarray(rule(y_all), bool)[rmask]
                resample = {"X": X[rmask], "target": target_src[rmask], "y": y_src[rmask], "groups": gid[rmask],
                            "strata_fine": fine_blk[rmask], "strata_coarse": coarse_src[rmask], "rule": rule_r}
            # the tests run after the loop, once the number of tested rows (the Holm family) is known
            jobs.append((len(rows), dict(X=X[use], cat_mask=cat_mask, target=tgt, y=y_src[use], groups=gid[use],
                                         units=units_src[use], strata_fine=fine_blk[use],
                                         strata_coarse=coarse_src[use], resample=resample)))
            rows.append({**base, "group_swap_strata": strata_name, "skipped": None})
    n_draws = auto_n_perm(len(jobs)) if n_perm == "auto" else int(n_perm)
    for i, job in jobs:
        rows[i] = {**rows[i], **label_dependence(**job, n_perm=n_draws, seed=seed, n_jobs=n_jobs)}
    cols = ["fields", "variant", "n_rows", "n_missing", "n_groups_missing", "auc_a", "auc_b", "auc_gain",
            "fold_gain_mean", "fold_gain_sd", "crt_p", "crt_null_mean", "crt_null_q975", "phat_logloss",
            "const_logloss", "label_r2", "phat_rounds", "group_swap_p", "group_swap_null_mean",
            "group_swap_movable_share", "group_swap_degenerate", "group_swap_strata", "crt_p_holm",
            "group_swap_p_holm",
            "label_dependence_supported",
            "n_perm", "inputs", "skipped"]
    out = pd.DataFrame(rows).reindex(columns=cols)
    # several fields / variants are tested: Holm over the rows, per test; a claim needs both
    from .stats import holm
    for k in ("crt_p", "group_swap_p"):
        out[f"{k}_holm"] = holm(out[k].to_numpy(dtype=float))
    both = out[["crt_p_holm", "group_swap_p_holm"]]
    out["label_dependence_supported"] = [bool((row <= 0.05).all()) if row.notna().all() else None
                                         for _, row in both.iterrows()]
    return out
