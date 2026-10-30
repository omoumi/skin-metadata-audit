# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Shortcut tests: how well can a feature set alone predict the label?

Each feature set is scored with grouped cross-validation (no group shared
between train and test) and compared with a null distribution built by
re-running the same pipeline on permuted labels.
"""
from __future__ import annotations

import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from .parallel import parallel_map, resolve_n_jobs

# 100 boosting rounds: the feature sets are small (1-20 columns), and every permutation refits
# the same model, so the rounds set most of the run time. Observed and permuted labels always
# use the same settings, so the number of rounds does not affect the validity of the tests,
# only their power. A single model class with fixed settings: a feature set without a
# significant gain may still carry signal that another model would find.
MODEL_PARAMS = dict(class_weight="balanced", max_iter=100, early_stopping=False)


@contextmanager
def _quiet_absent_classes():
    """Silence sklearn's 'y_pred contains classes not in y_true'. It fires when a rare class
    is absent from a test fold; balanced accuracy is then the mean recall over the classes
    present, and that case is recorded in min_classes_in_test_fold. Inside the function
    that scores, so it also holds in parallel worker processes."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
        yield


def _bacc(y_true, y_pred) -> float:
    with _quiet_absent_classes():
        return float(balanced_accuracy_score(y_true, y_pred))


MAX_CATEGORIES = 250


def category_codes(s: pd.Series, max_categories: int = MAX_CATEGORIES) -> np.ndarray:
    """Integer codes for a categorical column, NaN kept as NaN. Gradient boosting accepts at
    most 255 categories per feature, so beyond max_categories the rarest values share one
    code ('other')."""
    s = pd.Series(s)
    counts = s.value_counts()
    if len(counts) > max_categories:
        keep = set(counts.index[:max_categories - 1])
        s = s.where(s.isna() | s.isin(keep), "__other__")
    c = s.astype("category").cat.codes.to_numpy(dtype=float, copy=True)
    c[c < 0] = np.nan
    return c


def grouped_cv(X, y, groups, cat_mask, n_splits=5, seed=2026, model_params=None) -> list[dict]:
    """Per-fold balanced accuracy plus how many classes each test fold contains."""
    params = MODEL_PARAMS if model_params is None else model_params
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in cv.split(X, y, groups):
        clf = HistGradientBoostingClassifier(categorical_features=np.asarray(cat_mask),
                                             random_state=seed, **params)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        present = np.unique(y[te])
        folds.append({
            "bacc": _bacc(y[te], pred),
            "n_test": int(len(te)),
            "n_train": int(len(tr)),
            "n_classes_test": int(len(present)),
            "chance": 1.0 / len(present),
        })
    return folds


def make_group_permuter(groups):
    """Label permutation at the group level.

    Whole groups swap their label vectors with other groups of the same size, so
    labels stay clustered within patients/lesions exactly as in the real data. The vector
    a group receives is put in random order over its images, so no association between row
    order and the features survives into the null. Groups whose size is unique cannot move;
    their share is reported.
    """
    groups = np.asarray(groups)
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)
    by_size: dict[int, list] = {}
    for m in members:
        by_size.setdefault(len(m), []).append(m)
    blocks = [np.stack(ms) for ms in by_size.values()]
    movable_share = float(sum(b.size for b in blocks if len(b) > 1) / len(groups))

    def permute(y, rng):
        y = np.asarray(y)
        out = y.copy()
        for b in blocks:
            if len(b) > 1:
                out[b] = rng.permuted(y[b[rng.permutation(len(b))]], axis=1)
        return out
    return permute, movable_share


def label_constant_within(units, y) -> bool:
    """True if every unit (e.g. lesion) carries a single label."""
    return bool((pd.Series(np.asarray(y)).groupby(np.asarray(units)).nunique() == 1).all())


def make_unit_permuter(units, y):
    """Shuffle labels between units (e.g. lesions) whose images all share one label.

    Every image of a unit receives the unit's new label, so within-unit clustering is
    kept. Unlike size-matched group swaps, the link between unit size and label is
    broken, so signal that runs through group size is not absorbed into the null.
    """
    _, inv = np.unique(np.asarray(units), return_inverse=True)
    s = pd.Series(np.asarray(y))
    assert (s.groupby(inv).nunique() == 1).all(), "label must be constant within each unit"
    first = s.groupby(inv).first().to_numpy()

    def permute(_, rng):
        return first[rng.permutation(len(first))][inv]
    return permute, 1.0


def _null_summary(null: np.ndarray, prefix: str) -> dict:
    n = len(null)
    return {
        f"{prefix}mean": float(null.mean()) if n else np.nan,
        f"{prefix}sd": float(null.std(ddof=1)) if n > 1 else np.nan,
        f"{prefix}q025": float(np.quantile(null, 0.025)) if n else np.nan,
        f"{prefix}q975": float(np.quantile(null, 0.975)) if n else np.nan,
    }


def _cv_task(X, y_task, groups, cat_mask, n_splits, seed, model_params, keep_folds=False):
    """One grouped CV run: mean balanced accuracy, or the per-fold results (keep_folds=True).
    model_params travels with the task: worker processes re-import this module and would
    otherwise not see a MODEL_PARAMS changed at run time."""
    folds = grouped_cv(X, y_task, groups, cat_mask, n_splits, seed, model_params)
    return folds if keep_folds else float(np.mean([f["bacc"] for f in folds]))


REPEAT_SEED_STEP = 1000


def corrected_cv_interval(fold_scores, test_train_ratio, extra_var=0.0, alpha=0.05) -> tuple[float, float, float]:
    """Mean of repeated k-fold scores with the corrected variance of Nadeau and Bengio (2003)
    as used for repeated CV by Bouckaert and Frank (2004): (1/J + n_test/n_train) * S^2,
    J = number of fold scores, S^2 = their sample variance, with a t quantile on J - 1 degrees of
    freedom. The plain 1/J term alone would ignore that training sets overlap and give
    intervals that are too narrow. The correction is a heuristic (no unbiased estimator of the
    CV variance exists, Bengio and Grandvalet 2004); its coverage was checked by simulation
    (validation/README.md).
    extra_var is added to the variance (e.g. the variance of a null mean that is subtracted).
    Returns (standard error, t-quantile, degrees of freedom) for a 1 - alpha interval."""
    from scipy import stats as st
    s = np.asarray(fold_scores, float)
    j = len(s)
    if j < 2:
        return np.nan, np.nan, 0
    var = (1.0 / j + test_train_ratio) * s.var(ddof=1) + extra_var
    return float(np.sqrt(var)), float(st.t.ppf(1 - alpha / 2, j - 1)), j - 1


def _same_partition(units, groups) -> bool:
    """True when units and groups split the rows the same way (every group is one unit)."""
    pairs = pd.DataFrame({"u": np.asarray(units), "g": np.asarray(groups)})
    return bool((pairs.groupby("u")["g"].nunique() == 1).all() and (pairs.groupby("g")["u"].nunique() == 1).all())


def _p(null: np.ndarray, obs: float) -> float:
    return float((1 + (null >= obs).sum()) / (len(null) + 1)) if len(null) else np.nan


def run_feature_set(name, X, cat_mask, y, groups, n_perm=20, n_splits=5, seed=2026,
                    perm_level="unit", units=None, n_perm_reference=None, n_jobs=None, n_repeats=5,
                    resample=None) -> dict:
    """Observed grouped-CV score, two permutation nulls and the test for one feature set.

    Two nulls, each the same pipeline re-run on permuted labels:
    - unit null: one label per unit (lesion; defaults to the CV group) is shuffled and given
      to all images of that unit. It breaks every link between label and features, including
      the link that runs through how many images a patient has, so its mean is the chance
      level the gain is measured from. When several units (lesions) sit inside one CV group
      (patient), it also breaks the similarity of labels within a patient; with features that
      are constant within a patient (camera, missingness decided per patient) its spread is
      then too narrow and its p-value too small (a simulation with clustered labels gave about
      14% false positives at the 5% level; see validation/README.md).
    - group null: whole groups swap their label vectors with groups of the same size, so the
      clustering of labels is kept exactly. Valid as a test when patients of the same size are
      exchangeable under the null; it keeps the link between group size and label, so it tests
      for signal beyond group size (conservative when a feature works through group size).

    The test (p_perm, p_perm_holm in the audit) uses the unit null when units and groups are
    the same partition (then it is exact under exchangeable units) and the group null when units
    are nested inside groups. The other null is the reference: n_perm draws go to the test null and
    n_perm_reference (default n_perm) to the reference null. The gain is measured from the
    unit null mean, except when the label is not constant within units: then perm_level falls
    back to "group", there is no unit null, and the group null serves for both. "image" permutes single rows (not recommended).
    The group null is skipped when it is the same random process as the unit null (each group
    one unit, all groups of equal size).

    n_repeats: the observed score is the mean over n_repeats grouped CV runs with different
    fold assignments; its 95% interval and that of the gain use the corrected repeated-CV
    variance (corrected_cv_interval). Permutation p-values compare the first split alone with
    the nulls, which are built from single splits too, so the observed score and the null are
    computed the same way.

    resample: for tests that run on a label-dependent subsample (SLICE-3D: every malignant
    tile plus up to 20 benign tiles per patient). A dict with the arrays of every eligible
    row before subsampling ("X", "y", "groups", "units") and "rule", a function that maps a
    label vector over those rows to the subsample mask; X, y, groups above are the rows the
    rule picks for the real labels. Each null then permutes the labels of all eligible rows
    (sizes matched on the full patients) and re-applies the rule, the way the real data went
    through it. Without this, patients with the same subsample size also have the same number
    of malignant tiles, so the size-matched null cannot move any patient-level signal and its
    p-value is 1 whatever the effect.

    n_jobs: processes for the permutation fits (None = every CPU this session may use).
    All permuted labels are drawn first, in the main process and in the same order as a
    sequential run, so the result does not depend on n_jobs.
    """
    assert perm_level in ("unit", "group", "image")
    if resample is not None:
        Xb, yb, gb = resample["X"], np.asarray(resample["y"]), np.asarray(resample["groups"])
        ub = gb if resample.get("units") is None else np.asarray(resample["units"])
        rule = resample["rule"]
        assert int(np.asarray(rule(yb), bool).sum()) == len(y), "rule(real labels) must select the rows passed as X, y"
    else:
        Xb, yb, gb, rule = X, np.asarray(y), np.asarray(groups), None
        ub = None if units is None else np.asarray(units)
    n_ref = n_perm if n_perm_reference is None else int(n_perm_reference)
    t = time.time()
    folds = grouped_cv(X, y, groups, cat_mask, n_splits, seed)
    obs = float(np.mean([f["bacc"] for f in folds]))      # first split: the one the p-values use
    n_repeats = max(1, int(n_repeats))
    units = gb if ub is None else ub
    if perm_level == "unit" and not label_constant_within(units, yb):
        perm_level = "group"
    skipped = None
    if perm_level == "unit":
        same = _same_partition(units, gb)
        test = "unit" if same else "group"
        n_unit = n_perm if test == "unit" else max(n_ref, 2)
        n_group = n_ref if test == "unit" else n_perm
        if same and pd.Series(gb).value_counts().nunique() == 1:
            # each group is one unit and all groups have equal size: the size-matched group
            # permutation is the same random process as the unit permutation
            n_group, skipped = 0, "redundant: each group is one unit and all groups have equal size"
        unit_permute, movable_share = make_unit_permuter(units, yb)
    elif perm_level == "group":
        test, n_unit, n_group = "group", 0, n_perm
        unit_permute, movable_share = None, np.nan
    else:
        test, n_unit, n_group = "image", n_perm, 0
        unit_permute, movable_share = (lambda yy, r: r.permutation(yy)), 1.0
    n_jobs = resolve_n_jobs(n_jobs)
    rng = np.random.default_rng(seed)
    y_unit = [unit_permute(yb, rng) for _ in range(n_unit)]
    y_group, g_share = [], np.nan
    if n_group:
        g_permute, g_share = make_group_permuter(gb)
        rng_g = np.random.default_rng(seed + 1)
        y_group = [g_permute(yb, rng_g) for _ in range(n_group)]
    params = dict(MODEL_PARAMS)
    repeat_tasks = [(X, y, groups, cat_mask, n_splits, seed + REPEAT_SEED_STEP * r, params, True)
                    for r in range(1, n_repeats)]

    def null_task(yp):
        if rule is None:
            return (X, yp, groups, cat_mask, n_splits, seed, params)
        m = np.asarray(rule(yp), bool)                  # the subsample the permuted labels would give
        return (Xb[m], yp[m], gb[m], cat_mask, n_splits, seed, params)
    results = parallel_map(_cv_task, repeat_tasks + [null_task(yp) for yp in y_unit + y_group], n_jobs)
    all_folds = [folds] + results[:len(repeat_tasks)]
    scores = results[len(repeat_tasks):]
    null_u = np.asarray(scores[:n_unit], float)
    null_g = np.asarray(scores[n_unit:], float)
    # the gain is measured from the unit null; without one (labels vary within units) the group null
    centre = null_u if n_unit else null_g
    fold_scores = np.array([f["bacc"] for run in all_folds for f in run])
    ratio = float(np.mean([f["n_test"] / f["n_train"] for run in all_folds for f in run]))
    bacc_mean = float(fold_scores.mean())
    se_b, tq, _ = corrected_cv_interval(fold_scores, ratio)
    null_var = float(centre.var(ddof=1) / len(centre)) if len(centre) > 1 else 0.0
    se_g, tq_g, _ = corrected_cv_interval(fold_scores, ratio, extra_var=null_var)
    null_mean = float(centre.mean()) if len(centre) else np.nan
    gain = bacc_mean - null_mean
    n_classes = len(np.unique(y))
    p_unit, p_group = _p(null_u, obs), _p(null_g, obs)
    # a null that never moves (every draw scores the same) cannot test anything
    deg_u = bool(len(null_u) > 1 and np.ptp(null_u) == 0)
    deg_g = bool(len(null_g) > 1 and np.ptp(null_g) == 0)
    if (test == "group" and deg_g) or (test in ("unit", "image") and deg_u):
        print(f"WARNING: {name}: the {test} null gave the same score in every permutation, so its p-value "
              "carries no information (the permutation cannot change the label pattern the features see, or "
              "the model ignores the features)", flush=True)
    group = {"n_perm_group": int(n_group), **_null_summary(null_g, "null_group_"),
             "p_perm_group": p_group, "group_movable_share": round(g_share, 4) if n_group else np.nan,
             "null_group_degenerate": deg_g,
             "sensitivity_skipped": skipped}
    return {
        "feature_set": name,
        "n_rows": int(len(y)),
        "n_groups": int(len(np.unique(groups))),
        "n_classes": int(n_classes),
        "bacc_mean": bacc_mean,
        "bacc_sd": float(fold_scores.std(ddof=1)) if len(fold_scores) > 1 else np.nan,
        "n_repeats": n_repeats,
        "bacc_repeat_sd": float(np.std([np.mean([f["bacc"] for f in run]) for run in all_folds], ddof=1))
                          if n_repeats > 1 else np.nan,
        "bacc_se_corrected": se_b,
        "bacc_ci_low": bacc_mean - tq * se_b if not np.isnan(se_b) else np.nan,
        "bacc_ci_high": bacc_mean + tq * se_b if not np.isnan(se_b) else np.nan,
        "bacc_split1": obs,
        "gain": gain,
        "gain_ci_low": gain - tq_g * se_g if len(centre) and not np.isnan(se_g) else np.nan,
        "gain_ci_high": gain + tq_g * se_g if len(centre) and not np.isnan(se_g) else np.nan,
        "folds_bacc": [round(f["bacc"], 4) for f in folds],
        "min_classes_in_test_fold": int(min(f["n_classes_test"] for run in all_folds for f in run)),
        "every_fold_has_all_classes": bool(all(f["n_classes_test"] == n_classes for run in all_folds for f in run)),
        "test_null": test,
        "n_perm": int(n_perm),
        "perm_level": perm_level,
        "n_perm_unit": int(n_unit),
        "n_perm_units": int(len(np.unique(units))) if perm_level == "unit" else None,
        "perm_movable_share": round(movable_share, 4) if not np.isnan(movable_share) else np.nan,
        # null_* = the null the gain is measured from (unit null; group null when there is none)
        "null_mean": null_mean,
        "null_sd": float(centre.std(ddof=1)) if len(centre) > 1 else np.nan,
        "null_q025": float(np.quantile(centre, 0.025)) if len(centre) else np.nan,
        "null_q975": float(np.quantile(centre, 0.975)) if len(centre) else np.nan,
        "p_perm": p_unit if test in ("unit", "image") else p_group,
        "p_perm_unit": p_unit,
        "null_degenerate": deg_u,
        "resampled_null": rule is not None,
        # raw null scores for the null-distribution figure (the audit moves them to their own table)
        "null_scores": [round(float(v), 5) for v in null_u] if n_unit else [],
        "null_group_scores": [round(float(v), 5) for v in null_g],
        **group,
        "chance": 1.0 / n_classes,
        "n_jobs": int(n_jobs),
        "seconds": round(time.time() - t, 1),
    }


def explain_feature_set(X, y, groups, cat_mask, names, n_splits=5, seed=2026, n_repeats=10):
    """Which features carry the signal, and which classes it separates.

    Same grouped folds and model as grouped_cv. Returns
    - importance: permutation importance of each feature on the held-out fold
      (drop in balanced accuracy when that column is shuffled), mean and SD over folds;
    - oof: out-of-fold predicted label for every row (for a confusion matrix).
    """
    from sklearn.inspection import permutation_importance

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    imps = []
    for k, (tr, te) in enumerate(cv.split(X, y, groups)):
        clf = HistGradientBoostingClassifier(categorical_features=np.asarray(cat_mask),
                                             random_state=seed, **MODEL_PARAMS)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])
        with _quiet_absent_classes():
            r = permutation_importance(clf, X[te], y[te], scoring="balanced_accuracy", n_repeats=n_repeats,
                                       random_state=seed + k)
        imps.append(r.importances_mean)
    imps = np.asarray(imps)
    importance = pd.DataFrame({"feature": names, "importance_mean": imps.mean(0),
                               "importance_sd": imps.std(0)}).sort_values("importance_mean", ascending=False)
    return importance.set_index("feature"), oof
