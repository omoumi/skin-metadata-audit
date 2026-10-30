# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Intervals, multiple-testing adjustment and duplicate handling of the audit."""
import json

import numpy as np
import pandas as pd

from conftest import make_frame, make_spec
from skinaudit.audit import run_audit
from skinaudit.compare import normalized_gain_ci
from skinaudit.schema import AuditDataset
from skinaudit.shortcut import category_codes, corrected_cv_interval, run_feature_set


def test_corrected_cv_interval_formula():
    s = np.array([0.6, 0.62, 0.58, 0.61, 0.59, 0.63, 0.57, 0.6, 0.6, 0.6])
    se, tq, df = corrected_cv_interval(s, 0.25)
    assert df == 9 and abs(se - np.sqrt((1 / 10 + 0.25) * s.var(ddof=1))) < 1e-12
    assert 2.2 < tq < 2.3                      # t quantile, 9 degrees of freedom
    se2, _, _ = corrected_cv_interval(s, 0.25, extra_var=0.01)
    assert abs(se2 ** 2 - se ** 2 - 0.01) < 1e-12


def test_repeats_give_intervals_and_p_uses_first_split():
    df = make_frame(n_patients=80)
    X = category_codes(df["sex"])[:, None]
    y, g = df["label"].to_numpy(), df["group_id"].to_numpy()
    r = run_feature_set("x", X, [True], y, g, n_perm=4, units=df["lesion_id"].to_numpy(), n_jobs=1, n_repeats=3)
    assert r["n_repeats"] == 3 and r["bacc_ci_low"] < r["bacc_mean"] < r["bacc_ci_high"]
    assert r["gain_ci_low"] < r["gain"] < r["gain_ci_high"]
    assert abs(r["gain"] - (r["bacc_mean"] - r["null_mean"])) < 1e-12
    assert r["bacc_split1"] == np.mean(r["folds_bacc"]) or abs(r["bacc_split1"] - np.mean(r["folds_bacc"])) < 1e-3
    r1 = run_feature_set("x", X, [True], y, g, n_perm=4, units=df["lesion_id"].to_numpy(), n_jobs=1, n_repeats=1)
    # the p-value only depends on the first split, so it does not change with the repeats
    assert r1["p_perm"] == r["p_perm"] and r1["bacc_split1"] == r["bacc_split1"]


def test_conflicting_duplicate_keeps_lesion_null_and_leaves_images_out(tmp_path):
    df = make_frame()
    a = df[df.label == "A"].iloc[0]
    b = df[(df.label == "B") & (df.patient_id != a.patient_id)].iloc[0]
    pairs = pd.DataFrame({"image_id_1": [a.image_id], "image_id_2": [b.image_id], "reason": ["identical file"]})
    s = run_audit(AuditDataset(make_spec(), df, {}), str(tmp_path), n_perm=2, n_boot=0, make_figures=False,
                  image_stats=False, duplicate_pairs=pairs, n_jobs=1, mechanism=False, n_repeats=2)
    assert s["permutation"]["units"] == "lesion"
    assert set(s["permutation"]["used_per_set"].values()) == {"unit"}
    assert s["duplicates"]["pairs_with_conflicting_labels"] == 1
    assert s["model_labels"]["conflicting_duplicate_images_left_out"] == 2
    assert s["shortcut_cv"]["A_missing_indicators"]["n_rows"] == len(df) - 2
    assert s["duplicates"]["groups_after"] == s["duplicates"]["groups_before"] - 1   # groups still merged


def test_tables_label_naive_p_and_holm(tmp_path):
    df = make_frame()
    df.loc[np.random.default_rng(1).random(len(df)) < 0.2, "site"] = np.nan
    run_audit(AuditDataset(make_spec(), df, {}), str(tmp_path), n_perm=3, n_boot=0, make_figures=False,
              image_stats=False, n_jobs=1, mechanism=False, n_repeats=2)
    for t in ["missing_association", "missing_vs_other_fields", "value_association"]:
        cols = pd.read_csv(tmp_path / "tables" / f"{t}.csv").columns
        assert "p_naive" in cols and "p" not in cols
    cv = pd.read_csv(tmp_path / "tables" / "shortcut_cv.csv", index_col=0)
    assert (cv["p_perm_holm"] >= cv["p_perm"] - 1e-12).all() and (cv["p_perm_holm"] <= 1).all()
    s = json.load(open(tmp_path / "summary.json"))
    lo, hi = s["rubric"]["bacc_gain_ci"]["A_missing_indicators"]
    assert lo < s["rubric"]["informative_missingness_bacc_gain"] < hi


def test_rubric_smallest_class_uses_model_classes(tmp_path):
    df = make_frame()
    # one patient whose diagnosis is unspecified: a native class of one group that the
    # model-based tests never see
    df.loc[df["patient_id"] == "P0", "label"] = "UNSPECIFIED"
    spec = make_spec(classes=["A", "B", "C", "UNSPECIFIED"])
    s = run_audit(AuditDataset(spec, df, {}), str(tmp_path), n_perm=2, n_boot=0, make_figures=False,
                  image_stats=False, n_jobs=1, mechanism=False, n_repeats=1)
    assert s["complete_case"]["complete_core_min_groups_per_class"] == 1          # descriptive: native classes
    assert s["rubric"]["min_class_groups_complete_core"] > 10                      # rubric: model classes


def test_normalized_gain_ci():
    lo, hi = normalized_gain_ci({"gain_ci_low": 0.05, "gain_ci_high": 0.15, "null_mean": 0.5})
    assert abs(lo - 0.1) < 1e-12 and abs(hi - 0.3) < 1e-12
    assert np.isnan(normalized_gain_ci({})[0])
