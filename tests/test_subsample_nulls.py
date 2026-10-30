# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Nulls on a label-dependent subsample, grouping with partial ids, and other safeguards."""
import numpy as np
import pandas as pd

from conftest import make_frame, make_spec
from skinaudit.adapters.isic_archive import linked_groups
from skinaudit.adapters.slice3d import subsample_rule
from skinaudit.mechanism import label_dependence
from skinaudit.shortcut import category_codes, run_feature_set


def _tiles(seed, n_pat=120):
    """SLICE-3D style: many tiles per patient, rare malignant tiles, a camera per patient
    that raises the malignancy risk (a real patient-level shortcut)."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        cam = int(rng.integers(0, 4))
        risk = 0.05 if cam == 3 else 0.003
        for i in range(int(rng.integers(25, 60))):
            rows.append((f"P{p}", cam, "malignant" if rng.random() < risk else "benign"))
    return pd.DataFrame(rows, columns=["patient_id", "cam", "label"])


def test_subsample_rule_is_a_function_of_the_labels():
    d = _tiles(0)
    rule = subsample_rule(d["patient_id"], benign_cap=20, seed=1)
    m = rule(d["label"].to_numpy())
    assert m[d["label"].to_numpy() == "malignant"].all()
    assert (d[m & (d["label"] == "benign")].groupby("patient_id").size() <= 20).all()
    assert np.array_equal(rule(d["label"].to_numpy()), m)       # deterministic


def test_resampled_group_null_is_not_degenerate_on_a_subsample():
    """On the subsample, patients of equal size have equal malignant counts, so swapping them
    changes nothing: the plain group null equals the observed score (p = 1). Re-applying the
    subsample rule inside each permutation gives a null that can move."""
    d = _tiles(1)
    y_all = d["label"].to_numpy()
    rule = subsample_rule(d["patient_id"], benign_cap=20, seed=1)
    m = rule(y_all)
    X_all = d[["cam"]].to_numpy(float)
    g_all = d["patient_id"].to_numpy()
    units_all = np.arange(len(d))
    args = dict(name="cam", cat_mask=[True], n_perm=9, n_perm_reference=3, n_jobs=1, n_repeats=1)
    plain = run_feature_set(X=X_all[m], y=y_all[m], groups=g_all[m], units=units_all[m], **args)
    assert plain["test_null"] == "group" and plain["null_group_degenerate"] and plain["p_perm"] == 1.0
    res = run_feature_set(X=X_all[m], y=y_all[m], groups=g_all[m], units=units_all[m],
                          resample={"X": X_all, "y": y_all, "groups": g_all, "units": units_all, "rule": rule}, **args)
    assert res["resampled_null"] and not res["null_group_degenerate"]
    assert res["p_perm"] < 0.2 and res["bacc_split1"] == plain["bacc_split1"]


def test_resampled_group_swap_in_the_mechanism_test():
    d = _tiles(2)
    rng = np.random.default_rng(2)
    y_all = d["label"].to_numpy()
    rule = subsample_rule(d["patient_id"], benign_cap=20, seed=1)
    m = rule(y_all)
    g_all = d["patient_id"].to_numpy()
    # missingness decided per patient, more often for patients with a malignant tile
    has_mal = d.groupby("patient_id")["label"].transform(lambda s: (s == "malignant").any()).to_numpy()
    u = d.groupby("patient_id")["patient_id"].transform(lambda s: rng.random()).to_numpy()
    target = (u < np.where(has_mal, 0.7, 0.2)).astype(int)
    X_all = np.c_[d["cam"].to_numpy(float)]
    strata = np.zeros(len(d)).astype(str)
    common = dict(n_perm=9, seed=1, n_jobs=1, method=("group_swap",))
    plain = label_dependence(X_all[m], [True], target[m], y_all[m], g_all[m], np.arange(len(d))[m].astype(str),
                             strata_fine=strata[m], strata_coarse=strata[m], **common)
    assert plain["group_swap_degenerate"]
    res = label_dependence(X_all[m], [True], target[m], y_all[m], g_all[m], np.arange(len(d))[m].astype(str),
                           resample={"X": X_all, "target": target, "y": y_all, "groups": g_all,
                                     "strata_fine": strata, "strata_coarse": strata, "rule": rule}, **common)
    assert not res["group_swap_degenerate"]


def test_linked_groups_join_partial_ids():
    pid = pd.Series(["P1", None, None, None, "P2"])
    lid = pd.Series(["L1", "L1", "L2", None, None])
    iid = pd.Series(["a", "b", "c", "d", "e"])
    g = linked_groups(pid, lid, iid).tolist()
    # b shares lesion L1 with a, which belongs to P1: one group
    assert g == ["P1", "P1", "lesion:L2", "image:d", "P2"]


def test_category_codes_cap():
    s = pd.Series([f"c{i}" for i in range(300)] + ["c0"] * 5 + [None])
    c = category_codes(s, max_categories=250)
    assert np.nanmax(c) <= 249 and np.isnan(c[-1])
    assert len(np.unique(c[~np.isnan(c)])) == 250


def test_unit_check_ignores_rows_left_out_of_the_tests(tmp_path):
    """A lesion whose untested (UNSPECIFIED) image has another label must not switch the
    whole dataset to patient-level units."""
    from skinaudit.audit import run_audit
    from skinaudit.schema import AuditDataset
    df = make_frame(n_patients=60, seed=3)
    les = df["lesion_id"].iloc[0]
    df.loc[df.index[(df["lesion_id"] == les)][:1], "label"] = "UNSPECIFIED"
    extra = df[df["lesion_id"] == les].iloc[:1].assign(image_id="extra_img", label="UNSPECIFIED")
    df = pd.concat([df, extra], ignore_index=True)
    df.loc[df["lesion_id"] == les, "label"] = ["UNSPECIFIED"] + ["A"] * (int((df["lesion_id"] == les).sum()) - 1)
    s = run_audit(AuditDataset(make_spec(classes=["A", "B", "C", "UNSPECIFIED"]), df, {}), str(tmp_path), n_perm=2,
                  n_boot=0, make_figures=False, image_stats=False, n_jobs=1, mechanism=False, n_repeats=1)
    assert s["permutation"]["units"] == "lesion"


def test_auto_draws_for_the_mechanism_holm_family():
    from skinaudit.mechanism import auto_n_perm
    assert auto_n_perm(2) == 100 and auto_n_perm(8) == 200 and auto_n_perm(11) == 275 and auto_n_perm(40) == 300
    for m in (1, 4, 8, 12):
        assert m / (auto_n_perm(m) + 1) < 0.05


def test_dataset_mask_passed_by_hand_keeps_its_rule(tmp_path):
    from skinaudit.audit import run_audit
    from skinaudit.schema import AuditDataset
    d = _tiles(4, n_pat=60)
    d["image_id"] = [f"t{i}" for i in range(len(d))]
    d["group_id"] = d["patient_id"]
    d["lesion_id"] = "tile:" + d["image_id"]
    d["acquisition"] = "cam" + d["cam"].astype(str)
    rng = np.random.default_rng(0)
    d["age_approx"] = rng.normal(50, 10, len(d))
    d["sex"] = rng.choice(["m", "f"], len(d))
    d["site"] = rng.choice(["a", "b"], len(d))
    rule = subsample_rule(d["patient_id"], 20, 1)
    ds = AuditDataset(make_spec(classes=["benign", "malignant"]), d, {}, cv_rows_mask=rule(d["label"].to_numpy()),
                      cv_rule=rule)
    s = run_audit(ds, str(tmp_path), n_perm=2, n_boot=0, make_figures=False, image_stats=False, n_jobs=1,
                  mechanism=False, n_repeats=1, cv_rows_mask=ds.cv_rows_mask)
    assert s["cv_subsample"]["rule_reapplied"] is True
