# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""SLICE-3D helpers on tiny hand-made frames (no real data needed)."""
import numpy as np
import pandas as pd

from skinaudit.adapters.slice3d import _classify, cv_subsample
from skinaudit.taxonomy import harmonize


def test_cv_subsample_keeps_cases_and_caps_patients():
    """All malignant tiles kept; benign tiles capped per patient whether or not they
    have a specific diagnosis (keeping those would over-sample suspicious lesions)."""
    rows = []
    for p in range(5):
        for i in range(30):
            rows.append({"patient_id": f"P{p}", "label": "malignant" if (p == 0 and i == 0) else "benign",
                         "has_specific_dx": p == 1 and i < 3})
    df = pd.DataFrame(rows)
    m = cv_subsample(df, benign_cap=4, seed=1)
    assert m[(df["label"] == "malignant").to_numpy()].all()
    benign = df[df["label"] == "benign"]
    assert (benign[m[benign.index]].groupby("patient_id").size() <= 4).all()
    # patient P1 has 3 specific-diagnosis tiles among 30: they are not all forced in
    assert m[df["has_specific_dx"].to_numpy()].sum() <= 3
    assert (cv_subsample(df, 4, 1) == m).all() and not (cv_subsample(df, 4, 2) == m).all()


def test_dataset_mask_is_used_by_run_audit(tmp_path):
    from conftest import make_frame, make_spec
    from skinaudit.audit import run_audit
    from skinaudit.schema import AuditDataset
    df = make_frame(n_patients=60, seed=3)
    mask = np.zeros(len(df), bool)
    mask[: len(df) // 2] = True
    ds = AuditDataset(make_spec(), df, cv_rows_mask=mask).validate()
    s = run_audit(ds, str(tmp_path / "o"), n_perm=1, n_boot=5, image_stats=False, make_figures=False, n_jobs=1)
    assert s["cv_subsample"]["n_rows"] == int(mask.sum())
    assert all(v["n_rows"] <= mask.sum() for v in s["shortcut_cv"].values())


def test_column_classification():
    assert _classify("iddx_3").startswith("label") and _classify("diagnosis_full").startswith("label")
    assert _classify("tbp_lv_areaMM2").startswith("TBP") and _classify("mel_thick_mm").startswith("label leak")
    assert _classify("tbp_lv_dnn_lesion_confidence").startswith("label leak")
    assert _classify("age_approx") is None


def test_isic_dx_harmonize_on_repeated_paths():
    df = pd.DataFrame({"d1": ["Benign"] * 3 + ["Malignant"], "d2": [np.nan, np.nan, "Benign epidermal proliferations",
                                                                   "Malignant epidermal proliferations"],
                       "d3": [np.nan, np.nan, "Pigmented benign keratosis", "Squamous cell carcinoma in situ"]})
    h = harmonize("slice3d", df, ["d1", "d2", "d3"])
    assert h["l2"].tolist() == ["UNSPECIFIED", "UNSPECIFIED", "BKL", "SCC"]
