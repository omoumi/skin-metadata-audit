# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Small synthetic datasets for fast tests (no real data needed)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skinaudit import shortcut  # noqa: E402
from skinaudit.schema import AuditDataset, AuditSpec, LicenseInfo  # noqa: E402

CLASSES = ["A", "B", "C"]


@pytest.fixture(autouse=True)
def fast_model(monkeypatch):
    """Fewer boosting rounds: tests check behaviour, not the final numbers."""
    monkeypatch.setattr(shortcut, "MODEL_PARAMS", dict(class_weight="balanced", max_iter=40, early_stopping=False))


def make_frame(n_patients=120, seed=0, size_label_link=False):
    """Patients with 1-4 lesions, lesions with 1-3 images, one label per lesion.

    size_label_link=True gives class A patients many more images, the pattern
    seen in real datasets (one class whose patients have many images).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        base = rng.choice(CLASSES)
        n_les = (rng.integers(4, 7) if (size_label_link and base == "A") else rng.integers(1, 3))
        for l in range(n_les):
            lab = base if rng.random() < 0.7 else rng.choice(CLASSES)
            for i in range(rng.integers(1, 3)):
                rows.append({"image_id": f"img_{p}_{l}_{i}", "patient_id": f"P{p}", "lesion_id": f"L{p}_{l}",
                             "label": lab})
    df = pd.DataFrame(rows)
    df["group_id"] = df["patient_id"]
    df["acquisition"] = rng.choice(["cam1", "cam2"], len(df))
    df["age_approx"] = rng.integers(20, 85, len(df)).astype(float)
    df["sex"] = rng.choice(["male", "female"], len(df))
    df["site"] = rng.choice(["trunk", "head", "leg"], len(df))
    return df


def make_spec(**kw):
    lic = LicenseInfo(name="CC-BY 4.0", url="https://creativecommons.org/licenses/by/4.0/",
                      allows_redistribution=True, non_commercial=False, citations=["Synthetic data."])
    args = dict(name="synthetic", title="Synthetic", num_fields=["age_approx"], cat_fields=["sex", "site"],
                core_fields=["age_approx", "sex"], group_level="patient", acquisition_name="camera",
                license=lic, classes=CLASSES)
    args.update(kw)
    return AuditSpec(**args)


@pytest.fixture
def synthetic_ds():
    df = make_frame()
    rng = np.random.default_rng(1)
    df.loc[rng.random(len(df)) < 0.15, "site"] = np.nan
    df.loc[rng.random(len(df)) < 0.05, "age_approx"] = np.nan
    return AuditDataset(make_spec(), df, {"note": "synthetic"})
