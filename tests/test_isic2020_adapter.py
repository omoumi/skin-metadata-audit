# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""ISIC 2020 adapter on tiny hand-made inputs: official CSV and Kaggle competition CSV."""
import pandas as pd
import pytest
from PIL import Image

from skinaudit.adapters import isic2020
from skinaudit.audit import run_audit


@pytest.mark.parametrize("csv_name,with_lesion", [("ISIC_2020_Training_GroundTruth_v2.csv", True),
                                                  ("train.csv", False)])
def test_load_both_layouts(tmp_path, csv_name, with_lesion):
    names = [f"ISIC_{i:07d}" for i in range(40)]
    (tmp_path / "jpeg" / "train").mkdir(parents=True)
    for i, n in enumerate(names):
        Image.new("RGB", (64, 48) if i % 2 else (48, 64), (100 + i, 80, 60)).save(tmp_path / "jpeg" / "train" / f"{n}.jpg")
    dx = ["unknown", "nevus", "melanoma", "lentigo NOS"]
    gt = pd.DataFrame({"image_name": names, "patient_id": [f"IP_{i // 3}" for i in range(40)],
                       "sex": ["male", "female", ""] * 13 + ["male"], "age_approx": ["45", "", "60", "30"] * 10,
                       "anatom_site_general_challenge": ["torso", "head/neck", "", "torso"] * 10,
                       "diagnosis": [dx[i % 4] for i in range(40)],
                       "benign_malignant": ["malignant" if i % 4 == 2 else "benign" for i in range(40)],
                       "target": ["1" if i % 4 == 2 else "0" for i in range(40)]})
    if with_lesion:
        gt.insert(2, "lesion_id", [f"IL_{i}" for i in range(40)])
    gt.to_csv(tmp_path / csv_name, index=False)
    pd.DataFrame({"image_name_1": [names[0]], "image_name_2": [names[1]]}).to_csv(tmp_path / isic2020.DUP_FILE,
                                                                                  index=False)
    ds = isic2020.load(str(tmp_path), min_resolution_group=1)
    assert len(ds.df) == 39 and ds.inputs["duplicates_dropped"] == 1
    assert ds.inputs["has_lesion_id"] == with_lesion
    s = run_audit(ds, str(tmp_path / "out"), n_perm=1, n_boot=5, make_figures=False, n_jobs=1)
    assert s["permutation"]["units"] == ("lesion" if with_lesion else "patient")
    assert s["harmonized_classes"]["level2"]["images"]["UNSPECIFIED"] > 0
    assert "diagnosis" in s["excluded_columns_v_corrected"]
