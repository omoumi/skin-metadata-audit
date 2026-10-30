# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Generic ISIC Archive adapter on a hand-made metadata.csv (no download)."""
import pandas as pd
import pytest
from PIL import Image

from skinaudit.adapters import isic_archive
from skinaudit.schema import CLASSES_8


def _write(tmp_path, extra_col=None):
    rows = []
    paths = [("Benign", "Benign melanocytic proliferations", "Nevus"),
             ("Malignant", "Malignant melanocytic proliferations (Melanoma)", "Melanoma in situ"),
             ("Indeterminate", "Indeterminate melanocytic proliferations", "Atypical melanocytic neoplasm"),
             ("Benign", "Benign epidermal proliferations", "Solar lentigo")]
    for i in range(80):
        d1, d2, d3 = paths[i % 4]
        r = {"isic_id": f"ISIC_{i:07d}", "attribution": ["Clinic A", "Clinic B"][i % 2], "copyright_license": "CC-BY",
             "age_approx": str(30 + i % 40), "sex": ["male", "female"][i % 2], "anatom_site_1": "Trunk",
             "clin_size_long_diam_mm": "4.0", "fitzpatrick_skin_type": "" if i % 5 == 0 else "II",
             "diagnosis_1": d1, "diagnosis_2": d2, "diagnosis_3": d3, "image_type": "dermoscopic",
             "patient_id": f"IP_{i // 2}" if i < 40 else "", "lesion_id": f"IL_{i}" if i < 60 else ""}
        if extra_col:
            r[extra_col] = "x"
        rows.append(r)
        Image.new("RGB", (40, 30)).save(tmp_path / f"{r['isic_id']}.jpg")
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)


def test_generic_collection(tmp_path):
    _write(tmp_path)
    ds = isic_archive.load(218, str(tmp_path))
    assert ds.spec.title == "PROVe-AI" and ds.spec.taxonomy == "isic_archive"
    assert ds.spec.num_fields == ["age_approx", "clin_size_long_diam_mm"]
    assert ds.spec.cat_fields == ["sex", "anatom_site_1", "fitzpatrick_skin_type"]
    assert set(ds.df["label"]) == {"NV", "MEL", "OTHER", "BKL"}
    assert ds.df.loc[ds.df["label"] == "OTHER", "label_ambiguous"].all()
    assert ds.spec.group_level.startswith("patient (") and ds.spec.acquisition_name == "institution"
    g = ds.df.set_index("image_id")["group_id"]
    assert g["ISIC_0000000"] == "IP_0" and g["ISIC_0000050"] == "lesion:IL_50" and g["ISIC_0000070"] == "image:ISIC_0000070"


def test_unknown_column_and_allowed_labels_stop(tmp_path):
    _write(tmp_path, extra_col="new_field")
    with pytest.raises(AssertionError, match="new_field"):
        isic_archive.load(999, str(tmp_path))
    (tmp_path / "metadata.csv").unlink()
    _write(tmp_path)
    with pytest.raises(AssertionError, match="outside"):
        isic_archive.load(999, str(tmp_path), allowed_labels=CLASSES_8)


def test_lesion_ids_without_patient_ids(tmp_path):
    _write(tmp_path)
    df = pd.read_csv(tmp_path / "metadata.csv", dtype=str, keep_default_na=False)
    df["patient_id"] = ""
    df.to_csv(tmp_path / "metadata.csv", index=False)
    ds = isic_archive.load(218, str(tmp_path))
    assert ds.spec.group_level == "lesion (image where missing)"
    g = ds.df.set_index("image_id")["group_id"]
    assert g["ISIC_0000000"] == "lesion:IL_0" and g["ISIC_0000070"] == "image:ISIC_0000070"
