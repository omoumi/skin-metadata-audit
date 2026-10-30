# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
import numpy as np
import pandas as pd
import pytest

from conftest import make_frame, make_spec
from skinaudit.adapters.milk10k import CLASSES_MILK
from skinaudit.audit import run_audit
from skinaudit.compare import build_comparison
from skinaudit.schema import CLASSES_8, AuditDataset
from skinaudit.taxonomy import (LEVEL1, LEVEL2, NATIVE_MAPS, from_text, harmonize, harmonized_summary,
                                mapping_table)


def test_every_native_target_is_valid():
    for d, m in NATIVE_MAPS.items():
        for k, t in m.items():
            assert t.l2 in LEVEL2 and t.l1 in LEVEL1, (d, k)


def test_adapter_classes_are_all_mapped():
    assert set(CLASSES_8) == set(NATIVE_MAPS["isic2019"]) == set(NATIVE_MAPS["hiba"])
    assert set(CLASSES_MILK) == set(NATIVE_MAPS["milk10k"])


def test_conventions_hold_in_every_dataset():
    for d, m in NATIVE_MAPS.items():
        for k, t in m.items():
            if t.l2 == "AK":
                assert t.l1 == "indeterminate", (d, k)
            if t.l2 in ("MEL", "BCC", "SCC"):
                assert t.l1 == "malignant", (d, k)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        harmonize("isic2019", pd.DataFrame({"label": ["MEL", "XYZ"]}))
    with pytest.raises(KeyError):
        harmonize("not_a_dataset", pd.DataFrame({"label": ["MEL"]}))


@pytest.mark.parametrize("path,l2,l1", [
    (["Benign", "Benign melanocytic proliferations", "Nevus", "Nevus, Atypical"], "NV", "benign"),
    (["Indeterminate", "Indeterminate melanocytic proliferations", "Atypical melanocytic neoplasm"], "OTHER",
     "indeterminate"),
    (["Indeterminate", "Indeterminate epidermal proliferations", "Solar or actinic keratosis"], "AK", "indeterminate"),
    (["Malignant", "Malignant epidermal proliferations", "Squamous cell carcinoma in situ"], "SCC", "malignant"),
    (["Benign", "Flat melanotic pigmentations - not melanocytic nevus"], "BKL", "benign"),
    (["Benign", "Flat melanotic pigmentations - not melanocytic nevus", "Solar lentigo"], "BKL", "benign"),
    (["Benign"], "UNSPECIFIED", "benign"),
    (["Benign", "Inflammatory or infectious diseases"], "OTHER", "benign"),
    (["Malignant", "Malignant melanocytic proliferations (Melanoma)", "Melanoma in situ"], "MEL", "malignant"),
    (["Malignant", "Malignant melanocytic proliferations (Melanoma)", "Lentigo maligna"], "MEL", "malignant"),
    (["Benign", "Benign epidermal proliferations", "Epidermal nevus"], "OTHER", "benign"),
    (["Benign", "Flat melanotic pigmentations - not melanocytic nevus", "Cafe au lait macule"], "OTHER", "benign"),
])
def test_isic_dx_paths(path, l2, l1):
    t = from_text(path)
    assert (t.l2, t.l1) == (l2, l1)


@pytest.mark.parametrize("group,l2,l1", [
    ("Benign melanocytic proliferations", "NV", "benign"),
    ("Benign epidermal proliferations", "BKL", "benign"),
    ("Benign soft tissue proliferations - Fibro-histiocytic", "DF", "benign"),
    ("Malignant adnexal epithelial proliferations - Follicular", "BCC", "malignant"),
    ("Malignant epidermal proliferations", "SCC", "malignant"),
    ("Indeterminate epidermal proliferations", "AK", "indeterminate"),
])
def test_group_only_paths_go_to_dominant_class_flagged(group, l2, l1):
    t = from_text([group.split()[0], group])
    assert (t.l2, t.l1, t.ambiguous) == (l2, l1, True)


def test_group_rules_do_not_catch_specific_diagnoses():
    # a level-3 diagnosis is matched by its own rule, not by the group rule
    t = from_text(["Benign", "Benign epidermal proliferations", "Seborrheic keratosis"])
    assert (t.l2, t.ambiguous) == ("BKL", False)
    t = from_text(["Indeterminate", "Indeterminate melanocytic proliferations"])
    assert (t.l2, t.l1) == ("OTHER", "indeterminate")


def test_atypical_intraepithelial_melanocytic_is_ambiguous():
    t = from_text(["Indeterminate", "Indeterminate melanocytic proliferations",
                   "Atypical intraepithelial melanocytic proliferation"])
    assert (t.l2, t.l1, t.ambiguous) == ("OTHER", "indeterminate", True)


def test_malignant_benign_clash_stops():
    with pytest.raises(ValueError):
        from_text(["Benign", "Benign melanocytic proliferations", "Melanoma"])


def test_ak_stays_indeterminate_even_if_dataset_says_benign():
    assert from_text(["benign", "actinic keratosis"]).l1 == "indeterminate"


def test_harmonized_summary_and_mapping_table():
    df = pd.DataFrame({"label": ["AKIEC", "MEL", "NV", "NV"], "g": ["a", "b", "c", "c"]})
    h = harmonize("milk10k", df)
    s = harmonized_summary(h, df["g"])
    assert s["ambiguous_share"] == 0.25
    assert s["level2"]["groups"]["NV"] == 1 and s["level1"]["images"]["malignant"] == 1
    assert {"isic2019", "hiba", "milk10k", "pad_ufes_20", "isic2020", "slice3d", "isic_archive"} <= \
        set(mapping_table()["dataset"])


def test_comparison_uses_shared_taxonomy(tmp_path):
    outs = []
    for i, (labels, key) in enumerate([(["MEL", "NV", "BCC"], "isic2019"), (["MEL", "NV", "AKIEC"], "milk10k")]):
        df = make_frame(n_patients=60, seed=20 + i)
        df["label"] = df["label"].map(dict(zip(["A", "B", "C"], labels)))
        ds = AuditDataset(make_spec(name=f"d{i}", title=f"D{i}", classes=labels, taxonomy=key), df)
        run_audit(ds, str(tmp_path / f"d{i}"), n_perm=1, n_boot=5, image_stats=False, make_figures=False, n_jobs=1)
        outs.append(str(tmp_path / f"d{i}" / "summary.json"))
    res = build_comparison(outs, str(tmp_path / "cmp"))
    # MILK's AKIEC and ISIC's classes land in one shared column set
    assert {"MEL", "NV", "BCC", "AK"} <= set(res["class_share"].columns)
    assert "Ambiguous label share" in res["rubric"].columns
    assert (tmp_path / "cmp" / "taxonomy_mapping.csv").exists()


def test_normalized_gain():
    from skinaudit.compare import normalized_gain
    assert abs(normalized_gain({"bacc_mean": 0.6, "null_mean": 0.5}) - 0.2) < 1e-12
    assert abs(normalized_gain({"bacc_mean": 0.3, "null_mean": 0.125}) - 0.2) < 1e-12
    assert np.isnan(normalized_gain({})) and np.isnan(normalized_gain({"bacc_mean": 0.3, "null_mean": None}))
