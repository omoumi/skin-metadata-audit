# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""End-to-end run on synthetic data: every output is written and summary.json is consistent."""
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from skinaudit.audit import run_audit
from skinaudit.compare import build_comparison
from skinaudit.gallery import make_gallery, select_samples
from skinaudit.report import write_readme


def _add_images(ds, folder):
    rng = np.random.default_rng(0)
    paths = []
    for _, r in ds.df.iterrows():
        p = os.path.join(folder, f"{r.image_id}.jpg")
        w, h = (64, 48) if r.acquisition == "cam1" else (48, 64)
        Image.new("RGB", (w, h), tuple(int(x) for x in rng.integers(30, 220, 3))).save(p)
        paths.append(p)
    ds.df["image_path"] = paths
    ds.df["width"] = [Image.open(p).size[0] for p in paths]
    ds.df["height"] = [Image.open(p).size[1] for p in paths]
    ds.df["file_bytes"] = [os.path.getsize(p) for p in paths]


def test_full_run(synthetic_ds, tmp_path):
    _add_images(synthetic_ds, tmp_path)
    out = tmp_path / "out"
    s = run_audit(synthetic_ds, str(out), n_perm=2, n_boot=10)

    assert s["permutation"]["units"] == "lesion"
    assert s["permutation"]["n_perm_reference"] == s["permutation"]["n_perm"] == 2
    assert set(s["permutation"]["used_per_set"].values()) == {"unit"}
    sens = s["rubric"]["permutation_sensitivity"]
    assert sens and all(v["gap_se"] is not None for v in sens.values())
    # both nulls for every set; lesions sit inside patients, so the size-matched null is the test
    assert set(s["permutation"]["test_null_per_set"].values()) == {"group"}
    assert s["permutation"]["holm_family"] == sorted(s["shortcut_cv"])
    assert set(sens) == set(s["shortcut_cv"])
    assert "I_image_border_stats" in s["shortcut_cv"]
    # consistency table uses multi-image groups only
    wg = s["structure"]["within_group"]["site"]
    assert wg["groups_with_several_images"] < s["structure"]["n_groups"]

    for f in s["figures"]:
        assert os.path.getsize(out / "figures" / f) > 1000
    assert (out / "figures" / "missing_vs_other_fields_within_label.png").exists()
    for f in ["complete_case_bias.png", "group_size.png", "feature_importance.png", "missingness_confusion.png",
              "categorical_by_class.png", "null_distributions.png", "missingness_mechanism.png",
              "complete_case_smd.png"]:
        assert f in s["figures"], f
    for t in ["permutation_nulls", "importance_A_missing_indicators", "confusion_A_missing_indicators",
              "complete_case_class_share", "group_size_by_class", "missing_rate_by_group_size",
              "missingness_mechanism", "complete_case_smd"]:
        assert (out / "tables" / f"{t}.csv").exists(), t
    nulls = pd.read_csv(out / "tables" / "permutation_nulls.csv")
    assert len(nulls) == 2 * 2 * len(s["shortcut_cv"])
    assert "null_scores" not in s["shortcut_cv"]["A_missing_indicators"]
    assert s["rubric"]["label_dependent_missingness"]["fields"] == "site"
    assert s["rubric"]["complete_core_max_abs_smd"]["max_abs_smd"] is not None
    with open(out / "summary.json") as fh:
        assert json.load(fh)["schema_version"] == "1.0"

    sel = select_samples(synthetic_ds.df, synthetic_ds.spec, n_per_class=2)
    g = make_gallery(sel, synthetic_ds.spec, str(out), size=48)
    assert (out / g["grid"]).exists() and (out / "samples" / "ATTRIBUTION.md").exists()

    readme = open(write_readme(str(out), gallery=g)).read()
    assert "Sensitivity to the permutation scheme" in readme and "SE of gap" in readme

    res = build_comparison([str(out / "summary.json")], str(tmp_path / "cmp"))
    assert len(res["rubric"]) == 1
    assert (tmp_path / "cmp" / "figures" / "field_availability.png").exists()
    assert (tmp_path / "cmp" / "figures" / "dataset_map.png").exists()
    assert res["field_availability"].shape[1] == 7


def test_compare_refuses_mixed_versions(synthetic_ds, tmp_path):
    import pytest
    out = tmp_path / "out"
    run_audit(synthetic_ds, str(out), n_perm=1, n_boot=5, image_stats=False, make_figures=False)
    with open(out / "summary.json") as fh:
        old = json.load(fh)
    old["versions"]["skinaudit"], old["dataset"]["title"] = "0.0.9", "Older run"
    with open(tmp_path / "old.json", "w") as fh:
        json.dump(old, fh)
    paths = [str(out / "summary.json"), str(tmp_path / "old.json")]
    with pytest.raises(ValueError):
        build_comparison(paths, str(tmp_path / "cmp"))
    assert len(build_comparison(paths, str(tmp_path / "cmp"), strict=False)["rubric"]) == 2


def test_content_license_follows_dataset_terms(tmp_path):
    from skinaudit.report import content_license, write_content_license
    assert content_license({"non_commercial": True, "allows_redistribution": True})["name"] == "CC BY-NC 4.0"
    assert content_license({"non_commercial": False, "allows_redistribution": True})["name"] == "CC BY 4.0"
    assert content_license({"non_commercial": False, "allows_redistribution": False})["url"] == ""
    p = write_content_license(str(tmp_path), "X", {"name": "CC-BY-NC 4.0", "url": "u", "non_commercial": True})
    assert "CC BY-NC 4.0" in open(p).read()


def test_small_classes_merged_for_model_tests_only(tmp_path):
    from conftest import make_frame, make_spec
    from skinaudit.schema import AuditDataset
    df = make_frame(n_patients=90, seed=4)
    df.loc[df["patient_id"].isin(["P1", "P2", "P3"]), "label"] = "RARE"
    df.loc[df["patient_id"].isin(["P4", "P5"]), "label"] = "UNSPECIFIED"
    spec = make_spec(classes=["A", "B", "C", "RARE", "UNSPECIFIED"])
    s = run_audit(AuditDataset(spec, df), str(tmp_path / "o"), n_perm=1, n_boot=5, image_stats=False,
                  make_figures=False, n_jobs=1, n_perm_mechanism=1)
    ml = s["model_labels"]
    assert ml["merged_into_other"] == {"RARE": 3} and ml["unspecified_images_left_out"] == int((df.label == "UNSPECIFIED").sum())
    assert ml["classes"] == ["A", "B", "C", "OTHER"]
    assert s["class_counts"]["RARE"] > 0                      # descriptive tables keep the native class
    n_model = int((df.label != "UNSPECIFIED").sum())
    assert s["shortcut_cv"]["A_missing_indicators"]["n_rows"] == n_model
    assert s["shortcut_cv"]["A_missing_indicators"]["n_classes"] == 4
    readme = open(write_readme(str(tmp_path / "o"))).read()
    assert "merged into OTHER (RARE: 3)" in readme


def test_readme_wording_follows_group_level(tmp_path):
    import json
    from conftest import make_frame, make_spec
    from skinaudit.schema import AuditDataset
    out = tmp_path / "o"
    run_audit(AuditDataset(make_spec(), make_frame(n_patients=60, seed=5)), str(out), n_perm=1, n_boot=5,
              image_stats=False, make_figures=False, n_jobs=1, n_perm_mechanism=1)
    s = json.load(open(out / "summary.json"))
    s["dataset"]["group_level"], s["permutation"]["units"] = "image", "image"
    json.dump(s, open(out / "summary.json", "w"))
    readme = open(write_readme(str(out))).read()
    for bad in ["of one image are", "inside one image", "Within-image", "whole patients", "Please cite"]:
        assert bad not in readme, bad
    assert "labels are swapped between images within strata" in readme
