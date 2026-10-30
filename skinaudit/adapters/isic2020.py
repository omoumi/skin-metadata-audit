# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""ISIC 2020 challenge training set (SIIM-ISIC melanoma) -> common schema.

Reads a copy attached as a Kaggle input: either the official ground-truth CSV
(ISIC_2020_Training_GroundTruth_v2.csv or ISIC_2020_Training_GroundTruth.csv) or the
Kaggle competition's train.csv, plus the training JPEGs.

The audit label is binary (target). The 'diagnosis' column is a label leak ('unknown'
means not biopsied and followed up as benign) and is used only for the shared taxonomy
and the leak check. The metadata has no source column, so the acquisition variable is
the resolution group (device proxy). Pairs in the official duplicates list are reduced
to their first image; the list is read from the inputs if present, otherwise fetched
from the public ISIC challenge page.
"""
from __future__ import annotations

import glob
import os
import urllib.request

import numpy as np
import pandas as pd

from ..io import image_props, index_images, md5, resolution_groups
from ..schema import AuditDataset, AuditSpec, LicenseInfo
from ..stats import normalize_missing

GT_CANDIDATES = ["ISIC_2020_Training_GroundTruth_v2.csv", "ISIC_2020_Training_GroundTruth.csv", "train.csv"]
DUP_FILE = "ISIC_2020_Training_Duplicates.csv"
DUP_URL = "https://isic-archive.s3.amazonaws.com/challenges/2020/" + DUP_FILE
CLASSES_2020 = ["benign", "malignant"]
SITE = "anatom_site_general_challenge"

LICENSE = LicenseInfo(
    name="CC-BY-NC 4.0",
    url="https://creativecommons.org/licenses/by-nc/4.0/",
    allows_redistribution=True,
    non_commercial=True,
    citations=["International Skin Imaging Collaboration. SIIM-ISIC 2020 Challenge Dataset. International Skin "
               "Imaging Collaboration; 2020. doi:10.34970/2020-ds01",
               "Rotemberg V, Kurtansky N, Betz-Stablein B, et al. A patient-centric dataset of images and metadata "
               "for identifying melanomas using clinical context. Sci Data. 2021;8:34. doi:10.1038/s41597-021-00815-z"],
    notes="License as listed on https://challenge.isic-archive.com/data for ISIC 2020.",
)

EXCLUDED = {
    "image_name": "identifier", "patient_id": "identifier", "lesion_id": "identifier",
    "target": "label", "benign_malignant": "label",
    "diagnosis": "label leak ('unknown' = not biopsied, followed up as benign)",
}

SPEC = AuditSpec(
    name="isic2020", title="ISIC 2020",
    num_fields=["age_approx"], cat_fields=["sex", SITE],
    core_fields=["age_approx", "sex", SITE],
    group_level="patient", acquisition_name="resolution group", license=LICENSE,
    classes=list(CLASSES_2020), source_url="https://challenge.isic-archive.com/data/#2020",
    excluded_columns=EXCLUDED, taxonomy="isic2020", taxonomy_dx_columns=["diagnosis"],
)

README_NOTES = (
    "Labels are binary (target: melanoma or not); about 1.8 % of images are melanoma. The 'diagnosis' column is "
    "'unknown' for most benign images (not biopsied, followed up), so it leaks the label and is excluded; the "
    "shared taxonomy files those images as UNSPECIFIED. The metadata has no source column, so the acquisition "
    "variable is the resolution group. Of each pair in the official duplicates list only the first image is kept."
)


def _find(root: str, names: list[str]) -> str | None:
    for name in names:
        hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True), key=len)
        if hits:
            return hits[0]
    return None


def load(root: str = "/kaggle/input", images_root: str | None = None, gt_csv: str | None = None,
         dup_csv: str | None = None, read_image_headers: bool = True,
         min_resolution_group: int = 30) -> AuditDataset:
    """images_root: folder searched for the training JPEGs (default: root). Pass it when
    other inputs also hold ISIC JPEGs (e.g. a resized copy), so the originals are used."""
    gt_csv = gt_csv or _find(root, GT_CANDIDATES)
    assert gt_csv, f"none of {GT_CANDIDATES} found under {root}"
    raw = pd.read_csv(gt_csv, dtype=str, keep_default_na=False)
    unknown = [c for c in raw.columns if c not in EXCLUDED and c not in SPEC.fields]
    assert not unknown, f"columns not covered by SPEC.fields or EXCLUDED: {unknown}"
    assert raw["image_name"].is_unique
    assert set(raw["target"]) <= {"0", "1"}

    dup_csv = dup_csv or _find(root, [DUP_FILE])
    if dup_csv is None:
        dup_csv = os.path.join("/kaggle/temp", DUP_FILE) if os.path.isdir("/kaggle/temp") else DUP_FILE
        if not os.path.exists(dup_csv):
            urllib.request.urlretrieve(DUP_URL, dup_csv)
    dups = pd.read_csv(dup_csv, dtype=str, keep_default_na=False)
    assert {"image_name_1", "image_name_2"} <= set(dups.columns)
    dropped = sorted(set(dups["image_name_2"]) & set(raw["image_name"]))

    df = raw[~raw["image_name"].isin(dropped)].replace(r"^\s*$", np.nan, regex=True)
    tokens = {}
    for f in SPEC.cat_fields:
        df[f], tokens[f] = normalize_missing(df[f])
    df["age_approx"] = pd.to_numeric(df["age_approx"], errors="raise")
    df["label"] = np.where(df["target"] == "1", "malignant", "benign")
    df["image_id"] = df["image_name"]
    df["group_id"] = df["patient_id"]
    assert df["patient_id"].notna().all(), "patient_id has empty values"
    if "lesion_id" not in df.columns:   # the Kaggle competition CSV has no lesion_id
        df["lesion_id"] = np.nan
    df["license"] = LICENSE.name
    df["attribution"] = "ISIC 2020 challenge (International Skin Imaging Collaboration)"

    index, dup_names = index_images(images_root or root, ("jpg", "jpeg"))
    missing = [i for i in df["image_id"] if i not in index]
    assert not missing, f"{len(missing)} images have no file, e.g. {missing[:5]}"
    df["image_path"] = df["image_id"].map(index)
    df = df.reset_index(drop=True)
    if read_image_headers:
        df = df.join(image_props(df["image_path"]))
        df["acquisition"] = resolution_groups(df["width"], df["height"], min_resolution_group)
    else:
        df["acquisition"] = "unknown"

    keep = [c for c in df.columns if c not in ("image_name", "target", "benign_malignant")]
    inputs = {"groundtruth_csv": gt_csv, "groundtruth_md5": md5(gt_csv), "duplicates_csv": dup_csv,
              "duplicates_md5": md5(dup_csv), "n_rows_raw": int(len(raw)), "duplicates_dropped": len(dropped),
              "n_rows": int(len(df)), "n_patients": int(df["patient_id"].nunique()),
              "has_lesion_id": bool(df["lesion_id"].notna().any()), "duplicate_image_names": dup_names,
              "missing_tokens_converted": tokens, "diagnosis_values": raw["diagnosis"].value_counts().to_dict()}
    return AuditDataset(SPEC, df[keep], inputs).validate()
