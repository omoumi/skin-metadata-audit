# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""MILK10k (ISIC 2025 challenge training set) -> common schema.

One clinical close-up and one dermoscopic image per lesion; no patient id, so the
CV group is the lesion. Files come from the ISIC challenge page; download() fetches
them (Kaggle: turn Internet on).
"""
from __future__ import annotations

import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from ..io import image_props, index_images, md5, resolution_groups
from ..schema import AuditDataset, AuditSpec, LicenseInfo
from ..stats import normalize_missing

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/milk10k/"
FILES = ["MILK10k_Training_Metadata.csv", "MILK10k_Training_GroundTruth.csv",
         "MILK10k_Training_Supplement.csv", "MILK10k_Training_Input.zip"]

CLASSES_MILK = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCCKA", "INF", "BEN_OTH", "MAL_OTH"]

LICENSE = LicenseInfo(
    name="CC-BY-NC 4.0",
    url="https://creativecommons.org/licenses/by-nc/4.0/",
    allows_redistribution=True,
    non_commercial=True,
    citations=["MILK study team. MILK10k. ISIC Archive; 2025. doi:10.34970/648456",
               "Tschandl P, Akay BN, Rosendahl C, et al. MILK10k: a hierarchical multimodal imaging-learning toolkit "
               "for diagnosing pigmented and nonpigmented skin cancer and its simulators. J Invest Dermatol. "
               "2026;146(2):357-364.e7. doi:10.1016/j.jid.2025.06.1594"],
    notes="Per-image license and attribution are read from the metadata.",
)

EXCLUDED = {
    "lesion_id": "identifier", "isic_id": "identifier",
    "attribution": "administrative", "copyright_license": "administrative",
    "image_type": "acquisition (every lesion has both types)",
    "image_manipulation": "acquisition (instrument only / altered)",
    "diagnosis_full": "label (supplement file)",
    "diagnosis_confirm_type": "label leak (how the diagnosis was confirmed; supplement file)",
    "invasion_thickness_interval": "label leak (melanoma thickness; supplement file)",
    "MONET_*": "image-derived concept scores (computed from the image, not patient metadata)",
}

SPEC = AuditSpec(
    name="milk10k", title="MILK10k",
    num_fields=["age_approx", "skin_tone_class"], cat_fields=["sex", "site"],
    core_fields=["age_approx", "sex", "site"],
    group_level="lesion", acquisition_name="image type", license=LICENSE,
    classes=list(CLASSES_MILK),
    source_url="https://challenge.isic-archive.com/data/#2025",
    excluded_columns=EXCLUDED,
    taxonomy="milk10k",
)


README_NOTES = (
    "MILK10k has no patient id, so cross-validation groups are lesions. If one patient contributed several "
    "lesions, images of that patient can sit in both training and test folds; shortcut tests that rely on "
    "patient-level information (age, sex, skin tone, camera) may therefore look somewhat stronger than they "
    "would with patient-grouped folds. Each lesion has exactly one clinical close-up and one dermoscopic "
    "image, so the acquisition variable (image type) is balanced across classes by design."
)


def download(raw_dir: str = "/kaggle/temp/milk10k_raw") -> str:
    """Fetch the four training files and unzip the images (skips files already present).

    The default folder is not part of the notebook output, so the images are not saved with it.
    """
    os.makedirs(raw_dir, exist_ok=True)
    for name in FILES:
        dst = os.path.join(raw_dir, name)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(BASE_URL + name, dst)
    img_dir = os.path.join(raw_dir, "images")
    if not os.path.isdir(img_dir):
        with zipfile.ZipFile(os.path.join(raw_dir, "MILK10k_Training_Input.zip")) as z:
            z.extractall(img_dir)
    return raw_dir


def load(raw_dir: str = "/kaggle/temp/milk10k_raw", dermoscopic_only: bool = False,
         read_image_headers: bool = True, min_resolution_group: int = 30) -> AuditDataset:
    def read(name):
        return pd.read_csv(os.path.join(raw_dir, name), dtype=str, keep_default_na=False)

    meta, gt, supp = read(FILES[0]), read(FILES[1]), read(FILES[2])
    monet = [c for c in meta.columns if c.startswith("MONET_")]
    known = set(EXCLUDED) | set(SPEC.fields) | set(monet)
    unknown = [c for c in meta.columns if c not in known and (meta[c].str.strip() != "").any()]
    assert not unknown, f"metadata columns not covered by SPEC.fields or EXCLUDED: {unknown}"
    assert meta["isic_id"].is_unique and gt["lesion_id"].is_unique and supp["isic_id"].is_unique
    assert set(gt.columns) == {"lesion_id", *CLASSES_MILK}, f"unexpected ground-truth columns: {gt.columns.tolist()}"
    assert set(meta["lesion_id"]) == set(gt["lesion_id"]) and set(meta["isic_id"]) == set(supp["isic_id"])

    onehot = gt[CLASSES_MILK].astype(float).to_numpy()
    assert np.isin(onehot, [0.0, 1.0]).all() and (onehot.sum(1) == 1).all(), "ground truth is not one-hot"
    gt["label"] = np.array(CLASSES_MILK)[onehot.argmax(1)]

    df = meta.merge(supp, on="isic_id", validate="one_to_one") \
             .merge(gt[["lesion_id", "label"]], on="lesion_id", validate="many_to_one")
    df = df.replace(r"^\s*$", np.nan, regex=True)
    tokens = {}
    for f in SPEC.cat_fields:
        df[f], tokens[f] = normalize_missing(df[f])
    for f in SPEC.num_fields + monet:
        df[f] = pd.to_numeric(df[f], errors="raise")
    pairs = df.groupby("lesion_id")["image_type"].agg(lambda s: " + ".join(sorted(s))).value_counts().to_dict()

    df["image_id"] = df["isic_id"]
    df["group_id"] = df["lesion_id"]
    df["acquisition"] = df["image_type"]
    df["license"] = df["copyright_license"]
    index, dup = index_images(os.path.join(raw_dir, "images"))
    missing = [i for i in df["image_id"] if i not in index]
    assert not missing, f"{len(missing)} rows have no image file, e.g. {missing[:5]}"
    df["image_path"] = df["image_id"].map(index)
    if read_image_headers:
        df = df.join(image_props(df["image_path"]))

    spec = SPEC
    if dermoscopic_only:
        # one image type left, so the acquisition column becomes the resolution group (device proxy)
        assert read_image_headers, "dermoscopic_only needs read_image_headers=True (acquisition = resolution group)"
        df = df[df["image_type"] == "dermoscopic"].copy()
        df["acquisition"] = resolution_groups(df["width"], df["height"], min_resolution_group)
        spec = AuditSpec(**{**vars(SPEC), "name": "milk10k_dermoscopic", "title": "MILK10k (dermoscopic only)",
                            "acquisition_name": "resolution group"})

    keep = ["image_id", "group_id", "lesion_id", "label", "acquisition", *SPEC.fields, "image_path",
            "license", "attribution", "image_type", "image_manipulation", "diagnosis_confirm_type",
            "invasion_thickness_interval", *monet] + (["width", "height", "file_bytes"] if read_image_headers else [])
    inputs = {"files": {n: md5(os.path.join(raw_dir, n)) for n in FILES[:3]},
              "n_images": int(len(meta)), "n_lesions": int(len(gt)), "image_types_per_lesion": pairs,
              "duplicate_image_names": dup, "missing_tokens_converted": tokens, "monet_columns": monet,
              "license_values": meta["copyright_license"].value_counts().to_dict(),
              "attribution_values": meta["attribution"].value_counts().to_dict()}
    return AuditDataset(spec, df[keep].reset_index(drop=True), inputs).validate()


def monet_feature_set(ds: AuditDataset) -> dict:
    """Optional extra shortcut test: the MONET concept scores of each image.

    They are computed from the image, so a high score here is not a metadata
    shortcut; it shows how much of the label simple image concepts already carry.
    """
    cols = [c for c in ds.df.columns if c.startswith("MONET_")]
    X = ds.df[cols].to_numpy(float)
    return {"H_monet_concepts_reference": (X, [False] * len(cols), np.ones(len(ds.df), bool))}
