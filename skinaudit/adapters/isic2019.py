# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""ISIC 2019 training set -> common schema.

The source (BCN, HAM, MSK) is inferred from the lesion_id prefix;
images without lesion_id are their own group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import find_one, image_props, index_images, md5
from ..schema import CLASSES_8, AuditDataset, AuditSpec, LicenseInfo
from ..stats import normalize_missing

LICENSE = LicenseInfo(
    name="CC-BY-NC 4.0",
    url="https://creativecommons.org/licenses/by-nc/4.0/",
    allows_redistribution=True,
    non_commercial=True,
    citations=[
        "BCN_20000 Dataset: (c) Department of Dermatology, Hospital Clínic de Barcelona.",
        "HAM10000 Dataset: (c) by ViDIR Group, Department of Dermatology, Medical University of Vienna; "
        "https://doi.org/10.1038/sdata.2018.161",
        "MSK Dataset: (c) Anonymous; https://arxiv.org/abs/1710.05006; https://arxiv.org/abs/1902.03368",
        "Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic "
        "images of common pigmented skin lesions. Sci Data. 2018;5:180161. doi:10.1038/sdata.2018.161",
        "Codella NCF, Gutman D, Celebi ME, et al. Skin lesion analysis toward melanoma detection: a challenge at the "
        "2017 International Symposium on Biomedical Imaging (ISBI), hosted by the International Skin Imaging "
        "Collaboration (ISIC). arXiv:1710.05006. 2017.",
        "Hernández-Pérez C, Combalia M, Podlipnik S, et al. BCN20000: dermoscopic lesions in the wild. Sci Data. "
        "2024;11:641. doi:10.1038/s41597-024-03387-w",
    ],
    notes="License and citations as listed on https://challenge.isic-archive.com/data for ISIC 2019.",
)

ATTRIBUTION = {
    "BCN": "Department of Dermatology, Hospital Clinic de Barcelona (BCN_20000)",
    "HAM": "ViDIR Group, Medical University of Vienna (HAM10000)",
}
MSK_ATTRIBUTION = "ISIC 2019 MSK dataset (Memorial Sloan Kettering Cancer Center)"

SPEC = AuditSpec(
    name="isic2019", title="ISIC 2019",
    num_fields=["age_approx"], cat_fields=["anatom_site_general", "sex"],
    core_fields=["age_approx", "anatom_site_general", "sex"],
    group_level="lesion", acquisition_name="source", license=LICENSE,
    source_url="https://challenge.isic-archive.com/data/#2019",
    taxonomy="isic2019",
)


def _source(lesion_id, image):
    if isinstance(lesion_id, str) and "_" in lesion_id:
        return lesion_id.split("_")[0]
    if image.endswith("_downsampled"):
        return "DOWNSAMPLED"
    return "NO_LESION_ID"


def load(root: str = "/kaggle/input", age_nonpositive_as_missing: bool = False,
         read_image_headers: bool = True) -> AuditDataset:
    meta_csv, n_meta = find_one("ISIC_2019_Training_Metadata.csv", root)
    gt_csv, n_gt = find_one("ISIC_2019_Training_GroundTruth.csv", root)
    meta, gt = pd.read_csv(meta_csv), pd.read_csv(gt_csv)
    assert meta["image"].is_unique and gt["image"].is_unique and set(meta["image"]) == set(gt["image"])
    df = meta.merge(gt, on="image", validate="one_to_one")
    onehot = df[CLASSES_8].to_numpy(float)
    assert np.isin(onehot, [0.0, 1.0]).all() and (onehot.sum(1) == 1).all()
    if "UNK" in df.columns:
        assert (df["UNK"] == 0).all()
    df["label"] = np.array(CLASSES_8)[onehot.argmax(1)]

    tokens = {}
    for f in SPEC.cat_fields:
        df[f], tokens[f] = normalize_missing(df[f])
    n_age_nonpos = int((df["age_approx"] <= 0).sum())
    if age_nonpositive_as_missing:
        df.loc[df["age_approx"] <= 0, "age_approx"] = np.nan

    df["image_id"] = df["image"]
    df["lesion_id"] = df["lesion_id"].astype("string")
    df["acquisition"] = [_source(l, i) for l, i in zip(df["lesion_id"].astype(object), df["image"])]
    df["group_id"] = df["lesion_id"].astype(object).where(df["lesion_id"].notna(), df["image"])
    df["license"] = LICENSE.name
    df["attribution"] = df["acquisition"].map(ATTRIBUTION).fillna(MSK_ATTRIBUTION)

    index, dup = index_images(root, ("jpg",))
    missing = [n for n in df["image_id"] if n not in index]
    assert not missing, f"{len(missing)} images have no file, e.g. {missing[:5]}"
    df["image_path"] = df["image_id"].map(index)
    if read_image_headers:
        df = df.join(image_props(df["image_path"]))

    keep = ["image_id", "group_id", "lesion_id", "label", "acquisition", *SPEC.fields,
            "image_path", "license", "attribution"] + (["width", "height", "file_bytes"] if read_image_headers else [])
    inputs = {"metadata_csv": meta_csv, "metadata_md5": md5(meta_csv), "metadata_copies": n_meta,
              "groundtruth_csv": gt_csv, "groundtruth_md5": md5(gt_csv), "groundtruth_copies": n_gt,
              "duplicate_image_names": dup, "missing_tokens_converted": tokens,
              "age_nonpositive_rows": n_age_nonpos, "age_nonpositive_as_missing": age_nonpositive_as_missing}
    return AuditDataset(SPEC, df[keep].reset_index(drop=True), inputs).validate()
