# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""PAD-UFES-20 (smartphone clinical images, Mendeley Data zr7vgbcyr2) -> common schema.

Two kinds of "unknown" exist in this dataset (Pacheco et al. 2020, section 1.2.2):
an empty cell means the question was not asked or not recorded; "UNK" means the
patient was asked and does not know. Only empty cells count as missing here; "UNK"
stays a category of its own and its share is reported separately in inputs.

lesion_id is not unique across patients, so the lesion key is patient_id + lesion_id.
All images come from smartphones of different makes, so the acquisition column is the
resolution group (device proxy).
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from ..io import find_one, image_props, index_images, md5, resolution_groups
from ..schema import AuditDataset, AuditSpec, LicenseInfo

CLASSES_PAD = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]

LICENSE = LicenseInfo(
    name="CC BY 4.0",
    url="https://creativecommons.org/licenses/by/4.0/",
    allows_redistribution=True,
    non_commercial=False,
    citations=["Pacheco AGC, Lima GR, Salomão AS, et al. PAD-UFES-20: a skin lesion dataset composed of patient "
               "data and clinical images collected from smartphones. Data Brief. 2020;32:106221. "
               "doi:10.1016/j.dib.2020.106221",
               "Pacheco AGC, Lima GR, Salomão AS, et al. PAD-UFES-20: a skin lesion dataset composed of patient "
               "data and clinical images collected from smartphones [dataset]. Mendeley Data; 2020. "
               "doi:10.17632/zr7vgbcyr2.1"],
    notes="License CC BY 4.0 as shown on https://data.mendeley.com/datasets/zr7vgbcyr2/1 (checked by the author).",
)

NUM_FIELDS = ["age", "fitspatrick", "diameter_1", "diameter_2"]
CAT_FIELDS = ["gender", "region", "smoke", "drink", "background_father", "background_mother", "pesticide",
              "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system",
              "itch", "grew", "hurt", "changed", "bleed", "elevation"]

EXCLUDED = {
    "patient_id": "identifier", "lesion_id": "identifier", "img_id": "identifier",
    "diagnostic": "label",
    "biopsed": "label leak (every cancer is biopsied; part of the benign lesions is not)",
}

SPEC = AuditSpec(
    name="pad_ufes_20", title="PAD-UFES-20",
    num_fields=NUM_FIELDS, cat_fields=CAT_FIELDS,
    core_fields=["age", "gender", "region"],
    group_level="patient", acquisition_name="resolution group", license=LICENSE,
    classes=list(CLASSES_PAD),
    source_url="https://data.mendeley.com/datasets/zr7vgbcyr2/1",
    excluded_columns=EXCLUDED,
    taxonomy="pad_ufes_20",
)

README_NOTES = (
    "Empty cells are counted as missing; \"UNK\" (the patient was asked and does not know) is kept as its own "
    "category, as the dataset paper defines it. The UNK share per field is in summary.json under "
    "inputs.unk_counts. lesion_id repeats across patients, so lesions are keyed by patient_id + lesion_id. "
    "All images are smartphone photographs; the acquisition variable is the image resolution group, a proxy "
    "for the phone model."
)


def load(root: str = "/kaggle/input", read_image_headers: bool = True,
         min_resolution_group: int = 30, meta_csv: str | None = None) -> AuditDataset:
    """meta_csv: explicit path when other inputs also contain a file named metadata.csv."""
    if meta_csv is None:
        meta_csv, n_copies = find_one("metadata.csv", root)
    else:
        n_copies = 1
    raw = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
    nonempty = {c: int((raw[c].str.strip() != "").sum()) for c in raw.columns}
    unknown = [c for c in raw.columns if c not in EXCLUDED and c not in SPEC.fields and nonempty[c] > 0]
    assert not unknown, f"columns not covered by SPEC.fields or EXCLUDED: {unknown}"
    assert raw["img_id"].is_unique
    assert set(raw["diagnostic"]) <= set(CLASSES_PAD), f"unexpected labels: {set(raw['diagnostic']) - set(CLASSES_PAD)}"

    # only empty cells are missing; UNK is a value (see module docstring)
    df = raw.apply(lambda s: s.str.strip()).replace("", np.nan)
    unk_counts = {f: int((df[f] == "UNK").sum()) for f in SPEC.fields if (df[f] == "UNK").any()}
    for f in NUM_FIELDS:
        df[f] = pd.to_numeric(df[f], errors="raise")

    df["label"] = df["diagnostic"]
    df["image_id"] = df["img_id"].str.replace(r"\.png$", "", regex=True)
    df["lesion_id"] = df["patient_id"] + "_" + df["lesion_id"]
    df["group_id"] = df["patient_id"]
    conflicts = int((df.groupby("lesion_id")["label"].nunique() > 1).sum())
    assert conflicts == 0, f"{conflicts} lesions (patient_id + lesion_id) carry several labels"
    raw_lesion_shared = int((raw.groupby("lesion_id")["patient_id"].nunique() > 1).sum())

    index, dup = index_images(root, ("png",))
    missing = [i for i in df["image_id"] if i not in index]
    assert not missing, f"{len(missing)} images have no file, e.g. {missing[:5]}"
    df["image_path"] = df["image_id"].map(index)
    df["license"] = LICENSE.name
    df["attribution"] = "PAD-UFES-20, Federal University of Espirito Santo (UFES), Brazil"
    if read_image_headers:
        df = df.join(image_props(df["image_path"]))
        df["acquisition"] = resolution_groups(df["width"], df["height"], min_resolution_group)
    else:
        df["acquisition"] = "smartphone"

    keep = ["image_id", "group_id", "patient_id", "lesion_id", "label", "acquisition", *SPEC.fields,
            "image_path", "license", "attribution", "biopsed"] + \
           (["width", "height", "file_bytes"] if read_image_headers else [])
    inputs = {"metadata_csv": meta_csv, "metadata_md5": md5(meta_csv), "metadata_copies": n_copies,
              "n_rows": int(len(raw)), "n_patients": int(raw["patient_id"].nunique()),
              "raw_lesion_ids_shared_by_several_patients": raw_lesion_shared,
              "n_lesions_patient_keyed": int(df["lesion_id"].nunique()),
              "duplicate_image_names": dup, "unk_counts": unk_counts,
              "empty_counts": {f: int(df[f].isna().sum()) for f in SPEC.fields}}
    return AuditDataset(SPEC, df[keep].reset_index(drop=True), inputs).validate()
