# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""SLICE-3D (ISIC 2024 challenge training set, 3D total-body-photography tiles) -> common schema.

Files come from the public ISIC challenge page (no login); download() fetches them
(Kaggle: turn Internet on). Labels are binary (GroundTruth column 'malignant'); the
ISIC-Dx diagnosis columns are kept only for the shared taxonomy and the leak check.

The shortcut tests run on a subsample (see cv_subsample): every malignant tile and up to
`benign_cap` random benign tiles per patient. Benign tiles are drawn without looking at
whether they carry a specific diagnosis: a specific diagnosis likely means the lesion
was biopsied or reviewed because it looked suspicious, so keeping all of them could
over-represent suspicious benign lesions and bias the shortcut tests. Descriptive
tables use every tile.

Column names are discovered at load time and every column must be classified below;
an unknown column stops the run, so a changed file layout is caught on the first run.
"""
from __future__ import annotations

import glob
import os
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from ..io import image_props, index_images, md5
from ..schema import AuditDataset, AuditSpec, LicenseInfo
from ..stats import normalize_missing

BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2024/"
FILES = ["ISIC_2024_Training_GroundTruth.csv", "ISIC_2024_Training_Supplement.csv", "ISIC_2024_Training_Input.zip"]
CLASSES_SLICE = ["benign", "malignant"]

LICENSE = LicenseInfo(
    name="CC-BY-NC 4.0 (per image: CC-0, CC-BY or CC-BY-NC)",
    url="https://creativecommons.org/licenses/by-nc/4.0/",
    allows_redistribution=True,
    non_commercial=True,
    citations=["International Skin Imaging Collaboration. SLICE-3D 2024 Challenge Dataset. International Skin "
               "Imaging Collaboration; 2024. doi:10.34970/2024-slice-3d",
               "Kurtansky NR, D'Alessandro BM, Gillis MC, et al. The SLICE-3D dataset: 400,000 skin lesion image "
               "crops extracted from 3D TBP for skin cancer detection. Sci Data. 2024;11:884. "
               "doi:10.1038/s41597-024-03743-w"],
    notes="Per-image license and attribution are read from the metadata; the most restrictive (CC-BY-NC) "
          "covers the dataset.",
)

# exact names, then name patterns; every metadata column must match one of these
EXCLUDED = {
    "isic_id": "identifier", "patient_id": "identifier",
    "attribution": "acquisition (institution; used as the acquisition variable)",
    "copyright_license": "administrative",
    "image_type": "acquisition", "tbp_tile_type": "acquisition (XP vs white light)",
    "image_manipulation": "acquisition",
    "lesion_id": "label leak (present only for manually tagged lesions)",
    "target": "label", "malignant": "label",
    "concomitant_biopsy": "label leak (biopsy decision)",
    "diagnosis_confirm_type": "label leak (how the diagnosis was confirmed)",
    "tbp_lv_dnn_lesion_confidence": "label leak (training-only model score)",
    "anatom_site_2": "sub-site", "anatom_site_3": "sub-site",
}
EXCLUDED_PATTERNS = {
    r"^(iddx|diagnosis)_(full|[1-5])$": "label (ISIC-Dx hierarchy)",
    r"^mel_": "label leak (melanoma histology)",
    r"^tbp_lv_": "TBP device measurement of the lesion (image-derived; reference feature set)",
}
SITE_CANDIDATES = ["anatom_site_general", "anatom_site_1"]

README_NOTES = (
    "Labels are binary (malignant / benign); about 0.1 % of tiles are malignant. Missingness, associations "
    "and image statistics use every tile; the shortcut tests use a subsample: every malignant tile and a capped "
    "random sample of benign tiles per patient, drawn without regard to whether a tile has a specific diagnosis "
    "(a specific diagnosis likely marks lesions biopsied because they looked suspicious; rule and counts in "
    "summary.json: inputs.cv_subsample and cv_subsample). Most benign tiles have no specific diagnosis "
    "('Benign' only), so the shared taxonomy files them as UNSPECIFIED. tbp_lv_* columns are measurements "
    "made by the 3D imaging system from the image; they are tested only as a reference feature set."
)


def download(raw_dir: str = "/kaggle/temp/slice3d_raw") -> str:
    """Fetch the training files and unzip the tiles (skips files already present).

    The default folder is not part of the notebook output: 401k image files there
    would be saved with the notebook.
    """
    os.makedirs(raw_dir, exist_ok=True)
    for name in FILES:
        dst = os.path.join(raw_dir, name)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(BASE_URL + name, dst)
    img_dir = os.path.join(raw_dir, "images")
    if not os.path.isdir(img_dir):
        with zipfile.ZipFile(os.path.join(raw_dir, "ISIC_2024_Training_Input.zip")) as z:
            z.extractall(img_dir)
    return raw_dir


def _classify(col: str) -> str | None:
    if col in EXCLUDED:
        return EXCLUDED[col]
    for pat, reason in EXCLUDED_PATTERNS.items():
        if re.search(pat, col):
            return reason
    return None


def subsample_rule(patient_id, benign_cap: int = 20, seed: int = 2026):
    """The subsample as a function of the labels: every malignant tile and the benign_cap
    benign tiles of each patient with the smallest random key. The keys are drawn once per
    row, independently of the labels, so the rule can be re-applied to permuted labels
    (the permutation nulls do this). Deterministic for a given seed."""
    key = pd.Series(np.random.default_rng(seed).random(len(patient_id)))
    pid = pd.Series(np.asarray(patient_id))

    def rule(labels) -> np.ndarray:
        mal = np.asarray(labels) == "malignant"
        rank = key.where(~mal).groupby(pid).rank(method="first").to_numpy()
        return mal | (rank <= benign_cap)
    return rule


def cv_subsample(df: pd.DataFrame, benign_cap: int = 20, seed: int = 2026) -> np.ndarray:
    """Every malignant tile and up to benign_cap random benign tiles per patient.
    Benign tiles are drawn regardless of has_specific_dx (see module docstring)."""
    return subsample_rule(df["patient_id"], benign_cap, seed)(df["label"])


def load(raw_dir: str = "/kaggle/temp/slice3d_raw", read_image_headers: bool = True,
         benign_cap: int = 20, seed: int = 2026) -> AuditDataset:
    gt = pd.read_csv(os.path.join(raw_dir, FILES[0]), dtype=str, keep_default_na=False)
    meta = pd.read_csv(os.path.join(raw_dir, FILES[1]), dtype=str, keep_default_na=False)
    # a metadata CSV shipped inside the image zip, if any, is merged too
    inner = sorted(glob.glob(os.path.join(raw_dir, "images", "**", "*.csv"), recursive=True))
    for p in inner:
        extra = pd.read_csv(p, dtype=str, keep_default_na=False)
        if "isic_id" in extra.columns:
            new = [c for c in extra.columns if c not in meta.columns]
            meta = meta.merge(extra[["isic_id", *new]], on="isic_id", how="left", validate="one_to_one")
    assert meta["isic_id"].is_unique and gt["isic_id"].is_unique and set(meta["isic_id"]) == set(gt["isic_id"])
    assert set(gt["malignant"].astype(float)) <= {0.0, 1.0}

    site = next((c for c in SITE_CANDIDATES if c in meta.columns), None)
    assert site is not None, f"no site column among {SITE_CANDIDATES}: {meta.columns.tolist()}"
    num_fields = ["age_approx"] + (["clin_size_long_diam_mm"] if "clin_size_long_diam_mm" in meta.columns else [])
    cat_fields = ["sex", site]
    fields = num_fields + cat_fields
    other_sites = [c for c in SITE_CANDIDATES if c in meta.columns and c != site]
    unknown = [c for c in meta.columns if c not in fields and c not in other_sites and _classify(c) is None
               and (meta[c].str.strip() != "").any()]
    assert not unknown, f"columns not classified (add them to EXCLUDED or the fields): {unknown}"
    dx_cols = sorted([c for c in meta.columns if re.fullmatch(r"(iddx|diagnosis)_[1-5]", c)],
                     key=lambda c: int(c[-1]))
    assert dx_cols, "no ISIC-Dx diagnosis columns found (needed for the shared taxonomy)"

    df = meta.merge(gt[["isic_id", "malignant"]].rename(columns={"malignant": "_gt"}), on="isic_id",
                    validate="one_to_one").replace(r"^\s*$", np.nan, regex=True)
    tokens = {}
    for f in cat_fields:
        df[f], tokens[f] = normalize_missing(df[f])
    for f in num_fields:
        df[f] = pd.to_numeric(df[f], errors="raise")
    df["label"] = np.where(df["_gt"].astype(float) == 1.0, "malignant", "benign")
    df["has_specific_dx"] = df[dx_cols[1:]].notna().any(axis=1) if len(dx_cols) > 1 else False

    df["image_id"] = df["isic_id"]
    df["group_id"] = df["patient_id"]
    assert df["patient_id"].notna().all(), "patient_id has empty values"
    df["acquisition"] = df["attribution"].fillna("unknown") if "attribution" in df.columns else "unknown"
    df["license"] = df["copyright_license"] if "copyright_license" in df.columns else LICENSE.name
    index, dup = index_images(os.path.join(raw_dir, "images"))
    missing = [i for i in df["image_id"] if i not in index]
    assert not missing, f"{len(missing)} rows have no image file, e.g. {missing[:5]}"
    df["image_path"] = df["image_id"].map(index)
    if read_image_headers:
        df = df.join(image_props(df["image_path"]))

    spec = AuditSpec(
        name="slice3d", title="SLICE-3D (ISIC 2024)",
        num_fields=num_fields, cat_fields=cat_fields, core_fields=["age_approx", "sex", site],
        group_level="patient", acquisition_name="institution", license=LICENSE,
        classes=list(CLASSES_SLICE), source_url="https://challenge.isic-archive.com/data/#2024",
        excluded_columns={c: _classify(c) for c in meta.columns if _classify(c)},
        taxonomy="slice3d", taxonomy_dx_columns=dx_cols,
    )
    tbp = [c for c in meta.columns if c.startswith("tbp_lv_") and c != "tbp_lv_dnn_lesion_confidence"]
    # lesion_id exists only for manually tagged lesions: its presence is the leak, so keep a flag
    df["has_lesion_tag"] = df["lesion_id"].notna() if "lesion_id" in df.columns else False
    leak_cols = [c for c in ["has_lesion_tag", "concomitant_biopsy", "diagnosis_confirm_type", "tbp_tile_type",
                             "image_type", "copyright_license"] if c in df.columns]
    # every tile is its own lesion; a manually tagged lesion keeps its lesion_id. This is the
    # permutation unit (one label per lesion), as in the other datasets
    tag = df["lesion_id"] if "lesion_id" in df.columns else pd.Series(np.nan, index=df.index)
    df["lesion_id"] = tag.astype(object).where(tag.notna(), "tile:" + df["image_id"].astype(str))
    keep = ["image_id", "group_id", "patient_id", "lesion_id", "label", "acquisition", *fields, "image_path", "license",
            "attribution", "has_specific_dx", *dx_cols, *leak_cols, *tbp] + \
           (["width", "height", "file_bytes"] if read_image_headers else [])
    keep = list(dict.fromkeys(c for c in keep if c in df.columns))
    out = df[keep].reset_index(drop=True)
    # tbp_lv_* numeric columns become floats; text ones (e.g. location) stay as they are
    for c in tbp:
        conv = pd.to_numeric(out[c], errors="coerce")
        if conv.notna().sum() == out[c].notna().sum():
            out[c] = conv
    rule = subsample_rule(out["patient_id"], benign_cap, seed)
    mask = rule(out["label"])
    spec.excluded_columns["has_lesion_tag"] = "label leak (lesion_id present only for manually tagged lesions)"
    inputs = {"files": {n: md5(os.path.join(raw_dir, n)) for n in FILES[:2]}, "inner_csvs": inner,
              "n_rows": int(len(out)), "n_patients": int(out["patient_id"].nunique()),
              "site_column": site, "dx_columns": dx_cols, "tbp_columns": tbp,
              "duplicate_image_names": dup, "missing_tokens_converted": tokens,
              "license_values": meta["copyright_license"].value_counts().to_dict()
              if "copyright_license" in meta.columns else {},
              "cv_subsample": {"rule": f"all malignant + up to {benign_cap} random benign tiles per patient "
                                        f"(drawn regardless of a specific diagnosis)", "seed": seed,
                               "n_rows": int(mask.sum())}}
    return AuditDataset(spec, out, inputs, cv_rows_mask=mask, cv_rule=rule).validate()


def tbp_feature_set(ds: AuditDataset) -> dict:
    """Optional reference shortcut test: the numeric TBP measurements of each lesion.

    They are produced by the imaging system from the image (colour, size, shape,
    position on the body, a nevus-confidence score), so a high score is not a
    metadata shortcut; it shows how much of the label the device measurements carry.
    """
    cols = [c for c in ds.inputs["tbp_columns"] if pd.api.types.is_numeric_dtype(ds.df[c])]
    X = ds.df[cols].to_numpy(float)
    return {"J_tbp_measurements_reference": (X, [False] * len(cols), np.ones(len(ds.df), bool))}
