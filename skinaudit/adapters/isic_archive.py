# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Any ISIC Archive collection -> common schema.

Every collection downloaded with isic-cli has the same layout: one metadata.csv with
the ISIC Archive column names and one JPEG per row. This adapter classifies those
columns once, so a new collection needs only an entry in COLLECTIONS (or none: an
unknown id still works with a generic name).

Labels: the ISIC-Dx path (diagnosis_1 ... diagnosis_5) is mapped with the shared
taxonomy (taxonomy.from_text), so every collection uses the same classes: MEL, NV,
BCC, AK, BKL, DF, VASC, SCC, plus OTHER (a specific diagnosis outside these) and
UNSPECIFIED (only benign / malignant / indeterminate known).

Groups: patient_id when every image has one; otherwise images linked by a shared
patient_id or lesion_id form one group (linked_groups), an image with neither is its own
group, and the summary says so.

Acquisition: the contributing institution (attribution) when there are several with
at least `min_group` images each; else the image type when there are several; else the
resolution group (device proxy).
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..io import image_props, index_images, md5, resolution_groups
from ..schema import CLASSES_8, AuditDataset, AuditSpec, LicenseInfo
from ..stats import normalize_missing
from ..taxonomy import harmonize

ISIC_CLI_VERSION = "12.7.0"

# candidate metadata fields, in this order when present
NUM_CANDIDATES = ["age_approx", "clin_size_long_diam_mm"]
CAT_CANDIDATES = ["sex", "anatom_site_1", "anatom_site_general", "fitzpatrick_skin_type",
                  "personal_hx_mm", "family_hx_mm"]
CORE = ["age_approx", "sex", "anatom_site_1", "anatom_site_general"]

EXCLUDED = {
    "isic_id": "identifier", "patient_id": "identifier", "lesion_id": "identifier", "rcm_case_id": "identifier",
    "attribution": "acquisition (contributing institution)", "copyright_license": "administrative",
    "diagnosis_1": "label", "diagnosis_2": "label", "diagnosis_3": "label",
    "diagnosis_4": "label", "diagnosis_5": "label",
    "diagnosis_confirm_type": "label leak (how the diagnosis was confirmed)",
    "concomitant_biopsy": "label leak (biopsy decision)",
    "melanocytic": "label leak (derived from the diagnosis)",
    "mel_ulcer": "label leak (melanoma histology)", "mel_mitotic_index": "label leak (melanoma histology)",
    "mel_thick_mm": "label leak (melanoma histology)", "mel_class": "label leak (melanoma histology)",
    "mel_type": "label leak (melanoma histology)",
    "anatom_site_2": "sub-site, recorded only for some sites", "anatom_site_3": "sub-site",
    "anatom_site_4": "sub-site", "anatom_site_5": "sub-site",
    "anatom_site_special": "sub-site flag, recorded only when it applies",
    "image_type": "acquisition", "dermoscopic_type": "acquisition", "image_manipulation": "acquisition",
    "tbp_tile_type": "acquisition", "pixels_x": "acquisition", "pixels_y": "acquisition",
    "acquisition_day": "acquisition",
}
DX_COLUMNS = ["diagnosis_1", "diagnosis_2", "diagnosis_3", "diagnosis_4", "diagnosis_5"]
LABEL_ORDER = CLASSES_8 + ["OTHER", "UNSPECIFIED"]
LICENSE_URLS = {"CC-0": "https://creativecommons.org/publicdomain/zero/1.0/",
                "CC-BY": "https://creativecommons.org/licenses/by/4.0/",
                "CC-BY-NC": "https://creativecommons.org/licenses/by-nc/4.0/"}


@dataclass
class Collection:
    name: str
    title: str
    doi: str = ""
    citations: list[str] = field(default_factory=list)
    notes: str = ""


COLLECTIONS = {
    175: Collection("hiba", "HIBA", "10.34970/559884",
                    ["Hospital Italiano de Buenos Aires. HIBA Skin Lesions. ISIC Archive; 2023. doi:10.34970/559884",
                     "Ricci Lara MA, Rodríguez Kowalczuk MV, Eliceche ML, et al. A dataset of skin lesion images "
                     "collected in Argentina for the evaluation of AI tools in this population. Sci Data. "
                     "2023;10:712. doi:10.1038/s41597-023-02630-0"]),
    218: Collection("prove_ai", "PROVe-AI", "10.34970/576276",
                    ["Memorial Sloan Kettering Cancer Center. PROVe-AI. ISIC Archive; 2022. doi:10.34970/576276",
                     "Marchetti MA, Cowen EA, Kurtansky NR, et al. Prospective validation of dermoscopy-based "
                     "open-source artificial intelligence for melanoma diagnosis (PROVE-AI study). npj Digit Med. "
                     "2023;6:127. doi:10.1038/s41746-023-00872-1"]),
    216: Collection("consecutive_biopsies_2020", "Consecutive biopsies 2020", "10.34970/151324",
                    ["Memorial Sloan Kettering Cancer Center. Consecutive biopsies for melanoma across year 2020. "
                     "ISIC Archive; 2022. doi:10.34970/151324"],
                    "Every lesion was biopsied (diagnosis confirmed by histopathology); no patient id."),
    212: Collection("ham10000", "HAM10000", "10.1038/sdata.2018.161",
                    ["Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large collection of multi-source "
                     "dermatoscopic images of common pigmented skin lesions. Sci Data. 2018;5:180161. "
                     "doi:10.1038/sdata.2018.161"],
                    "No patient id (lesion ids only). The 10,015 training images are also part of ISIC 2019 "
                    "(lesion ids HAM_...), so the two audits are not independent samples."),
    294: Collection("melanoma_nevus_histopathology", "Melanoma and nevus (histopathology)", "10.34970/277003",
                    ["Melanoma and Nevus Dermoscopy Images with Confirmed Histopathological Diagnosis. ISIC "
                     "Archive; 2023. doi:10.34970/277003"],
                    "Pooled from several contributors; patient id for part of the images only. Some images "
                    "appear to be the same photos as images in other public collections (e.g. HAM10000, HIBA); "
                    "the overlap is reported in the cross-dataset comparison."),
}


def download(collection_id: int, raw_dir: str | None = None) -> str:
    """Download images + metadata.csv with isic-cli (pip install isic-cli==12.7.0).
    The default folder is under /kaggle/temp, outside the saved notebook output."""
    raw_dir = raw_dir or f"/kaggle/temp/isic_{collection_id}_raw"
    if not os.path.exists(os.path.join(raw_dir, "metadata.csv")):
        subprocess.run(["isic", "image", "download", "--collections", str(collection_id), raw_dir], check=True)
    return raw_dir


def _license(values: pd.Series) -> LicenseInfo:
    present = [v for v in ["CC-BY-NC", "CC-BY", "CC-0"] if v in set(values)]
    strictest = present[0] if present else "unknown"
    name = strictest + (" (per image: " + ", ".join(sorted(set(values))) + ")" if len(set(values)) > 1 else "")
    return LicenseInfo(name=name, url=LICENSE_URLS.get(strictest, ""), allows_redistribution=strictest in LICENSE_URLS,
                       non_commercial=strictest == "CC-BY-NC")


def linked_groups(patient_id: pd.Series, lesion_id: pd.Series, image_id: pd.Series) -> pd.Series:
    """CV groups when patient_id is missing for some images: images that share a patient_id
    or a lesion_id end up in one group (connected components), so a lesion photographed
    once with and once without a patient id cannot fall into two groups (and into both a
    training and a test fold). Group names: the patient ids of the component, else
    'lesion:<id>', else 'image:<id>'."""
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    keys = []
    for p, l, i in zip(patient_id, lesion_id, image_id):
        k = "I:" + str(i)
        find(k)
        if isinstance(p, str) and p:
            union(k, "P:" + p)
        if isinstance(l, str) and l:
            union(k, "L:" + l)
        keys.append(k)
    comps: dict[str, set] = {}
    for k in list(parent):
        comps.setdefault(find(k), set()).add(k)

    def name(root):
        members = comps[root]
        for prefix, label in (("P:", ""), ("L:", "lesion:"), ("I:", "image:")):
            ids = sorted(m[2:] for m in members if m.startswith(prefix))
            if ids:
                return label + "+".join(ids)
    names = {r: name(r) for r in comps}
    return pd.Series([names[find(k)] for k in keys], index=patient_id.index)


def load(collection_id: int, raw_dir: str | None = None, dermoscopic_only: bool = False,
         read_image_headers: bool = True, min_group: int = 30, acquisition: str = "auto",
         allowed_labels: list[str] | None = None) -> AuditDataset:
    """allowed_labels: stop if a mapped label falls outside this list (e.g. CLASSES_8).
    acquisition: 'auto' (see module docstring), 'attribution', 'image_type' or 'resolution'."""
    raw_dir = raw_dir or f"/kaggle/temp/isic_{collection_id}_raw"
    coll = COLLECTIONS.get(collection_id, Collection(f"isic_{collection_id}", f"ISIC Archive collection {collection_id}"))
    meta_csv = os.path.join(raw_dir, "metadata.csv")
    raw = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
    nonempty = {c: int((raw[c].str.strip() != "").sum()) for c in raw.columns}
    num_fields = [c for c in NUM_CANDIDATES if nonempty.get(c, 0) > 0]
    cat_fields = [c for c in CAT_CANDIDATES if nonempty.get(c, 0) > 0]
    fields = num_fields + cat_fields
    unknown = [c for c in raw.columns if c not in EXCLUDED and c not in fields and nonempty[c] > 0]
    assert not unknown, f"columns not covered by the fields or EXCLUDED: {unknown}"
    assert raw["isic_id"].is_unique
    dx = [c for c in DX_COLUMNS if c in raw.columns]
    assert dx and nonempty[dx[0]] == len(raw), "diagnosis_1 must be present for every image"

    df = raw.replace(r"^\s*$", np.nan, regex=True)
    tokens = {}
    for f in cat_fields:
        df[f], tokens[f] = normalize_missing(df[f])
    for f in num_fields:
        df[f] = pd.to_numeric(df[f], errors="raise")

    h = harmonize("isic_archive", df, dx)
    df["label"] = h["l2"]
    df["label_ambiguous"] = h["ambiguous"]
    if allowed_labels is not None:
        bad = df.loc[~df["label"].isin(allowed_labels), dx].drop_duplicates()
        assert bad.empty, f"diagnoses outside {allowed_labels}:\n{bad}"
    classes = [c for c in LABEL_ORDER if (df["label"] == c).any()]

    for c in ["patient_id", "lesion_id"]:
        if c not in df.columns:
            df[c] = np.nan
    if df["patient_id"].notna().all():
        group_level = "patient"
        df["group_id"] = df["patient_id"]
    elif df["patient_id"].notna().any() or df["lesion_id"].notna().any():
        # the groups are the same either way; only the description differs
        if df["patient_id"].notna().any():
            group_level = "patient (lesion or image where missing)"
        else:
            group_level = "lesion" + (" (image where missing)" if df["lesion_id"].isna().any() else "")
        df["group_id"] = linked_groups(df["patient_id"], df["lesion_id"], df["isic_id"])
    else:
        group_level = "image"
        df["group_id"] = df["isic_id"]

    df["image_id"] = df["isic_id"]
    df["license"] = df["copyright_license"]
    index, dup = index_images(raw_dir)
    missing = [i for i in df["image_id"] if i not in index]
    assert not missing, f"{len(missing)} rows have no image file, e.g. {missing[:5]}"
    df["image_path"] = df["image_id"].map(index)
    size_mismatch = None
    if read_image_headers:
        df = df.join(image_props(df["image_path"]))
        if {"pixels_x", "pixels_y"} <= set(df.columns) and df["pixels_x"].notna().all():
            px = df[["pixels_x", "pixels_y"]].apply(pd.to_numeric)
            size_mismatch = int(((px["pixels_x"] != df["width"]) | (px["pixels_y"] != df["height"])).sum())

    title, name = coll.title, coll.name
    if dermoscopic_only:
        df = df[df["image_type"] == "dermoscopic"].copy()
        title, name = f"{coll.title} (dermoscopic only)", f"{coll.name}_dermoscopic"

    def big_groups(s):
        vc = s.value_counts()
        return int((vc >= min_group).sum())
    mode = acquisition
    if mode == "auto":
        if "attribution" in df.columns and big_groups(df["attribution"].fillna("unknown")) >= 2:
            mode = "attribution"
        elif "image_type" in df.columns and df["image_type"].nunique() >= 2:
            mode = "image_type"
        else:
            mode = "resolution"
    if mode == "attribution":
        df["acquisition"], acq_name = df["attribution"].fillna("unknown"), "institution"
    elif mode == "image_type":
        df["acquisition"], acq_name = df["image_type"], "image type"
    else:
        assert read_image_headers, "acquisition by resolution needs read_image_headers=True"
        df["acquisition"], acq_name = resolution_groups(df["width"], df["height"], min_group), "resolution group"

    lic = _license(raw["copyright_license"])
    lic.citations, lic.notes = coll.citations, coll.notes
    spec = AuditSpec(
        name=name, title=title, num_fields=num_fields, cat_fields=cat_fields,
        core_fields=[c for c in CORE if c in fields], group_level=group_level, acquisition_name=acq_name,
        license=lic, classes=classes, source_url=f"https://api.isic-archive.com/collections/{collection_id}/",
        excluded_columns={c: r for c, r in EXCLUDED.items() if c in raw.columns},
        taxonomy="isic_archive", taxonomy_dx_columns=dx,
    )
    leak = [c for c in EXCLUDED if c in df.columns and c not in ("isic_id", "patient_id", "lesion_id", "attribution")]
    keep = ["image_id", "group_id", "patient_id", "lesion_id", "label", "label_ambiguous", "acquisition", *fields,
            "image_path", "license", "attribution", *leak] + \
           (["width", "height", "file_bytes"] if read_image_headers else [])
    keep = list(dict.fromkeys(c for c in keep if c in df.columns))
    inputs = {"collection_id": collection_id, "doi": coll.doi, "metadata_csv": meta_csv,
              "metadata_md5": md5(meta_csv), "n_rows": int(len(raw)), "fields_found": fields,
              "group_level": group_level, "acquisition_mode": mode,
              "patient_id_share": round(float(raw["patient_id"].ne("").mean()), 4) if "patient_id" in raw else 0.0,
              "duplicate_image_names": dup, "missing_tokens_converted": tokens,
              "image_vs_metadata_size_mismatch": size_mismatch,
              "label_counts": df["label"].value_counts().to_dict(),
              "license_values": raw["copyright_license"].value_counts().to_dict(),
              "attribution_values": raw["attribution"].value_counts().to_dict() if "attribution" in raw else {}}
    return AuditDataset(spec, df[keep].reset_index(drop=True), inputs).validate()
