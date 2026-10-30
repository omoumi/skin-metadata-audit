# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Shared two-level taxonomy, used only to compare datasets.

Each dataset audit keeps its own labels. For the comparison every native label is
mapped once, here, to

  level 1: malignant / benign / indeterminate
  level 2: MEL, NV, BCC, SCC, AK, BKL, DF, VASC, OTHER, UNSPECIFIED

UNSPECIFIED is a lesion with no specific diagnosis (e.g. "unknown" in ISIC 2020, a
bare "Benign" in SLICE-3D); OTHER is a specific diagnosis outside the shared classes.
A mapping is flagged ambiguous when the native class mixes shared classes (e.g. AKIEC
mixes actinic keratosis and intraepidermal carcinoma); the share of ambiguous images
is reported next to every comparison.

Conventions (applied the same way in every dataset):
- Actinic keratosis is "indeterminate" at level 1, as in the ISIC-Dx hierarchy
  ("Indeterminate epidermal proliferations").
- SCC in situ / Bowen's disease and keratoacanthoma go to SCC.
- Seborrheic keratosis, solar lentigo, lentigo and lichen-planus-like keratosis go to BKL.
- Melanoma in situ, lentigo maligna and melanoma metastasis go to MEL.
- Epidermal and sebaceous nevi are not melanocytic: OTHER, benign.
- Atypical / intermediate melanocytic lesions (severely dysplastic, Spitz tumours with
  atypia, atypical intraepithelial melanocytic proliferation, ...) go to OTHER,
  indeterminate, ambiguous.

Keyword rules look only at the most specific level of a diagnosis path; a specific
diagnosis without a rule is OTHER, never re-labelled from a more general level.
When the most specific level is only an ISIC-Dx group name (e.g. "Benign melanocytic
proliferations" with no level 3), the group goes to its dominant shared class and is
flagged ambiguous, so it is counted where it most likely belongs and its share stays visible.

The ISIC 2020 'diagnosis' vocabulary is taken from the dataset publication;
harmonize() raises on any label not listed, so a mismatch is caught on the first run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

LEVEL1 = ["malignant", "benign", "indeterminate"]
LEVEL2 = ["MEL", "NV", "BCC", "SCC", "AK", "BKL", "DF", "VASC", "OTHER", "UNSPECIFIED"]
SHARED_LEVEL2 = ["MEL", "NV", "BCC", "SCC", "AK", "BKL", "DF", "VASC"]


@dataclass(frozen=True)
class Target:
    l2: str
    l1: str
    ambiguous: bool = False
    note: str = ""


def T(l2, l1, ambiguous=False, note=""):
    assert l2 in LEVEL2 and l1 in LEVEL1, (l2, l1)
    return Target(l2, l1, ambiguous, note)


_ISIC8 = {
    "MEL": T("MEL", "malignant"), "NV": T("NV", "benign"), "BCC": T("BCC", "malignant"),
    "AK": T("AK", "indeterminate"), "BKL": T("BKL", "benign"), "DF": T("DF", "benign"),
    "VASC": T("VASC", "benign"), "SCC": T("SCC", "malignant"),
}

NATIVE_MAPS: dict[str, dict[str, Target]] = {
    # ISIC 2019 challenge classes
    "isic2019": dict(_ISIC8),
    # HIBA: the adapter already maps ISIC-Dx to the ISIC 2019 classes
    "hiba": dict(_ISIC8),
    # MILK10k (ISIC 2025) ground-truth columns
    "milk10k": {
        "MEL": T("MEL", "malignant"), "NV": T("NV", "benign"), "BCC": T("BCC", "malignant"),
        "AKIEC": T("AK", "indeterminate", True, "actinic keratosis + intraepidermal carcinoma"),
        "BKL": T("BKL", "benign"), "DF": T("DF", "benign"), "VASC": T("VASC", "benign"),
        "SCCKA": T("SCC", "malignant", False, "squamous cell carcinoma + keratoacanthoma"),
        "INF": T("OTHER", "benign", False, "inflammatory and infectious"),
        "BEN_OTH": T("OTHER", "benign"), "MAL_OTH": T("OTHER", "malignant"),
    },
    # PAD-UFES-20 'diagnostic' column (Pacheco et al. 2020); SCC includes Bowen's disease
    "pad_ufes_20": {
        "ACK": T("AK", "indeterminate"), "BCC": T("BCC", "malignant"), "MEL": T("MEL", "malignant"),
        "NEV": T("NV", "benign"), "SCC": T("SCC", "malignant", False, "includes Bowen's disease"),
        "SEK": T("BKL", "benign"),
    },
    # ISIC 2020 'diagnosis' column; 'unknown' rows are benign per benign_malignant
    "isic2020": {
        "melanoma": T("MEL", "malignant"), "nevus": T("NV", "benign"),
        "seborrheic keratosis": T("BKL", "benign"), "lentigo NOS": T("BKL", "benign"),
        "solar lentigo": T("BKL", "benign"), "lichenoid keratosis": T("BKL", "benign"),
        "cafe-au-lait macule": T("OTHER", "benign"),
        "atypical melanocytic proliferation": T("OTHER", "indeterminate", True),
        "unknown": T("UNSPECIFIED", "benign", False, "no specific diagnosis; benign per benign_malignant"),
    },
}

# Datasets mapped from the ISIC-Dx hierarchy (diagnosis_1 ... diagnosis_5, or iddx_1 ... iddx_5)
ISIC_DX_DATASETS = {"slice3d", "isic_archive"}

# ISIC-Dx level-2 group names used as the most specific level: dominant class, ambiguous.
# Full-string patterns, so a specific diagnosis that merely contains these words is not caught.
_GROUP_NOTE = "ISIC-Dx group only (no specific diagnosis); dominant class of the group"
_GROUP_RULES: list[tuple[str, Target]] = [
    (r"^benign melanocytic proliferations$", T("NV", "benign", True, _GROUP_NOTE)),
    (r"^benign epidermal proliferations$", T("BKL", "benign", True, _GROUP_NOTE)),
    (r"^benign soft tissue proliferations - fibro-histiocytic$", T("DF", "benign", True, _GROUP_NOTE)),
    (r"^malignant adnexal epithelial proliferations - follicular$", T("BCC", "malignant", True, _GROUP_NOTE)),
    (r"^malignant epidermal proliferations$", T("SCC", "malignant", True, _GROUP_NOTE)),
    (r"^indeterminate epidermal proliferations$", T("AK", "indeterminate", True, _GROUP_NOTE)),
]

# Ordered keyword rules: the first match on the most specific diagnosis wins.
_RULES: list[tuple[str, Target]] = [
    (r"melanoma|lentigo maligna", T("MEL", "malignant")),
    (r"\bmis\b|in[- ]situ melanoma", T("MEL", "malignant")),
    (r"basal cell|\bbcc\b", T("BCC", "malignant")),
    (r"squamous cell|\bscc|keratoacanthoma|bowen|intraepidermal carcinoma", T("SCC", "malignant")),
    (r"actinic|solar keratosis|\bak\b|\backs?\b", T("AK", "indeterminate")),
    (r"atypical.*melanocytic|intermediate melanocytic|severely dysplastic|melanocytoma|spitz tumou?r"
     r"|\bsem\b|surgically eligible",
     T("OTHER", "indeterminate", True, "intermediate melanocytic")),
    (r"epidermal nevus|nevus sebaceus|sebaceous nevus", T("OTHER", "benign", False, "non-melanocytic nevus")),
    (r"not melanocytic nevus|flat melanotic", T("BKL", "benign", True, "flat melanotic pigmentation, unspecified")),
    (r"nevus|naevus|nevi", T("NV", "benign")),
    (r"seborrh|lentigo|lichen planus like|lichenoid keratosis|benign keratosis|\bsk\b", T("BKL", "benign")),
    (r"dermatofibroma", T("DF", "benign")),
    (r"vascular|hemangioma|haemangioma|angioma|angiokeratoma|pyogenic granuloma", T("VASC", "benign")),
]
_TOP = {"benign": "benign", "malignant": "malignant", "indeterminate": "indeterminate"}


def from_text(levels: list) -> Target:
    """Map an ISIC-Dx diagnosis path (most general first) with the keyword rules.

    Only the most specific non-empty level is matched. With no match: OTHER if a
    specific diagnosis exists, UNSPECIFIED if only the top level (benign / malignant /
    indeterminate) is known. A rule whose level 1 contradicts the dataset's own top
    level (malignant vs benign) stops the run, so a wrong rule cannot pass silently.
    """
    vals = [str(v).strip() for v in levels if isinstance(v, str) and str(v).strip()]
    if not vals:
        raise ValueError("empty diagnosis")
    top = next((_TOP[k] for k in _TOP if vals[0].lower().startswith(k)), None)
    specific = vals[-1].lower()
    for pat, tgt in _GROUP_RULES + _RULES:
        if re.search(pat, specific):
            # the shared convention decides level 1 (e.g. AK is indeterminate even where a
            # dataset files it under benign); only a malignant / benign clash is an error
            if {top, tgt.l1} == {"malignant", "benign"}:
                raise ValueError(f"rule gives {tgt.l1} but the dataset says {top}: {vals}")
            return tgt
    if top is None:
        raise ValueError(f"no rule matches diagnosis {vals}")
    return T("UNSPECIFIED", top) if len(vals) == 1 else T("OTHER", top)


def harmonize(dataset: str, df: pd.DataFrame, dx_columns: list[str] | None = None) -> pd.DataFrame:
    """Per-image level-1 / level-2 labels and the ambiguity flag.

    Native-map datasets use df['label'], or dx_columns[0] when given (e.g. ISIC 2020,
    whose audit label is binary but whose 'diagnosis' column carries the native class);
    ISIC-Dx and keyword datasets use dx_columns (most general first). Raises on any
    label without a mapping.
    """
    if dataset in NATIVE_MAPS:
        m = NATIVE_MAPS[dataset]
        col = dx_columns[0] if dx_columns else "label"
        unknown = sorted(set(df[col].dropna()) - set(m))
        if unknown:
            raise ValueError(f"{dataset}: labels without a taxonomy mapping: {unknown}")
        tg = df[col].map(m)
    elif dataset in ISIC_DX_DATASETS:
        assert dx_columns, f"{dataset}: pass the diagnosis columns (most general first)"
        # map each distinct diagnosis path once (large datasets repeat a few paths many times)
        paths = df[dx_columns].astype(object).where(df[dx_columns].notna(), None)
        keys = list(map(tuple, paths.to_numpy()))
        lookup = {k: from_text(list(k)) for k in set(keys)}
        tg = pd.Series([lookup[k] for k in keys], index=df.index)
    else:
        raise KeyError(f"{dataset}: no taxonomy mapping defined")
    return pd.DataFrame({"l1": [t.l1 for t in tg], "l2": [t.l2 for t in tg],
                         "ambiguous": [t.ambiguous for t in tg]}, index=df.index)


def harmonized_summary(h: pd.DataFrame, groups: pd.Series) -> dict:
    """Counts per level-1 and level-2 class (images and groups) for summary.json."""
    def counts(col, order):
        imgs = h[col].value_counts().reindex(order).fillna(0).astype(int)
        grps = groups.groupby(h[col]).nunique().reindex(order).fillna(0).astype(int)
        return {"images": imgs.to_dict(), "groups": grps.to_dict()}
    present = [c for c in SHARED_LEVEL2 if (h["l2"] == c).any()]
    return {"level1": counts("l1", LEVEL1), "level2": counts("l2", LEVEL2),
            "shared_classes_present": present,
            "ambiguous_share": round(float(h["ambiguous"].mean()), 4),
            "unspecified_share": round(float((h["l2"] == "UNSPECIFIED").mean()), 4)}


def mapping_table() -> pd.DataFrame:
    """Every native mapping as one table (for the comparison repository's README)."""
    rows = [{"dataset": d, "native_label": k, "level2": t.l2, "level1": t.l1, "ambiguous": t.ambiguous,
             "note": t.note} for d, m in NATIVE_MAPS.items() for k, t in m.items()]
    rows += [{"dataset": d, "native_label": "(ISIC-Dx path)", "level2": "rules", "level1": "from diagnosis_1",
              "ambiguous": np.nan, "note": "keyword rules in taxonomy._RULES"} for d in sorted(ISIC_DX_DATASETS)]
    return pd.DataFrame(rows)
