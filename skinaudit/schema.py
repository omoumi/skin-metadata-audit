# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Common schema that every dataset adapter must produce.

An adapter returns an ``AuditDataset``: one row per image, with standard column
names, plus a spec describing which columns are metadata fields. All audit,
plot, gallery and report code reads only this schema, so a new dataset needs
only a new adapter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CLASSES_8 = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Required columns in AuditDataset.df
REQUIRED = ["image_id", "group_id", "label", "acquisition"]
# Optional columns that unlock extra checks when present: patient_id, lesion_id, image_path,
# width, height, file_bytes, license, attribution


@dataclass
class LicenseInfo:
    name: str                      # e.g. "CC-BY-NC 4.0"
    url: str                       # license text URL
    allows_redistribution: bool    # may sample images be shown in a public repo
    non_commercial: bool
    citations: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class AuditSpec:
    name: str                      # short id, e.g. "isic2019"
    title: str                     # human-readable name
    num_fields: list[str]          # numeric metadata fields
    cat_fields: list[str]          # categorical metadata fields
    core_fields: list[str]         # fields used for the "core" complete-case subset
    group_level: str               # "patient", "lesion" or "image": what group_id means
    acquisition_name: str          # what the acquisition column holds, e.g. "source"
    license: LicenseInfo
    classes: list[str] = field(default_factory=lambda: list(CLASSES_8))
    source_url: str = ""
    excluded_columns: dict[str, str] = field(default_factory=dict)  # column -> reason
    taxonomy: str = ""             # key in skinaudit.taxonomy for the cross-dataset comparison
    taxonomy_dx_columns: list[str] = field(default_factory=list)  # ISIC-Dx / free-text columns, general first

    @property
    def fields(self) -> list[str]:
        return list(self.num_fields) + list(self.cat_fields)


@dataclass
class AuditDataset:
    spec: AuditSpec
    df: pd.DataFrame
    inputs: dict = field(default_factory=dict)  # file paths, md5s, row counts
    # optional boolean mask over df rows: shortcut tests use only these rows (very large datasets)
    cv_rows_mask: np.ndarray | None = None
    # optional rule that built cv_rows_mask from the labels (labels over df rows -> mask); when
    # the subsample depends on the label, the permutation nulls re-apply it to permuted labels
    cv_rule: object = None

    def validate(self) -> "AuditDataset":
        df, spec = self.df, self.spec
        missing = [c for c in REQUIRED + spec.fields if c not in df.columns]
        assert not missing, f"adapter output lacks columns: {missing}"
        assert df["image_id"].is_unique, "image_id must be unique"
        assert df.index.is_unique, "df index must be unique (use reset_index(drop=True))"
        assert df["group_id"].notna().all(), "group_id has empty values"
        assert df["acquisition"].notna().all(), "acquisition has empty values"
        bad = sorted(set(df["label"]) - set(spec.classes))
        assert not bad, f"labels outside spec.classes: {bad}"
        for f in spec.num_fields:
            assert pd.api.types.is_numeric_dtype(df[f]), f"{f} must be numeric"
        assert set(spec.core_fields) <= set(spec.fields)
        if self.cv_rows_mask is not None:
            m = np.asarray(self.cv_rows_mask)
            assert m.dtype == bool and m.shape == (len(df),), "cv_rows_mask must be a bool array, one entry per row"
        if self.cv_rule is not None:
            assert self.cv_rows_mask is not None and callable(self.cv_rule), "cv_rule needs cv_rows_mask"
            assert np.array_equal(np.asarray(self.cv_rule(df["label"].to_numpy()), bool), np.asarray(self.cv_rows_mask)), \
                "cv_rule(labels) must reproduce cv_rows_mask"
        return self
