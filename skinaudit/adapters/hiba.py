# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""HIBA Skin Lesions (ISIC Archive collection 175) -> common schema.

A thin wrapper around the generic ISIC Archive adapter (isic_archive.py). It keeps the
HIBA-specific check that every diagnosis maps to one of the eight ISIC 2019 classes.
Download first with isic_archive.download(175, raw_dir), or:
    pip install isic-cli==12.7.0
    isic image download --collections 175 <raw_dir>
"""
from __future__ import annotations

from ..schema import CLASSES_8, AuditDataset
from . import isic_archive

COLLECTION_ID = 175


def load(raw_dir: str = "/kaggle/temp/hiba_raw", dermoscopic_only: bool = False,
         read_image_headers: bool = True, min_resolution_group: int = 30) -> AuditDataset:
    return isic_archive.load(COLLECTION_ID, raw_dir, dermoscopic_only=dermoscopic_only,
                             read_image_headers=read_image_headers, min_group=min_resolution_group,
                             allowed_labels=CLASSES_8)
