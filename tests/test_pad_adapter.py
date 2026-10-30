# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""PAD-UFES-20 adapter on a tiny hand-made metadata.csv (no real data needed)."""
import numpy as np
import pandas as pd
from PIL import Image

from skinaudit.adapters import pad_ufes_20 as pad

COLS = ["patient_id", "lesion_id", "smoke", "drink", "background_father", "background_mother", "age", "pesticide",
        "gender", "skin_cancer_history", "cancer_history", "has_piped_water", "has_sewage_system", "fitspatrick",
        "region", "diameter_1", "diameter_2", "diagnostic", "itch", "grew", "hurt", "changed", "bleed", "elevation",
        "img_id", "biopsed"]


def _write(tmp_path):
    rows = []
    for i, (pat, les, grew, father) in enumerate([("PAT_1", "10", "UNK", "ITALY"), ("PAT_1", "10", "UNK", "ITALY"),
                                                  ("PAT_2", "10", "", ""), ("PAT_3", "11", "True", "UNK")]):
        r = dict.fromkeys(COLS, "")
        r.update(patient_id=pat, lesion_id=les, age="50", region="FACE", diagnostic=["BCC", "BCC", "NEV", "MEL"][i],
                 itch="False", grew=grew, hurt="False", changed="False", bleed="False", elevation="True",
                 background_father=father, img_id=f"{pat}_{les}_{i}.png", biopsed="True")
        rows.append(r)
        Image.new("RGB", (40, 30)).save(tmp_path / r["img_id"])
    pd.DataFrame(rows, columns=COLS).to_csv(tmp_path / "metadata.csv", index=False)


def test_unk_is_a_value_and_lesions_are_patient_keyed(tmp_path):
    _write(tmp_path)
    ds = pad.load(str(tmp_path), min_resolution_group=1)
    df = ds.df
    assert (df["grew"] == "UNK").sum() == 2 and df["grew"].isna().sum() == 1
    assert ds.inputs["unk_counts"] == {"background_father": 1, "grew": 2}
    # same raw lesion_id "10" in two patients -> two lesions
    assert ds.inputs["raw_lesion_ids_shared_by_several_patients"] == 1
    assert df["lesion_id"].nunique() == 3
    assert df["image_id"].str.endswith(".png").sum() == 0
    assert np.issubdtype(df["age"].dtype, np.number)
