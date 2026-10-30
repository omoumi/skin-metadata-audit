# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""The fast code paths give the same numbers as the straightforward pandas versions."""
import os

import numpy as np
import pandas as pd
from PIL import Image

from skinaudit.stats import cramers_v, grouped_bootstrap_ci, smd_categorical, smd_numeric


def test_bincount_bootstrap_equals_crosstab_bootstrap():
    r = np.random.default_rng(3)
    n = 3000
    a = pd.Series(r.choice(["x", "y", "z"], n)).where(r.random(n) > 0.1)      # with missing values
    b = pd.Series(r.choice(list("abcdefgh"), n, p=[.4, .2, .1, .1, .1, .05, .04, .01]))
    g = r.integers(0, 400, n).astype(str)
    fast = grouped_bootstrap_ci(a, b, g, n_boot=50, seed=7)
    slow = grouped_bootstrap_ci(a, b, g, stat=lambda x, y: cramers_v(x, y)["v_corrected"], n_boot=50, seed=7)
    assert np.allclose(fast, slow, rtol=0, atol=1e-12)


def _reference_smd(df, spec, n_boot, seed):
    """Reference implementation: pandas on a resampled DataFrame."""
    variables = ["label", "acquisition", *spec.fields]

    def all_rows(d):
        out = []
        for subset in ["complete", "complete_core"]:
            inside = d[subset].to_numpy() == 1
            for ref in [np.ones(len(d), bool), ~inside]:
                for v in variables:
                    a, b = d[v].to_numpy()[inside], d[v].to_numpy()[ref]
                    out.append(smd_numeric(a, b) if v in spec.num_fields else smd_categorical(a, b))
        return out
    point = all_rows(df)
    uniq, inv = np.unique(df["group_id"].astype(str).to_numpy(), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.flatnonzero(np.diff(inv[order])) + 1)
    rng = np.random.default_rng(seed)
    boots = [all_rows(df.iloc[np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])])
             for _ in range(n_boot)]
    return np.asarray(point, float), np.asarray(boots, float)


def test_array_smd_equals_dataframe_smd():
    from conftest import make_frame, make_spec
    from skinaudit.cc_bias import complete_case_smd
    df = make_frame(n_patients=60, seed=11)
    spec = make_spec()
    miss = df[spec.fields].isna()
    df["complete"] = (~miss).all(axis=1).astype(int)
    df["complete_core"] = (~miss[spec.core_fields]).all(axis=1).astype(int)
    t = complete_case_smd(df, spec, n_boot=20, seed=4)
    point, boots = _reference_smd(df, spec, n_boot=20, seed=4)
    assert np.allclose(t["smd"].to_numpy(float), point, atol=1e-9, equal_nan=True)
    cat = (t["kind"] == "categorical").to_numpy()
    med = np.nanmedian(boots, axis=0)
    expected = np.where(cat, np.clip(2 * point - med, 0, None), point)
    assert np.allclose(t["smd_corrected"].to_numpy(float), expected, atol=1e-9, equal_nan=True)
    assert np.allclose(t["ci_high"].to_numpy(float),
                       np.where(cat, np.clip(2 * point - np.nanquantile(boots, 0.025, axis=0), 0, None),
                                np.nanquantile(boots, 0.975, axis=0)), atol=1e-9, equal_nan=True)


def test_image_stats_cache_reuses_and_refreshes(tmp_path):
    from skinaudit import io
    io.clear_image_stats_cache()
    p = str(tmp_path / "a.png")
    Image.new("RGB", (64, 64), (10, 20, 30)).save(p)
    s = pd.Series([p, p])
    first = io.image_stats(s, size=32)
    assert len(io._STATS_CACHE) == 1
    assert first.equals(io.image_stats(s, size=32))
    Image.new("RGB", (64, 64), (200, 20, 30)).save(p)          # file changes: must be recomputed
    os.utime(p, ns=(1, 10 ** 18))
    assert io.image_stats(s, size=32)["mean_r"].iloc[0] > first["mean_r"].iloc[0] + 100
