# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
import os

import numpy as np

from conftest import make_frame
from skinaudit import parallel
from skinaudit.parallel import available_cpus, resolve_n_jobs
from skinaudit.shortcut import run_feature_set


def test_available_cpus_is_sane():
    n = available_cpus()
    assert 1 <= n <= (os.cpu_count() or 1)


def test_cgroup_quota_caps_cpu_count(monkeypatch):
    monkeypatch.setattr(parallel, "_cgroup_cpu_limit", lambda: 1.5)
    assert available_cpus() == 1


def test_resolve_n_jobs():
    n = available_cpus()
    assert resolve_n_jobs(None) == resolve_n_jobs(-1) == n
    assert resolve_n_jobs(10_000) == n
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(-2) == max(1, n - 1)


def test_result_does_not_depend_on_n_jobs():
    df = make_frame(n_patients=80, seed=11, size_label_link=True)
    X = np.c_[np.random.default_rng(0).normal(size=len(df)),
              df["group_id"].map(df["group_id"].value_counts()).to_numpy(float)]
    args = dict(name="x", X=X, cat_mask=[False, False], y=df["label"].to_numpy(), groups=df["group_id"].to_numpy(),
                n_perm=3, units=df["lesion_id"].to_numpy(), n_perm_reference=3)
    one = run_feature_set(**args, n_jobs=1)
    two = run_feature_set(**args, n_jobs=2)
    for k in ["bacc_mean", "null_mean", "null_sd", "null_group_mean", "null_group_sd", "p_perm", "p_perm_group"]:
        assert one[k] == two[k], k
