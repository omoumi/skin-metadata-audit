# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""CPU detection and a small parallel map for independent model fits."""
from __future__ import annotations

import os


def _cgroup_cpu_limit() -> float | None:
    """CPU limit set by the container (Kaggle, Docker), or None if there is none."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def available_cpus() -> int:
    """Number of CPUs this process may really use.

    os.cpu_count() reports the host machine, which in a container (Kaggle) can be
    more than the session is allowed to use; the scheduler affinity and the cgroup
    quota give the real limit. The smallest of the three wins.
    """
    counts = [os.cpu_count() or 1]
    if hasattr(os, "sched_getaffinity"):
        counts.append(len(os.sched_getaffinity(0)))
    limit = _cgroup_cpu_limit()
    if limit is not None:
        counts.append(max(1, int(limit)))
    return max(1, min(counts))


def resolve_n_jobs(n_jobs: int | None) -> int:
    """None or -1 -> every available CPU; other values are capped at what is available."""
    cpus = available_cpus()
    if n_jobs is None or n_jobs == -1:
        return cpus
    if n_jobs < -1:  # joblib convention: -2 = all but one
        return max(1, cpus + 1 + n_jobs)
    return max(1, min(int(n_jobs), cpus))


def _call_single_threaded(fn, args):
    """Run fn(*args) with native thread pools limited to one thread (no oversubscription)."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return fn(*args)


def parallel_map(fn, arg_list: list[tuple], n_jobs: int) -> list:
    """[fn(*a) for a in arg_list], spread over n_jobs processes; order is preserved."""
    if n_jobs <= 1 or len(arg_list) <= 1:
        return [_call_single_threaded(fn, a) for a in arg_list]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(_call_single_threaded)(fn, a) for a in arg_list)
