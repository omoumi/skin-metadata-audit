# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Patient-swap and CRT tests when the age field itself is the target (age missing for whole
patients). null: missingness depends on sex, which is related to the label; power: it also
depends on the label."""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests")
from skinaudit import shortcut
from skinaudit.mechanism import missingness_mechanism
from conftest import make_spec
shortcut.MODEL_PARAMS = dict(class_weight="balanced", max_iter=30, early_stopping=False)
scenario, nsim, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]


def sample(rng, power, n_pat=250):
    rows = []
    for p in range(n_pat):
        base = rng.random() < 0.4
        sex = "male" if rng.random() < (0.7 if base else 0.4) else "female"
        p_miss = 0.08 + 0.17 * (sex == "male") + (0.25 * base if power else 0.0)
        miss = rng.random() < p_miss
        age = float(rng.integers(20, 85))
        acq = rng.choice(["c1", "c2"])
        for i in range(int(rng.integers(1, 5))):
            lab = base if rng.random() < 0.85 else not base
            rows.append({"image_id": f"{p}_{i}", "group_id": f"P{p}", "lesion_id": f"P{p}_{i}",
                         "label": "B" if lab else "A", "acquisition": acq, "sex": sex,
                         "site": rng.choice(["trunk", "head", "leg"]), "age_approx": np.nan if miss else age})
    return pd.DataFrame(rows)


power = scenario == "power"
res = []
for sim in range(nsim):
    df = sample(np.random.default_rng(500 + sim), power)
    t = missingness_mechanism(df, make_spec(classes=["A", "B"]), df["lesion_id"].to_numpy(), n_perm=19, seed=sim, n_jobs=4)
    r = t[(t["fields"] == "age_approx") & (t["variant"] == "with_indicators")].iloc[0]
    res.append({"crt": float(r["crt_p"]), "swap": float(r["group_swap_p"]), "gain": float(r["auc_gain"])})
    json.dump(res, open(out, "w"))
