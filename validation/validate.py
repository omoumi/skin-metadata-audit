# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Validation of the audit's tests and intervals by simulation (data with a known answer)."""
import sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
from skinaudit import shortcut, mechanism
# a smaller model than the audit's (30 instead of 100 rounds) keeps hundreds of replications
# affordable; pass "full" as the fifth argument to use the audit's own settings (needed when a
# rare label must be picked up by the model, as in the SLICE-3D style mechanism scenarios)
FULL = len(sys.argv) > 5 and sys.argv[5] == "full"
if not FULL:
    shortcut.MODEL_PARAMS = dict(class_weight="balanced", max_iter=30, early_stopping=False)
    mechanism.LABEL_MODEL_MAX_ITER = 60
from skinaudit.shortcut import run_feature_set
from skinaudit.mechanism import label_dependence
scenario, nsim, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]


def patients(rng, n_pat=200, concord=0.85, signal=0.0, max_img=8):
    """Patients with 1..max_img images (one lesion per image, as in ISIC 2020); labels alike
    within a patient; a patient-level camera whose link to the label is `signal`."""
    rows = []
    for p in range(n_pat):
        base = rng.random() < 0.3
        cam = rng.integers(0, 4)
        if signal and rng.random() < signal:
            cam = 3 if base else rng.integers(0, 3)
        age = rng.normal(50, 15)
        n = int(rng.integers(1, max_img + 1))
        for i in range(n):
            lab = base if rng.random() < concord else (not base)
            rows.append((p, cam, age, int(lab), n))
    return pd.DataFrame(rows, columns=["g", "cam", "age", "y", "size"])


res = []
t0 = time.time()
for sim in range(nsim):
    rng = np.random.default_rng(10_000 + sim)
    if scenario in ("shortcut_nested_null", "shortcut_nested_power"):
        d = patients(rng, signal=0.0 if scenario.endswith("null") else 0.25)
        # 19 draws for both nulls, so the lesion null's false-positive rate is measured too
        r = run_feature_set("cam", d[["cam"]].to_numpy(float), [True], d["y"].to_numpy(), d["g"].to_numpy(),
                            n_perm=19, n_perm_reference=19, units=np.arange(len(d)), n_jobs=2, n_repeats=1, seed=sim)
        res.append({k: r[k] for k in ["p_perm", "p_perm_unit", "p_perm_group", "test_null", "gain"]})
    elif scenario == "shortcut_lesion_groups_null":
        # groups = lesions with 1-3 images (MILK10k / ISIC 2019 style); feature constant per lesion
        rows = []
        for l in range(400):
            lab, cam = int(rng.random() < 0.3), rng.integers(0, 4)
            for i in range(int(rng.integers(1, 4))):
                rows.append((l, cam, lab))
        d = pd.DataFrame(rows, columns=["g", "cam", "y"])
        r = run_feature_set("cam", d[["cam"]].to_numpy(float), [True], d["y"].to_numpy(), d["g"].to_numpy(),
                            n_perm=19, n_perm_reference=5, units=d["g"].to_numpy(), n_jobs=2, n_repeats=1, seed=sim)
        res.append({k: r[k] for k in ["p_perm", "p_perm_unit", "p_perm_group", "test_null"]})
    elif scenario == "shortcut_rare_classes_null":
        # 8 classes, three of them rare (8-15 patients); labels alike within a patient;
        # camera per patient, unrelated to the label
        sizes = [80, 50, 30, 20, 15, 15, 10, 8]
        rows = []
        for c, n_pat in enumerate(sizes):
            for p in range(n_pat):
                cam = rng.integers(0, 4)
                for i in range(int(rng.integers(1, 6))):
                    lab = c if rng.random() < 0.85 else rng.integers(0, 8)
                    rows.append((f"{c}_{p}", cam, lab))
        d = pd.DataFrame(rows, columns=["g", "cam", "y"])
        y = np.array([f"k{v}" for v in d["y"]])
        r = run_feature_set("cam", d[["cam"]].to_numpy(float), [True], y, d["g"].to_numpy(),
                            n_perm=19, n_perm_reference=5, units=np.arange(len(d)), n_jobs=2, n_repeats=1, seed=sim)
        res.append({k: r[k] for k in ["p_perm", "test_null", "gain", "min_classes_in_test_fold"]})
    elif scenario.startswith(("shortcut_subsample", "mech_subsample")):
        # SLICE-3D style: patients with many tiles, malignant tiles rare; the tests run on every
        # malignant tile plus up to 20 random benign tiles per patient, and the nulls re-apply
        # that rule to the permuted labels. "power": the camera (constant within a patient)
        # raises the malignancy risk, or (mechanism) a patient with a malignant tile more often
        # lacks the field.
        from skinaudit.adapters.slice3d import subsample_rule
        power = scenario.endswith("power")
        rows = []
        for p in range(300):
            cam, age = int(rng.integers(0, 4)), rng.normal(55, 12)
            n = int(rng.integers(25, 100))
            if scenario.startswith("shortcut") and power:
                risk = 0.03 if cam == 3 else 0.004
            else:
                risk = 0.03 if rng.random() < 0.25 else 0.0
            for i in range(n):
                rows.append((f"P{p}", cam, age, "malignant" if rng.random() < risk else "benign"))
        d = pd.DataFrame(rows, columns=["g", "cam", "age", "y"])
        y_all, g_all = d["y"].to_numpy(), d["g"].to_numpy()
        rule = subsample_rule(g_all, benign_cap=20, seed=sim)
        m = rule(y_all)
        if scenario.startswith("shortcut"):
            X_all = d[["cam"]].to_numpy(float)
            u_all = np.arange(len(d))
            common = dict(n_perm=19, n_perm_reference=5, n_jobs=2, n_repeats=1, seed=sim)
            r = run_feature_set("cam", X_all[m], [True], y_all[m], g_all[m], units=u_all[m],
                                resample={"X": X_all, "y": y_all, "groups": g_all, "units": u_all, "rule": rule},
                                **common)
            out_r = {"p_perm": r["p_perm"], "test_null": r["test_null"], "gain": r["gain"],
                     "degenerate": r["null_group_degenerate"]}
            if power:   # the same test without re-applying the rule, for comparison
                r0 = run_feature_set("cam", X_all[m], [True], y_all[m], g_all[m], units=u_all[m], **common)
                out_r.update({"p_plain": r0["p_perm"], "plain_degenerate": r0["null_group_degenerate"]})
            res.append(out_r)
        else:
            has_mal = d.groupby("g")["y"].transform(lambda s: (s == "malignant").any()).to_numpy()
            # null: missingness depends on age only; power: 0.2, or 0.8 for a patient with a malignant tile
            pm = 0.2 + 0.6 * has_mal if power else 1 / (1 + np.exp(-(d["age"].to_numpy() - 60) / 10))
            u = d.groupby("g")["g"].transform(lambda s: rng.random()).to_numpy()
            target = (u < pm).astype(int)                       # decided per patient
            size_all = d.groupby("g")["g"].transform("size").to_numpy(float)
            X_all = np.c_[d["age"].to_numpy(), d["cam"].to_numpy(float), size_all]
            coarse = d["cam"].astype(str).to_numpy()
            band = pd.qcut(d.groupby("g")["age"].median(), 5, labels=False).astype(str)
            fine = coarse + "|" + d["g"].map(band).to_numpy()
            units = np.arange(len(d)).astype(str)
            r = label_dependence(X_all[m], [False, True, False], target[m], y_all[m], g_all[m], units[m],
                                 strata_fine=fine[m], strata_coarse=coarse[m], n_perm=19, seed=sim, n_jobs=2,
                                 resample={"X": X_all, "target": target, "y": y_all, "groups": g_all,
                                           "strata_fine": fine, "strata_coarse": coarse, "rule": rule})
            res.append({k: r.get(k) for k in ["auc_gain", "crt_p", "group_swap_p", "group_swap_degenerate",
                                              "group_swap_movable_share"]})
    elif scenario == "gain_ci":
        d = patients(rng, signal=0.4)
        r = run_feature_set("cam", d[["cam"]].to_numpy(float), [True], d["y"].to_numpy(), d["g"].to_numpy(),
                            n_perm=2, n_perm_reference=20, units=np.arange(len(d)), n_jobs=2, n_repeats=5, seed=sim)
        res.append({k: r[k] for k in ["bacc_mean", "bacc_ci_low", "bacc_ci_high", "gain", "gain_ci_low",
                                      "gain_ci_high", "bacc_split1"]})
    elif scenario in ("mech_null", "mech_power"):
        d = patients(rng)
        lab = d["y"].to_numpy()
        pm = 1 / (1 + np.exp(-(d["age"].to_numpy() - 60) / 10))
        if scenario == "mech_power":
            pm = np.clip(pm + 0.25 * lab, 0, 1)
        # missingness decided per patient (first image's draw)
        u = d.groupby("g")["g"].transform(lambda s: rng.random()).to_numpy()
        target = (u < pm).astype(int)
        X = d[["age", "cam", "size"]].to_numpy(float)
        coarse = d["cam"].astype(str).to_numpy()
        band = pd.qcut(d.groupby("g")["age"].median(), 5, labels=False).astype(str)
        fine = coarse + "|" + d["g"].map(band).to_numpy()
        r = label_dependence(X, [False, True, False], target, np.where(lab == 1, "m", "b"), d["g"].to_numpy(),
                             np.arange(len(d)).astype(str), strata_fine=fine, strata_coarse=coarse, n_perm=19,
                             seed=sim, n_jobs=2)
        res.append({k: r.get(k) for k in ["auc_gain", "crt_p", "group_swap_p", "label_r2", "group_swap_movable_share"]})
    json.dump({"scenario": scenario, "seconds": round(time.time() - t0), "runs": res}, open(out, "w"))
