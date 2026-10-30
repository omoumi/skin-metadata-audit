# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Run the full audit on an AuditDataset and write tables, figures and summary.json."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn

from . import __version__
from .io import BORDER_STAT_COLUMNS, IMAGE_STAT_COLUMNS, image_stats as compute_image_stats
from .schema import AuditDataset
from .parallel import available_cpus, resolve_n_jobs
from .shortcut import category_codes, explain_feature_set, label_constant_within, run_feature_set
from .stats import cramers_v, grouped_bootstrap_ci, holm, kruskal_epsilon2
from .cc_bias import complete_case_smd, max_abs_smd

SUMMARY_SCHEMA_VERSION = "1.0"
# feature sets whose name ends with this are image-derived reference sets (e.g. MONET concepts,
# TBP device measurements), not shortcut hypotheses: they stay out of the Holm family
REFERENCE_SUFFIX = "_reference"


def _prepare(ds: AuditDataset) -> pd.DataFrame:
    spec, df = ds.spec, ds.df.copy()
    for f in spec.fields:
        df[f"miss_{f}"] = df[f].isna().astype(int)
    miss = [f"miss_{f}" for f in spec.fields]
    df["miss_pattern"] = df[miss].astype(str).agg("".join, axis=1)
    df["complete"] = (df[miss].sum(axis=1) == 0).astype(int)
    df["complete_core"] = df[spec.core_fields].notna().all(axis=1).astype(int)
    if {"width", "height"} <= set(df.columns):
        df["resolution"] = df["width"].astype(str) + "x" + df["height"].astype(str)
        top = df["resolution"].value_counts().index[:15]
        df["resolution_top"] = df["resolution"].where(df["resolution"].isin(top), "other")
        df["megapixels"] = df["width"] * df["height"] / 1e6
        df["aspect"] = (df["width"] / df["height"]).round(2)
    return df


def _counts(df, classes, by_col="label"):
    out = pd.DataFrame({"images": df[by_col].value_counts(),
                        "groups": df.groupby(by_col)["group_id"].nunique()})
    if "patient_id" in df.columns and df["patient_id"].notna().any():
        out["patients"] = df.groupby(by_col)["patient_id"].nunique()
    if "lesion_id" in df.columns and df["lesion_id"].notna().any():
        out["lesions"] = df.groupby(by_col)["lesion_id"].nunique()
    return out.reindex(classes).fillna(0).astype(int)


def _naive_p(t: pd.DataFrame) -> pd.DataFrame:
    """Rename the chi-square / Kruskal-Wallis p column to p_naive: those tests treat images as
    independent, while images of one patient or lesion are not, so the p-values come out too
    small. Effect sizes with group-bootstrap intervals are the quantities to read."""
    return t.rename(columns={"p": "p_naive"})


def _strongest_label_dependence(mech: pd.DataFrame | None) -> dict | None:
    """The field (block) with the strongest evidence that its missingness depends on the
    label. Evidence = the larger of the two Holm-adjusted p-values (CRT and group swap), since
    a claim needs both; ties broken by the larger AUC gain. A large gain on a few missing rows
    is noisy, so the gain alone does not pick the row."""
    if mech is None or mech["auc_gain"].dropna().empty:
        return None
    m = mech[mech["auc_gain"].notna()].copy()
    cols = [c for c in ("crt_p_holm", "group_swap_p_holm") if c in m.columns]
    m["_p"] = m[cols].max(axis=1, skipna=False).fillna(np.inf) if cols else np.inf
    r = m.sort_values(["_p", "auc_gain"], ascending=[True, False]).iloc[0]

    def num(v):
        return None if v is None or pd.isna(v) else round(float(v), 4)
    return {"fields": r["fields"], "variant": r["variant"], "auc_gain": num(r["auc_gain"]),
            "crt_p": num(r.get("crt_p")), "crt_p_holm": num(r.get("crt_p_holm")),
            "group_swap_p": num(r.get("group_swap_p")), "group_swap_p_holm": num(r.get("group_swap_p_holm")),
            "supported": bool(r.get("label_dependence_supported")) if not pd.isna(r.get("label_dependence_supported"))
            else None,
            "label_r2": num(r.get("label_r2"))}


def run_audit(ds: AuditDataset, out_dir: str, n_perm: int = 20, n_boot: int = 1000,
              seed: int = 2026, extra_feature_sets: dict | None = None,
              make_figures: bool = True, perm_level: str = "unit", image_stats: bool = True,
              n_perm_reference: int | None = None,
              n_jobs: int | None = None, cv_rows_mask=None, duplicate_pairs=None,
              mechanism: bool = True, n_perm_mechanism: int | None = None,
              min_class_groups: int = 10, n_repeats: int = 5, duplicate_action: str = "merge") -> dict:
    """Compute every audit table, save CSVs + figures + summary.json, return the summary.

    n_perm: label permutations of the null that gives the test (p_perm): the lesion null when
    lesions and CV groups coincide, the size-matched group null when lesions sit inside
    patients (see skinaudit.shortcut.run_feature_set). With Holm over about seven feature
    sets, n_perm must be at least about 150 for any test to reach 0.05.
    n_perm_reference: permutations of the other null (the reference: it gives the chance level
    of the gain, or the size-matched sensitivity); None = n_perm. Its mean and band need
    fewer draws than a p-value, so it can be set lower on large datasets.
    cv_rows_mask: boolean array over ds.df rows; when given, the shortcut tests (and their
    permutations) use only these rows, while every descriptive table uses all rows. For
    very large datasets; the rule that built the mask belongs in ds.inputs. None means
    ds.cv_rows_mask (set by adapters such as slice3d), so the subsample cannot be forgotten.
    mechanism: fit the missingness-mechanism models (skinaudit.mechanism); n_perm_mechanism is
    the number of label draws for their tests (None = n_perm; "auto" = enough for Holm over the
    tested rows, see mechanism.auto_n_perm).
    min_class_groups: in the model-based tests (shortcut CV, feature importance, missingness
    mechanism) classes with fewer groups than this are merged into OTHER, and rows labelled
    UNSPECIFIED (only benign / malignant known) are left out; descriptive tables keep every class.
    duplicate_pairs: confirmed duplicate image pairs (identical files, or checked by eye;
    dedup.load_confirmed), or None.
    duplicate_action: what to do with confirmed duplicate pairs. "merge" (default) keeps the
    images and merges their groups, so copies cannot sit in two CV folds. "drop" removes all
    copies but one (dedup.duplicate_drops) before anything is computed, for datasets where
    the same photos were uploaded twice (copies also distort the descriptive tables); a
    cluster with conflicting labels is removed entirely.
    n_repeats: grouped CV runs (different fold assignments) averaged for the observed score
    of each shortcut test; they give its interval (skinaudit.shortcut.corrected_cv_interval).
    Confirmed duplicate pairs whose two images carry different labels: both images are left
    out of the model-based tests (which label is right is unknown) and their permutation
    units are not merged; their groups are still merged, and descriptive tables keep them.
    """
    if n_perm_reference is None:
        n_perm_reference = n_perm
    n_jobs = resolve_n_jobs(n_jobs)
    print(f"using {n_jobs} CPU(s) for permutation fits (detected {available_cpus()})", flush=True)
    t0 = time.time()
    ds.validate()
    spec = ds.spec
    classes = spec.classes
    tables_dir = os.path.join(out_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    def save(t, name):
        t.to_csv(os.path.join(tables_dir, f"{name}.csv"))
        tables[name] = t
        return t

    df = _prepare(ds)
    # permutation unit: the lesion when known, otherwise the image itself
    if "lesion_id" in df.columns and df["lesion_id"].notna().any():
        df["perm_unit"] = df["lesion_id"].astype(object).where(df["lesion_id"].notna(),
                                                               "image:" + df["image_id"].astype(str))
    duplicates = None
    conflict_ids: set = set()
    assert duplicate_action in ("merge", "drop")
    if duplicate_action == "drop" and duplicate_pairs is not None and len(duplicate_pairs):
        from .dedup import duplicate_drops
        assert ds.cv_rows_mask is None and cv_rows_mask is None, "duplicate_action='drop' does not support a CV subsample"
        drops, clusters = duplicate_drops(df, duplicate_pairs, spec.fields)
        ids = set(df["image_id"])
        used = duplicate_pairs[duplicate_pairs["image_id_1"].isin(ids) & duplicate_pairs["image_id_2"].isin(ids)]
        used.to_csv(os.path.join(tables_dir, "confirmed_duplicates.csv"), index=False)
        pd.DataFrame({"image_id": sorted(drops)}).to_csv(os.path.join(tables_dir, "dropped_duplicates.csv"), index=False)
        n_before = len(df)
        df = df[~df["image_id"].isin(drops)].reset_index(drop=True)
        duplicates = {"action": "drop", "confirmed_pairs": int(len(used)),
                      "pairs_not_in_this_data": int(len(duplicate_pairs) - len(used)),
                      "clusters": len(clusters),
                      "clusters_with_conflicting_labels": sum(c["kept"] is None for c in clusters),
                      "images_before": int(n_before), "images_dropped": int(len(drops)), "images_after": int(len(df)),
                      "cluster_records": clusters}
        duplicate_pairs = None                    # nothing left to merge
    if duplicate_pairs is not None and len(duplicate_pairs):
        from .dedup import merge_duplicate_groups
        n_groups_before = int(df["group_id"].nunique())
        ids = set(df["image_id"])
        used = duplicate_pairs[duplicate_pairs["image_id_1"].isin(ids) & duplicate_pairs["image_id_2"].isin(ids)]
        # pairs outside this data (e.g. a dermoscopic-only run) are skipped and counted
        df["group_id"], merged = merge_duplicate_groups(df, used)
        # two copies of one photo are one lesion: merge their permutation units too, so the
        # null does not give the copies two independent labels
        # pairs whose images carry different labels: which label is right is unknown, so both
        # images leave the model-based tests; their permutation units are not merged (a unit
        # with two labels would force the group-level null on the whole dataset)
        lab_of = df.set_index("image_id")["label"].astype(str)
        differ = (used["image_id_1"].map(lab_of) != used["image_id_2"].map(lab_of)).to_numpy()
        conflict_ids = set(used.loc[differ, "image_id_1"]) | set(used.loc[differ, "image_id_2"])
        if "perm_unit" in df.columns:
            df["perm_unit"], _ = merge_duplicate_groups(df, used[~differ], column="perm_unit")
        duplicates = {"action": "merge", "confirmed_pairs": int(len(used)),
                      "pairs_not_in_this_data": int(len(duplicate_pairs) - len(used)),
                      "pairs_with_conflicting_labels": int(differ.sum()),
                      "conflicting_label_images_left_out_of_model_tests": sorted(conflict_ids),
                      "groups_before": n_groups_before,
                      "groups_after": int(df["group_id"].nunique()), "merged_groups": merged,
                      "pairs": used.to_dict(orient="records")}
        used.to_csv(os.path.join(tables_dir, "confirmed_duplicates.csv"), index=False)
    miss = [f"miss_{f}" for f in spec.fields]
    ind_cols = miss + ["complete", "complete_core"]

    # --- structure -------------------------------------------------------
    per_group = df.groupby("group_id").agg(n_images=("image_id", "size"), n_labels=("label", "nunique"))
    structure = {
        "n_images": int(len(df)), "n_groups": int(len(per_group)), "group_level": spec.group_level,
        "groups_with_multiple_images": int((per_group["n_images"] > 1).sum()),
        "max_images_per_group": int(per_group["n_images"].max()),
        "max_share_one_group": round(float(per_group["n_images"].max() / len(df)), 4),
        "groups_with_several_labels": int((per_group["n_labels"] > 1).sum()),
    }
    if "lesion_id" in df.columns and df["lesion_id"].notna().any():
        per_lesion = df.dropna(subset=["lesion_id"]).groupby("lesion_id")["label"].nunique()
        structure["n_lesions"] = int(len(per_lesion))
        structure["lesions_with_conflicting_labels"] = int((per_lesion > 1).sum())
    if "patient_id" in df.columns and df["patient_id"].notna().any():
        structure["n_patients"] = int(df["patient_id"].nunique())

    # --- within-group consistency -----------------------------------------
    # Only groups with several images can show conflicts or partial missingness;
    # single-image groups would make every check pass trivially.
    group_size = df.groupby("group_id").size()
    multi_ids = group_size.index[group_size > 1]
    dm = df[df["group_id"].isin(multi_ids)]
    by_group = dm.groupby("group_id")
    msize = by_group.size()
    cons_rows = []
    for f in spec.fields:
        n_values = by_group[f].nunique()
        n_missing = dm[f].isna().groupby(dm["group_id"]).sum()
        partly = (n_missing > 0) & (n_missing < msize)
        cons_rows.append({
            "field": f,
            "groups_with_several_images": int(len(msize)),
            "groups_with_conflicting_values": int((n_values > 1).sum()),
            "groups_all_missing": int((n_missing == msize).sum()),
            "groups_partly_missing": int(partly.sum()),
            # None when no multi-image group has this field missing: nothing to decide
            "missingness_is_group_level": (bool(partly.sum() == 0) if len(msize) and n_missing.sum() > 0 else None),
        })
    consistency = save(pd.DataFrame(cons_rows).set_index("field"), "within_group_consistency")
    structure["within_group"] = consistency.to_dict(orient="index")

    save(_counts(df, classes), "class_counts")

    # shared taxonomy for the cross-dataset comparison (the audit itself keeps native labels)
    harmonized = None
    if spec.taxonomy:
        from .taxonomy import harmonize, harmonized_summary
        h = harmonize(spec.taxonomy, df, spec.taxonomy_dx_columns or None)
        save(pd.crosstab(df["label"], [h["l1"], h["l2"]]), "taxonomy_native_to_shared")
        harmonized = {"taxonomy": spec.taxonomy, **harmonized_summary(h, df["group_id"])}
    save(pd.crosstab(df["acquisition"], df["label"]).reindex(columns=classes, fill_value=0), "acquisition_by_class")

    # --- missingness -----------------------------------------------------
    save(df[ind_cols].mean().to_frame("rate"), "missing_rate_overall")
    save(df.groupby("label")[ind_cols].mean().reindex(classes).round(4), "missing_rate_by_class")
    save(df.groupby("acquisition")[ind_cols].mean().round(4), "missing_rate_by_acquisition")
    save(pd.crosstab(df["miss_pattern"], df["label"]).reindex(columns=classes, fill_value=0), "missing_pattern_by_class")

    rows = []
    for m in ind_cols:
        for versus, col in [("label", "label"), ("acquisition", "acquisition")]:
            r = cramers_v(df[m], df[col])
            lo, hi = grouped_bootstrap_ci(df[m], df[col], df["group_id"], n_boot=n_boot, seed=seed)
            rows.append({"indicator": m, "versus": versus, **r, "v_corrected_ci_low": lo, "v_corrected_ci_high": hi})
        for a, g in df.groupby("acquisition"):
            r = cramers_v(g[m], g["label"])
            rows.append({"indicator": m, "versus": f"label_within_{a}", **r})
    miss_assoc = save(_naive_p(pd.DataFrame(rows)), "missing_association")

    # missingness of one field vs the observed values of the other fields (MAR check);
    # numeric fields are cut into quintiles so every cell uses the same statistic
    binned = {f: (pd.qcut(df[f], 5, duplicates="drop").astype("string") if f in spec.num_fields else df[f])
              for f in spec.fields}
    mar_rows = []
    for f in spec.fields:
        m = df[f"miss_{f}"]
        for other in spec.fields:
            if other == f:
                continue
            obs = binned[other].notna()
            r = cramers_v(m[obs], binned[other][obs])
            mar_rows.append({"missing_field": f, "other_field": other, "n_missing_where_other_observed": int(m[obs].sum()),
                             **r})
    mar = save(_naive_p(pd.DataFrame(mar_rows)), "missing_vs_other_fields")
    mar_matrix = save(mar.pivot(index="missing_field", columns="other_field", values="v_corrected")
                      .reindex(index=spec.fields, columns=spec.fields), "missing_vs_other_fields_matrix")

    # Same check within each class, so an association that only runs through the
    # label (e.g. age -> class -> missing history) does not show up here.
    # Cell = size-weighted mean of the within-class corrected V over classes with
    # at least 30 rows and both missing and observed values.
    mar_lab_rows = []
    for f in spec.fields:
        for other in spec.fields:
            if other == f:
                continue
            vs, ns = [], []
            for c, gi in df.groupby("label").groups.items():
                sub = df.loc[gi]
                obs = binned[other].loc[gi].notna()
                m = sub[f"miss_{f}"][obs]
                if obs.sum() < 30 or m.nunique() < 2:
                    continue
                v = cramers_v(m, binned[other].loc[gi][obs])["v_corrected"]
                if not np.isnan(v):
                    vs.append(v)
                    ns.append(int(obs.sum()))
            mar_lab_rows.append({"missing_field": f, "other_field": other, "n_classes_used": len(vs),
                                 "v_corrected_within_label": float(np.average(vs, weights=ns)) if vs else np.nan})
    mar_lab = save(pd.DataFrame(mar_lab_rows), "missing_vs_other_fields_within_label")
    mar_lab_matrix = save(mar_lab.pivot(index="missing_field", columns="other_field",
                                        values="v_corrected_within_label")
                          .reindex(index=spec.fields, columns=spec.fields),
                          "missing_vs_other_fields_within_label_matrix")

    # --- metadata values vs label -----------------------------------------
    value_rows = []
    for f in spec.num_fields:
        save(df.groupby("label")[f].describe().reindex(classes).round(1), f"{f}_by_class")
        k = kruskal_epsilon2(df[f], df["label"])
        value_rows.append({"field": f, "test": "kruskal", "effect": k["epsilon2"], "effect_name": "epsilon_squared",
                           "p": k["p"], "n": k["n"]})
    for f in spec.cat_fields:
        obs = df[df[f].notna()]
        save(pd.crosstab(obs[f], obs["label"], normalize="columns").reindex(columns=classes).round(3),
             f"{f}_share_within_class")
        r = cramers_v(obs[f], obs["label"])
        lo, hi = grouped_bootstrap_ci(obs[f], obs["label"], obs["group_id"], n_boot=n_boot, seed=seed)
        value_rows.append({"field": f, "test": "chi2", "effect": r["v_corrected"], "effect_name": "cramers_v_corrected",
                           "p": r["p"], "n": r["n"], "ci_low": lo, "ci_high": hi})
    value_assoc = save(_naive_p(pd.DataFrame(value_rows)), "value_association")

    # --- excluded columns (label-leak check) -------------------------------
    leak_rows = []
    for c, reason in spec.excluded_columns.items():
        if c in df.columns and c not in ("image_id", "patient_id", "lesion_id"):
            s = df[c].astype("string").fillna("<empty>")
            if 1 < s.nunique() < 50:
                r = cramers_v(s, df["label"])
                leak_rows.append({"column": c, "reason_excluded": reason, "v_corrected": r["v_corrected"], "p": r["p"]})
    leak = save(_naive_p(pd.DataFrame(leak_rows, columns=["column", "reason_excluded", "v_corrected", "p"])),
                "excluded_columns_check")

    # --- acquisition and image properties ---------------------------------
    acq_v = cramers_v(df["acquisition"], df["label"])
    image_props = {}
    if "resolution_top" in df.columns:
        save(pd.crosstab(df["resolution_top"], df["label"]).reindex(columns=classes, fill_value=0), "resolution_by_class")
        save(pd.crosstab(df["resolution_top"], df["acquisition"]), "resolution_by_acquisition")
        save(df.groupby(["acquisition", "label"])[["megapixels", "aspect", "file_bytes"]].median().round(3),
             "image_size_by_acquisition_and_class")
        image_props = {
            "n_distinct_resolutions": int(df["resolution"].nunique()),
            "resolution_vs_label_v_corrected": cramers_v(df["resolution_top"], df["label"])["v_corrected"],
            "resolution_vs_acquisition_v_corrected": cramers_v(df["resolution_top"], df["acquisition"])["v_corrected"],
            "megapixels_vs_label_epsilon2": kruskal_epsilon2(df["megapixels"], df["label"])["epsilon2"],
        }

    # --- image statistics (colour, border, frame) ----------------------------
    stats_assoc = None
    if image_stats and "image_path" in df.columns:
        t_img = time.time()
        df = df.join(compute_image_stats(df["image_path"]))
        if {"width", "height", "file_bytes"} <= set(df.columns):
            df["bytes_per_pixel"] = df["file_bytes"] / (df["width"] * df["height"])
        stat_cols = IMAGE_STAT_COLUMNS + (["bytes_per_pixel"] if "bytes_per_pixel" in df.columns else [])
        save(df.groupby(["acquisition", "label"])[stat_cols].median().round(3), "image_stats_by_acquisition_and_class")
        rows = []
        for c in stat_cols:
            k = kruskal_epsilon2(df[c], df["label"])
            rows.append({"statistic": c, "region": "border" if c in BORDER_STAT_COLUMNS else
                         ("file" if c == "bytes_per_pixel" else "whole image"), "epsilon2_vs_label": k["epsilon2"],
                         "epsilon2_vs_acquisition": kruskal_epsilon2(df[c], df["acquisition"])["epsilon2"]
                         if df["acquisition"].nunique() > 1 else np.nan, "p": k["p"]})
        stats_assoc = save(_naive_p(pd.DataFrame(rows)).set_index("statistic"), "image_stats_association")
        image_props["image_stats_seconds"] = round(time.time() - t_img, 1)

    # --- shortcut tests ---------------------------------------------------
    y, g = df["label"].to_numpy(), df["group_id"].to_numpy()
    # Model-based tests need each class in most test folds: with 5 folds, a class of a few
    # groups is missing from some folds and its recall is noise. Small classes are merged into
    # OTHER and UNSPECIFIED rows are left out of those tests only.
    groups_per_class = df.groupby("label")["group_id"].nunique()
    small = sorted(str(c) for c, n in groups_per_class.items()
                   if n < min_class_groups and c not in ("OTHER", "UNSPECIFIED"))
    y = np.where(np.isin(y, small), "OTHER", y).astype(object)
    model_mask = (df["label"].to_numpy() != "UNSPECIFIED") & ~df["image_id"].isin(conflict_ids).to_numpy()
    model_classes = [c for c in classes if c not in small and c != "UNSPECIFIED"]
    if small and "OTHER" not in model_classes:
        model_classes.append("OTHER")
    assert len(set(y[model_mask])) >= 2, "fewer than two classes left for the model-based tests"
    model_labels = {"min_class_groups": int(min_class_groups),
                    "merged_into_other": {c: int(groups_per_class[c]) for c in small},
                    "unspecified_images_left_out": int((df["label"] == "UNSPECIFIED").sum()),
                    "conflicting_duplicate_images_left_out": int(len(conflict_ids)),
                    "classes": model_classes}
    vals = np.column_stack([df[f].to_numpy(dtype=float) for f in spec.num_fields] +
                           [category_codes(df[f]) for f in spec.cat_fields])
    vals_cat = [False] * len(spec.num_fields) + [True] * len(spec.cat_fields)
    ind = df[miss].to_numpy(float)
    ind_cat = [False] * len(miss)
    acq = category_codes(df["acquisition"])[:, None]
    full = np.ones(len(df), bool)
    core_idx = [spec.fields.index(f) for f in spec.core_fields]
    sets = {
        "A_missing_indicators": (ind, ind_cat, full),
        "B_acquisition": (acq, [True], full),
        "D_acquisition_plus_indicators": (np.c_[acq, ind], [True] + ind_cat, full),
        "E_values_with_missing": (np.c_[vals, ind], vals_cat + ind_cat, full),
        "F_values_complete": (vals, vals_cat, df["complete"].to_numpy() == 1),
        "G_core_values_complete_core": (vals[:, core_idx], [vals_cat[i] for i in core_idx],
                                        df["complete_core"].to_numpy() == 1),
    }
    if "resolution_top" in df.columns:
        size = np.c_[category_codes(df["resolution_top"]), df[["megapixels", "aspect", "file_bytes"]].to_numpy(float)]
        sets["C_image_size"] = (size, [True, False, False, False], full)
    if stats_assoc is not None:
        border_cols = BORDER_STAT_COLUMNS + (["bytes_per_pixel"] if "bytes_per_pixel" in df.columns else [])
        sets["I_image_border_stats"] = (df[border_cols].to_numpy(float), [False] * len(border_cols), full)
    if set(spec.core_fields) == set(spec.fields):
        sets.pop("F_values_complete")  # identical to G when every field is a core field
    if df["acquisition"].nunique() < 2:
        for k in ["B_acquisition", "D_acquisition_plus_indicators"]:
            sets.pop(k)
    if extra_feature_sets:
        sets.update(extra_feature_sets)

    # Permutation units: lesions when the label is constant within each lesion and
    # every lesion sits inside one CV group; otherwise the CV groups themselves.
    # Both checks use the rows the model-based tests see (UNSPECIFIED and conflicting
    # duplicates left out), so a label conflict in a row that is never tested does not
    # switch the whole dataset to group-level units.
    units = g
    if "perm_unit" in df.columns:
        les = df["perm_unit"].to_numpy()
        dm_ = df[model_mask]
        nested = bool((dm_.groupby(dm_["perm_unit"])["group_id"].nunique() == 1).all())
        if nested and label_constant_within(les[model_mask], y[model_mask]):
            units = les
    perm_units_name = "lesion" if units is not g else spec.group_level

    # Acquisition and image size can be the same variable (e.g. when acquisition is a
    # resolution group); then the two shortcut tests are one finding, not two.
    redundant = []
    if "B_acquisition" in sets and "C_image_size" in sets:
        if cramers_v(df["acquisition"], df["resolution_top"])["v_corrected"] > 0.9:
            redundant = ["B_acquisition", "C_image_size"]

    # A label-dependent subsample (ds.cv_rule, e.g. SLICE-3D) is re-applied inside every
    # permutation, so the null goes through the same selection as the real labels; a mask
    # passed by hand (cv_rows_mask) has no rule and is treated as fixed rows.
    cv_rule = None
    if cv_rows_mask is None:
        cv_rows_mask = ds.cv_rows_mask
        cv_rule = ds.cv_rule
    elif ds.cv_rule is not None and ds.cv_rows_mask is not None and \
            np.array_equal(np.asarray(cv_rows_mask, bool), np.asarray(ds.cv_rows_mask, bool)):
        cv_rule = ds.cv_rule              # the dataset's own subsample passed by hand: keep its rule
    if cv_rows_mask is not None:
        cv_rows_mask = np.asarray(cv_rows_mask, dtype=bool)
        assert cv_rows_mask.shape == (len(df),), "cv_rows_mask must have one entry per row of ds.df"
    if cv_rule is not None:
        assert np.array_equal(np.asarray(cv_rule(y), bool), cv_rows_mask), \
            "ds.cv_rule(labels) must reproduce ds.cv_rows_mask"
    base_sets = {k: (X, cm, m & model_mask) for k, (X, cm, m) in sets.items()}
    sets = {k: (X, cm, m if cv_rows_mask is None else m & cv_rows_mask) for k, (X, cm, m) in base_sets.items()}

    def base_rule(base):
        """The subsample rule seen from the eligible rows of one test: labels of the other
        rows stay as they are."""
        def rule(y_base):
            y_all = y.copy()
            y_all[base] = y_base
            return np.asarray(cv_rule(y_all), bool)[base]
        return rule

    cv_rows = []
    for name in sorted(sets):
        X, cat_mask, mask = sets[name]
        print(f"shortcut test {name} on {int(mask.sum())} rows, {n_perm} + {n_perm_reference} permutations ...",
              flush=True)
        base = base_sets[name][2]
        resample = None if cv_rule is None else {"X": X[base], "y": y[base], "groups": g[base],
                                                 "units": units[base], "rule": base_rule(base)}
        cv_rows.append(run_feature_set(name, X[mask], cat_mask, y[mask], g[mask], n_perm=n_perm, seed=seed,
                                       perm_level=perm_level, units=units[mask],
                                       n_perm_reference=n_perm_reference,
                                       n_jobs=n_jobs, n_repeats=n_repeats, resample=resample))
    nulls = []
    for r in cv_rows:
        for kind, key in [("unit", "null_scores"), ("size-matched group", "null_group_scores")]:
            nulls += [{"feature_set": r["feature_set"], "null": kind, "bacc": v} for v in r.pop(key)]
    save(pd.DataFrame(nulls, columns=["feature_set", "null", "bacc"]), "permutation_nulls")
    cv = pd.DataFrame(cv_rows)
    # several feature sets are tested on one dataset: Holm adjustment of the main-null p-values
    # reference sets (image-derived) are not shortcut hypotheses and stay out of the family
    family = ~cv["feature_set"].str.endswith(REFERENCE_SUFFIX).to_numpy()
    p_holm = np.full(len(cv), np.nan)
    p_holm[family] = holm(cv.loc[family, "p_perm"])
    cv.insert(cv.columns.get_loc("p_perm") + 1, "p_perm_holm", p_holm)
    save(cv.set_index("feature_set"), "shortcut_cv")

    # which features carry the signal (A: missing indicators, E: values + indicators) and
    # which classes the missingness separates (confusion matrix of A)
    feature_names = {"A_missing_indicators": list(miss), "E_values_with_missing": list(spec.fields) + list(miss)}
    for name, names in feature_names.items():
        if name not in sets:
            continue
        X, cat_mask, mask = sets[name]
        imp, oof = explain_feature_set(X[mask], y[mask], g[mask], cat_mask, names, seed=seed)
        save(imp, f"importance_{name}")
        if name == "A_missing_indicators":
            conf = pd.crosstab(pd.Series(y[mask], name="true"), pd.Series(oof, name="predicted"))
            save(conf.reindex(index=model_classes, columns=model_classes, fill_value=0),
                 "confusion_A_missing_indicators")

    # --- missingness mechanism: what explains that a field is empty? ---------------------
    mech, mech_info = None, None
    if mechanism:
        from .mechanism import missingness_mechanism
        n_mech = n_perm if n_perm_mechanism is None else n_perm_mechanism
        print(f"missingness mechanism models, {n_mech} label draws per test ...", flush=True)
        mech_rows = model_mask if cv_rows_mask is None else (model_mask & cv_rows_mask)
        mech = save(missingness_mechanism(df.assign(label=y), spec, units, n_perm=n_mech, seed=seed, n_jobs=n_jobs,
                                          rows_mask=mech_rows, base_mask=model_mask,
                                          rule=None if cv_rule is None else base_rule(model_mask)),
                    "missingness_mechanism")
        tested = int(mech["crt_p"].notna().sum()) if "crt_p" in mech.columns else 0
        if n_mech == "auto":
            from .mechanism import auto_n_perm
            n_mech = auto_n_perm(tested)
        floor = tested / (n_mech + 1) if tested else None
        mech_info = {"rows_tested": tested, "n_perm": int(n_mech), "smallest_attainable_holm_p":
                     None if floor is None else round(min(1.0, floor), 4)}
        if floor is not None and floor > 0.05:
            print(f"WARNING: {tested} mechanism rows are Holm-adjusted with {n_mech} draws: the smallest attainable "
                  f"adjusted p is {floor:.3f} > 0.05, so no field can be called label-dependent; raise "
                  f"n_perm_mechanism to at least {int(np.ceil(tested / 0.05)) - 1}", flush=True)

    # --- complete-case bias: is the complete subset like the whole dataset? -------------
    smd = save(complete_case_smd(df, spec, n_boot=n_boot, seed=seed), "complete_case_smd")
    sub = {"all": df, "complete": df[df["complete"] == 1], "incomplete": df[df["complete"] == 0],
           "complete_core": df[df["complete_core"] == 1]}
    save(pd.DataFrame({k: v["label"].value_counts(normalize=True) for k, v in sub.items()})
         .reindex(classes).fillna(0).round(4), "complete_case_class_share")
    for f in spec.core_fields:
        if f in spec.num_fields:
            save(pd.DataFrame({k: v[f].describe() for k, v in sub.items()}).round(2), f"complete_case_{f}")
        else:
            save(pd.DataFrame({k: v[f].value_counts(normalize=True) for k, v in sub.items()}).fillna(0).round(4),
                 f"complete_case_{f}")

    # --- group size: how many images a patient / lesion has, and missingness by size -----
    df["group_size"] = df.groupby("group_id")["image_id"].transform("size")
    save(df.groupby("label")["group_size"].describe().reindex(classes).round(2), "group_size_by_class")
    size_bin = pd.cut(df["group_size"], [0, 1, 2, 5, 10, np.inf], labels=["1", "2", "3-5", "6-10", ">10"])
    gs = df.groupby(size_bin, observed=False)[miss].mean().T
    gs.index = list(spec.fields)
    gs.loc["images"] = df.groupby(size_bin, observed=False).size()
    save(gs.round(4), "missing_rate_by_group_size")

    # --- complete-case subsets --------------------------------------------
    cc = pd.concat({rule: _counts(df[df[rule] == 1], classes) for rule in ["complete", "complete_core"]}, axis=1)
    cc.insert(0, ("all", "images"), _counts(df, classes)["images"])
    save(cc, "complete_case_by_class")
    df[df["complete"] == 1][["image_id", "group_id", "label", "acquisition", *spec.fields]] \
        .sort_values("image_id").to_csv(os.path.join(tables_dir, "complete_case_images.csv"), index=False)

    # --- summary ----------------------------------------------------------
    cvi = cv.set_index("feature_set")
    def min_groups(rule):
        return int(df[df[rule] == 1].groupby("label")["group_id"].nunique().reindex(classes).fillna(0).min())

    def min_model_groups(rule):
        # the classes the model-based tests see (small ones merged, UNSPECIFIED left out)
        m = model_mask & (df[rule].to_numpy() == 1)
        return int(pd.Series(df["group_id"].to_numpy()[m]).groupby(y[m]).nunique()
                   .reindex(model_classes).fillna(0).min())

    def gain(k, key="gain"):
        if not n_perm or k not in cvi.index or pd.isna(cvi.loc[k, key]):
            return None
        return round(float(cvi.loc[k, key]), 4)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "dataset": {"name": spec.name, "title": spec.title, "source_url": spec.source_url,
                    "group_level": spec.group_level, "acquisition_name": spec.acquisition_name,
                    "fields": {"numeric": spec.num_fields, "categorical": spec.cat_fields, "core": spec.core_fields}},
        "license": vars(spec.license),
        "inputs": ds.inputs,
        "versions": {"skinaudit": __version__, "python": sys.version.split()[0], "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "structure": structure,
        "class_counts": tables["class_counts"]["images"].to_dict(),
        "harmonized_classes": harmonized,
        "acquisition_counts": df["acquisition"].value_counts().to_dict(),
        "acquisition_vs_label_v_corrected": acq_v["v_corrected"],
        "missing_rate": df[miss].mean().round(4).to_dict(),
        "missing_vs_label_v_corrected": miss_assoc.query("versus == 'label'").set_index("indicator")["v_corrected"].round(4).to_dict(),
        "missing_vs_acquisition_v_corrected": miss_assoc.query("versus == 'acquisition'").set_index("indicator")["v_corrected"].round(4).to_dict(),
        "missing_vs_other_fields_max_v_corrected": mar_matrix.max(axis=1).round(4).dropna().to_dict(),
        "missing_vs_other_fields_within_label_max_v_corrected":
            mar_lab_matrix.max(axis=1).round(4).dropna().to_dict(),
        "value_association": value_assoc.set_index("field")["effect"].round(4).to_dict(),
        "excluded_columns_v_corrected": leak.set_index("column")["v_corrected"].round(4).to_dict() if len(leak) else {},
        "image_properties": image_props,
        "shortcut_cv": {k: {c: (None if pd.isna(v) else v) for c, v in r.items() if c != "folds_bacc"}
                        for k, r in cvi.to_dict(orient="index").items()},
        "complete_case": {
            "complete_n": int(df["complete"].sum()), "complete_share": round(float(df["complete"].mean()), 4),
            "complete_core_n": int(df["complete_core"].sum()),
            "complete_core_share": round(float(df["complete_core"].mean()), 4),
            "complete_min_groups_per_class": min_groups("complete"),
            "complete_core_min_groups_per_class": min_groups("complete_core"),
        },
        "rubric": {
            "n_images": int(len(df)),
            "has_patient_id": "n_patients" in structure,
            "n_metadata_fields": len(spec.fields),
            "min_class_images": int(tables["class_counts"]["images"].min()),
            "min_class_groups_complete_core": min_model_groups("complete_core"),
            "max_missing_rate": round(float(df[miss].mean().max()), 4),
            "fields_missing_at_group_level": [f for f, r in structure["within_group"].items()
                                              if r["missingness_is_group_level"]],
            "max_missing_vs_other_fields_v": round(float(np.nanmax(mar_matrix.to_numpy(dtype=float))), 4)
                                             if mar_matrix.notna().any().any() else None,
            "max_missing_vs_other_fields_within_label_v":
                round(float(np.nanmax(mar_lab_matrix.to_numpy(dtype=float))), 4)
                if mar_lab_matrix.notna().any().any() else None,
            "redundant_shortcut_sets": redundant,
            # sensitivity analysis: gain over each null, their gap, and the standard error of
            # the gap (the observed score is shared, so only the two null means add noise)
            "permutation_sensitivity": {
                k: {"gain_main_null": round(float(r["bacc_mean"] - r["null_mean"]), 4),
                    "gain_group_null": round(float(r["bacc_mean"] - r["null_group_mean"]), 4),
                    "gap": round(float(r["null_group_mean"] - r["null_mean"]), 4),
                    "gap_se": round(float(np.sqrt(r["null_sd"] ** 2 / r["n_perm_unit"]
                                                  + r["null_group_sd"] ** 2 / r["n_perm_group"])), 4)
                              if r["n_perm_unit"] > 1 and r["n_perm_group"] > 1 else None,
                    "test_null": r["test_null"]}
                for k, r in cvi.iterrows() if r["n_perm_unit"] > 0 and not pd.isna(r["null_group_mean"])},
            "informative_missingness_bacc_gain": gain("A_missing_indicators"),
            "informative_missingness_p": float(cvi.loc["A_missing_indicators", "p_perm"]) if n_perm else None,
            "informative_missingness_p_holm": float(cvi.loc["A_missing_indicators", "p_perm_holm"]) if n_perm else None,
            "acquisition_shortcut_bacc_gain": gain("B_acquisition"),
            "image_size_shortcut_bacc_gain": gain("C_image_size"),
            "image_border_shortcut_bacc_gain": gain("I_image_border_stats"),
            "metadata_signal_bacc_gain": gain("G_core_values_complete_core"),
            # 95% interval of every gain (repeated CV, corrected variance, null-mean noise included)
            "bacc_gain_ci": {k: [gain(k, "gain_ci_low"), gain(k, "gain_ci_high")] for k in cvi.index},
            "complete_case_max_abs_smd": max_abs_smd(smd, "complete"),
            "complete_core_max_abs_smd": max_abs_smd(smd, "complete_core"),
            "label_dependent_missingness": _strongest_label_dependence(mech),
            "license": spec.license.name,
            "samples_can_be_shown": spec.license.allows_redistribution,
        },
        "missingness_mechanism": None if mech is None else
            mech.replace({np.nan: None}).to_dict(orient="records"),
        "missingness_mechanism_tests": mech_info,
        "model_labels": model_labels,
        "permutation": {"requested": perm_level, "units": perm_units_name,
                        "n_perm": n_perm, "n_perm_reference": n_perm_reference,
                        "used_per_set": cv.set_index("feature_set")["perm_level"].to_dict(),
                        "test_null_per_set": cv.set_index("feature_set")["test_null"].to_dict(),
                        "holm_family": sorted(cv.loc[family, "feature_set"]),
                        "sensitivity_skipped": {k: v for k, v in
                                                cv.set_index("feature_set")["sensitivity_skipped"].items()
                                                if isinstance(v, str)}},
        "runtime": {"n_jobs": n_jobs, "cpus_detected": available_cpus()},
        "duplicates": duplicates,
        "cv_subsample": None if cv_rows_mask is None else {
            "n_rows": int(cv_rows_mask.sum()), "share_of_rows": round(float(cv_rows_mask.mean()), 4),
            "class_counts": pd.Series(y[cv_rows_mask]).value_counts().to_dict(),
            "rule_reapplied": cv_rule is not None},
        "elapsed_seconds": None,
    }
    if make_figures:
        from .plots import make_all_figures
        summary["figures"] = make_all_figures(df, spec, tables, os.path.join(out_dir, "figures"),
                                              perm_units=perm_units_name)
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return summary
