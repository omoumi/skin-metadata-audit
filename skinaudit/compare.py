# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Cross-dataset comparison built only from per-dataset summary.json files."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .plots import INK_2, SERIES, _finish, _legend_below, _subtitle  # noqa: E402

RUBRIC_COLUMNS = {
    "n_images": "Images",
    "has_patient_id": "Patient id",
    "n_metadata_fields": "Metadata fields",
    "min_class_images": "Smallest class (images)",
    "min_class_groups_complete_core": "Smallest class, groups with core fields",
    "max_missing_rate": "Max missing rate",
    "informative_missingness_bacc_gain": "Missingness signal (BACC gain)",
    "acquisition_shortcut_bacc_gain": "Acquisition shortcut (BACC gain)",
    "image_size_shortcut_bacc_gain": "Image-size shortcut (BACC gain)",
    "image_border_shortcut_bacc_gain": "Image-border shortcut (BACC gain)",
    "metadata_signal_bacc_gain": "Core metadata signal (BACC gain)",
    "license": "License",
    "samples_can_be_shown": "Samples showable",
}


SUPPORTED_SCHEMAS = ("1.0",)


def load_summaries(paths: list[str], strict: bool = True) -> dict[str, dict]:
    """Read summary.json files. All must come from the same schema and skinaudit version:
    numbers from different versions are not comparable (the null or the statistics may have changed).
    strict=False turns the mismatch error into a printed warning."""
    out = {}
    for p in paths:
        with open(p) as f:
            s = json.load(f)
        assert s.get("schema_version") in SUPPORTED_SCHEMAS, f"{p}: unsupported summary schema"
        out[s["dataset"]["title"]] = s
    versions = {(s.get("schema_version"), s.get("versions", {}).get("skinaudit")) for s in out.values()}
    if len(versions) > 1:
        msg = "summaries come from different schema / skinaudit versions: " + ", ".join(
            f"{n}: {s.get('schema_version')} / {s.get('versions', {}).get('skinaudit')}" for n, s in out.items())
        if strict:
            raise ValueError(msg + " (re-run the older datasets, or pass strict=False)")
        print("WARNING:", msg)
    # datasets in alphabetical order of their titles, whatever order the files were found in
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


def _harmonized(s: dict) -> dict | None:
    return s.get("harmonized_classes")


def normalized_gain(entry: dict | None) -> float:
    """(BACC - null mean) / (1 - null mean): the share of the possible improvement over the
    permuted-label null that a feature set achieves. Like sklearn's adjusted balanced
    accuracy, but with the dataset's own null instead of 1/K, so the clustering of the data
    is part of the baseline. Puts binary and multi-class datasets on a common chance level;
    it does not equalise how hard the task is."""
    if not entry or entry.get("bacc_mean") is None or entry.get("null_mean") is None:
        return np.nan
    null = float(entry["null_mean"])
    return (float(entry["bacc_mean"]) - null) / (1.0 - null) if null < 1 else np.nan


def normalized_gain_ci(entry: dict | None) -> tuple[float, float]:
    """95% interval of the normalized gain: the gain interval divided by (1 - null mean)."""
    if not entry or entry.get("gain_ci_low") is None or entry.get("null_mean") is None:
        return np.nan, np.nan
    null = float(entry["null_mean"])
    if null >= 1:
        return np.nan, np.nan
    return float(entry["gain_ci_low"]) / (1.0 - null), float(entry["gain_ci_high"]) / (1.0 - null)


NORMALIZED_SETS = {"A_missing_indicators": "Missingness signal (normalized gain)",
                   "B_acquisition": "Acquisition shortcut (normalized gain)",
                   "C_image_size": "Image-size shortcut (normalized gain)",
                   "I_image_border_stats": "Image-border shortcut (normalized gain)",
                   "G_core_values_complete_core": "Core metadata signal (normalized gain)"}


def gain_reference(s: dict) -> str:
    """Which null the gains of a dataset are measured from: the chance-level unit null, the
    size-matched group null (labels differ within units, so no unit null exists), or a unit
    null that re-applies a label-dependent subsample (its mean can lie above 1/K)."""
    used = set((s.get("permutation") or {}).get("used_per_set", {}).values())
    if "group" in used:
        return "size-matched group null (conservative)"
    if (s.get("cv_subsample") or {}).get("rule_reapplied"):
        return "unit null with the subsample re-applied"
    return "unit null"


def _marked(name: str, s: dict) -> str:
    """Dataset name with a dagger when its gains are not measured from the plain unit null."""
    return name + (" \u2020" if gain_reference(s) != "unit null" else "")


GAIN_REFERENCE_NOTE = ("\u2020 = gains measured from another null than the plain unit (lesion or image) null: the "
                       "size-matched group null where labels differ within units (conservative), or a unit null "
                       "that re-applies a label-dependent subsample; see 'Gain reference' in rubric.csv.")


def rubric_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """Rubric per dataset; class-size columns use the shared taxonomy so they are comparable."""
    from .taxonomy import SHARED_LEVEL2
    rows = {}
    for name, s in summaries.items():
        row = {RUBRIC_COLUMNS[k]: s["rubric"].get(k) for k in RUBRIC_COLUMNS}
        row["Gain reference"] = gain_reference(s)
        # raw gains stay (they are what each dataset README reports); normalized ones sit next to them
        for k, col in NORMALIZED_SETS.items():
            v = normalized_gain(s["shortcut_cv"].get(k))
            row[col] = None if np.isnan(v) else round(v, 4)
            lo, hi = normalized_gain_ci(s["shortcut_cv"].get(k))
            row[col.replace(")", ", 95% CI)")] = None if np.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"
        r = s["rubric"]
        row["Max |SMD|, all-complete vs all"] = (r.get("complete_case_max_abs_smd") or {}).get("max_abs_smd")
        row["Max |SMD|, core-complete vs all"] = (r.get("complete_core_max_abs_smd") or {}).get("max_abs_smd")
        dep = r.get("label_dependent_missingness") or {}
        row["Label-dependent missingness: field"] = dep.get("fields")
        row["Label-dependent missingness: AUC gain"] = dep.get("auc_gain")
        row["Label-dependent missingness: CRT p (Holm)"] = dep.get("crt_p_holm")
        row["Label-dependent missingness: group-swap p (Holm)"] = dep.get("group_swap_p_holm")
        row["Label-dependent missingness: supported by both tests"] = dep.get("supported")
        row["Label-dependent missingness: label R2"] = dep.get("label_r2")
        h = _harmonized(s)
        if h:
            groups = {c: h["level2"]["groups"].get(c, 0) for c in h["shared_classes_present"]}
            row["Shared classes present"] = f"{len(h['shared_classes_present'])}/{len(SHARED_LEVEL2)}"
            row["Smallest shared class (groups)"] = min(groups.values()) if groups else None
            row["Malignant share"] = round(h["level1"]["images"]["malignant"] / s["structure"]["n_images"], 4)
            row["Ambiguous label share"] = h["ambiguous_share"]
            row["Unspecified diagnosis share"] = h["unspecified_share"]
        rows[name] = row
    return pd.DataFrame(rows).T


def class_share_table(summaries: dict[str, dict], level: str = "level2") -> pd.DataFrame:
    """Share of images per class. Uses the shared taxonomy when every dataset has it;
    otherwise falls back to native labels (which are not comparable across schemes)."""
    if all(_harmonized(s) for s in summaries.values()):
        t = pd.DataFrame({name: _harmonized(s)[level]["images"] for name, s in summaries.items()}).T
        t = t.loc[:, t.sum(axis=0) > 0]
    else:
        print("WARNING: some summaries lack harmonized_classes; class shares use native labels")
        t = pd.DataFrame({name: s["class_counts"] for name, s in summaries.items()}).T
    return t.div(t.sum(axis=1), axis=0)


def shortcut_comparison_plot(summaries: dict[str, dict], path: str,
                             sets=("A_missing_indicators", "B_acquisition", "C_image_size",
                                   "I_image_border_stats", "G_core_values_complete_core")) -> str:
    names = list(summaries)
    labels = {"A_missing_indicators": "Missing indicators", "B_acquisition": "Acquisition",
              "C_image_size": "Image size", "I_image_border_stats": "Image border stats",
              "G_core_values_complete_core": "Core metadata values"}
    y = np.arange(len(names))
    h = 0.8 / len(sets)
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(names) * len(sets) * 0.35 + 1.8))
    for i, k in enumerate(sets):
        gains = np.array([normalized_gain(summaries[n]["shortcut_cv"].get(k)) for n in names])
        cis = np.array([normalized_gain_ci(summaries[n]["shortcut_cv"].get(k)) for n in names])
        pos = y + (i - (len(sets) - 1) / 2) * h
        ax.barh(pos, gains, height=h - 0.03, color=SERIES[i], label=labels.get(k, k))
        ok = ~np.isnan(cis[:, 0]) & ~np.isnan(gains)
        if ok.any():
            ax.errorbar(gains[ok], pos[ok], xerr=[gains[ok] - cis[ok, 0], cis[ok, 1] - gains[ok]], fmt="none",
                        ecolor=INK_2, elinewidth=0.9, capsize=0, zorder=3)
        for p, g, c in zip(pos, gains, cis):
            if not np.isnan(g):
                right = c[1] if not np.isnan(c[1]) else g
                ax.text(max(right, 0), p, f" {g:.3f}", va="center", fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_yticks(y, [_marked(n, summaries[n]) for n in names])
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Normalized gain: (BACC - null) / (1 - null)")
    ax.set_title("What predicts the label without the image, per dataset", pad=22)
    _subtitle(ax, "Share of the possible improvement over the permuted-label null (line = 95% interval); "
                  "for acquisition and image size a high value suggests a possible shortcut")
    _legend_below(fig, *ax.get_legend_handles_labels(), ncol=min(3, len(sets)))
    note = ("Normalization puts binary and multi-class datasets on a common chance level, but not on a common "
            "difficulty; raw gains are in rubric.csv.")
    if any(gain_reference(s) != "unit null" for s in summaries.values()):
        note += " " + GAIN_REFERENCE_NOTE
    return _finish(fig, path, note=note)


CANONICAL_FIELDS = {
    "Age": ["age_approx", "age"],
    "Sex": ["sex", "gender"],
    "Anatomic site": ["anatom_site_1", "anatom_site_general", "anatom_site_general_challenge", "site", "region"],
    "Skin type": ["fitzpatrick_skin_type", "fitspatrick", "skin_tone_class"],
    "Lesion size": ["clin_size_long_diam_mm", "diameter_1"],
    "Personal history of melanoma": ["personal_hx_mm"],
    "Family history of melanoma": ["family_hx_mm"],
}


def field_availability_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """Missing rate per dataset and canonical field; NaN = the dataset does not record the field.
    When several native columns map to one canonical field, the most complete one is used."""
    rows = {}
    for name, s in summaries.items():
        rates = {k[len("miss_"):]: v for k, v in s.get("missing_rate", {}).items()}
        row = {}
        for canon, natives in CANONICAL_FIELDS.items():
            vals = [rates[n] for n in natives if n in rates]
            row[canon] = min(vals) if vals else np.nan
        rows[name] = row
    return pd.DataFrame(rows).T[list(CANONICAL_FIELDS)]


def field_availability_plot(table: pd.DataFrame, path: str) -> str:
    from matplotlib.colors import ListedColormap
    from .plots import SEQ, INK
    fig, ax = plt.subplots(figsize=(0.95 * table.shape[1] + 3.4, 0.5 * table.shape[0] + 2.2))
    vals = table.to_numpy(dtype=float)
    ax.imshow(np.isnan(vals).astype(float), cmap=ListedColormap(["#ffffff", "#efeee9"]), vmin=0, vmax=1,
              aspect="auto")
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            txt, col = ("not recorded", INK_2) if np.isnan(v) else (f"{v:.0%}", "white" if v > 0.55 else INK)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5 if np.isnan(v) else 9, color=col)
    ax.set_xticks(range(vals.shape[1]), table.columns, rotation=30, ha="right")
    ax.set_yticks(range(vals.shape[0]), table.index)
    ax.set_title("Which metadata each dataset records, and how often it is missing", loc="left", pad=22)
    _subtitle(ax, "Cell = missing rate among images; grey = field not in the dataset")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Missing rate")
    return _finish(fig, path)


def dataset_map_plot(summaries: dict[str, dict], path: str) -> str:
    """x = core-metadata signal, y = missingness signal (both normalized gain over the null);
    point area = number of CV groups."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs, ys, key = [], [], []
    for i, (name, s) in enumerate(summaries.items()):
        cv = s["shortcut_cv"]
        x = normalized_gain(cv.get("G_core_values_complete_core", {}))
        y = normalized_gain(cv.get("A_missing_indicators", {}))
        if np.isnan(x) or np.isnan(y):
            continue
        n = s["structure"]["n_groups"]
        num = len(key) + 1
        # points carry a number and the names go in a key beside the axes, so long names or
        # datasets that sit close together never overlap
        ax.scatter(x, y, s=60 + 260 * np.sqrt(n / 1000), color=SERIES[i % len(SERIES)], alpha=0.85,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(str(num), (x, y), ha="center", va="center", fontsize=8, color="white", fontweight="bold",
                    zorder=4)
        key.append(f"{num}  {_marked(name, s)} ({n:,} groups)")
        xs.append(x)
        ys.append(y)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    if xs:
        for lim, vals in [(ax.set_xlim, xs), (ax.set_ylim, ys)]:
            lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
            pad = 0.12 * (hi - lo) + 0.02
            lim(lo - pad, hi + pad)
        import textwrap
        ax.text(1.03, 1.0, "\n".join(textwrap.fill(k, 40, subsequent_indent="    ") for k in key),
                transform=ax.transAxes, va="top", ha="left", fontsize=8, color=INK_2)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Core metadata signal (normalized gain, complete-core rows)")
    ax.set_ylabel("Missingness signal (normalized gain, missing indicators only)")
    ax.set_title("Where each dataset sits: metadata value vs informative missingness", loc="left", pad=22)
    _subtitle(ax, "Up = missingness predicts the label; right = metadata values predict the label; area = CV groups")
    note = ("Gains are normalized to the null, which aligns the chance level across label schemes; how far a "
            "dataset can get without images still differs, so read distances between binary and multi-class "
            "datasets as approximate.")
    if any(gain_reference(s) != "unit null" for s in summaries.values()):
        note += " " + GAIN_REFERENCE_NOTE
    return _finish(fig, path, note=note)


def build_comparison(summary_paths: list[str], out_dir: str, strict: bool = True) -> dict:
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    s = load_summaries(summary_paths, strict=strict)
    rubric = rubric_table(s)
    rubric.to_csv(os.path.join(out_dir, "rubric.csv"))
    shares = class_share_table(s)
    shares.to_csv(os.path.join(out_dir, "class_share.csv"))
    from .plots import heatmap
    from .taxonomy import mapping_table
    mapping_table().to_csv(os.path.join(out_dir, "taxonomy_mapping.csv"), index=False)
    figs = [heatmap(shares, os.path.join(out_dir, "figures", "class_share.png"), "Class mix per dataset",
                    "Share of each dataset's images per class of the shared taxonomy (see taxonomy_mapping.csv)",
                    vmax=float(shares.max().max()), cbar_label="Share of images", tiny_text="<1%")]
    if all(_harmonized(x) for x in s.values()):
        l1 = class_share_table(s, "level1")
        l1.to_csv(os.path.join(out_dir, "class_share_level1.csv"))
        figs.append(heatmap(l1, os.path.join(out_dir, "figures", "class_share_level1.png"),
                            "Malignant / benign / indeterminate per dataset",
                            "Level 1 of the shared taxonomy; actinic keratosis counts as indeterminate everywhere",
                            vmax=1.0, cbar_label="Share of images", tiny_text="<1%"))
    avail = field_availability_table(s)
    avail.to_csv(os.path.join(out_dir, "field_availability.csv"))
    figs += [shortcut_comparison_plot(s, os.path.join(out_dir, "figures", "shortcut_gain.png")),
             field_availability_plot(avail, os.path.join(out_dir, "figures", "field_availability.png")),
             dataset_map_plot(s, os.path.join(out_dir, "figures", "dataset_map.png"))]
    return {"rubric": rubric, "class_share": shares, "field_availability": avail, "figures": figs}
