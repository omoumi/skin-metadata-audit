# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""README.md generator for a per-dataset repository, built from summary.json."""
from __future__ import annotations

import json
import os

FIGURE_TEXT = {
    "class_counts.png": "Images per class, and how many remain when rows with missing metadata are dropped.",
    "missing_by_class.png": "Missing rate of each metadata field within each class.",
    "missing_patterns.png": "Which fields are missing together (UpSet-style).",
    "association.png": "Association of each field's missingness with the label and with the acquisition "
                       "variable, next to the association of the observed value with the label.",
    "acquisition_by_class.png": "Share of each class coming from each acquisition group. A class that comes "
                                "from one group only could be recognised by acquisition rather than by the lesion.",
    "missing_vs_other_fields.png": "Association between the missingness of each field and the observed values of "
                                   "the other fields. A value near 0 suggests little association with that field (as measured "
                                   "on these categories and quintiles); MCAR also needs no dependence on the label (see "
                                   "the association figure). A clearly positive value is evidence against MCAR; observed "
                                   "data cannot tell MAR from MNAR.",
    "missing_vs_other_fields_within_label.png": "The same check inside each class. A cell that is strong in the "
                                                "previous figure but weak here may run through the label (for "
                                                "example, age differs by class and missingness differs by class).",
    "resolution_by_class.png": "Image resolution by class. Resolution can leave traces (resampling, compression) "
                              "that a CNN may pick up even after resizing, so it is a possible shortcut.",
    "image_stats.png": "Colour, border, frame and compression statistics of the images by class. Border and file "
                       "statistics mostly come from the camera and the frame, but a large lesion can reach the "
                       "border and file size depends on texture, so the border test may overstate the pure "
                       "camera / frame shortcut.",
    "shortcut_cv.png": "Balanced accuracy of a gradient-boosting model that sees only the listed features, "
                       "compared with the same pipeline on permuted labels.",
    "complete_case_bias.png": "Class mix and field values in all images, in rows where every field is present, and "
                              "in rows with some field missing. Experiments that start from complete rows and then "
                              "remove metadata inherit any difference shown here.",
    "group_size.png": "Images per CV group (patient, lesion or image) by class, and the missing rate of each field by group size. "
                      "When both differ, missingness could predict the label through group size; the gap between "
                      "the two nulls in the table above gives an indication of that path.",
    "feature_importance.png": "Permutation importance on held-out folds: the drop in balanced accuracy when one "
                              "feature is shuffled. Left: the missing-indicator model; right: values plus "
                              "indicators. Correlated features share credit, so a low bar does not mean no signal.",
    "missingness_confusion.png": "Out-of-fold predictions of the model that sees only which fields are missing. "
                                 "A bright off-diagonal cell means two classes have the same missingness pattern; "
                                 "a bright diagonal cell means much of that class is recognised from missingness alone.",
    "categorical_by_class.png": "Distribution of each categorical field within each class (observed values only; "
                                "the 7 most common categories, the rest grouped as other).",
    "missingness_mechanism.png": None,   # filled from skinaudit.mechanism.FIGURE_TEXT below
    "complete_case_smd.png": "Standardized mean difference between the rows used for missingness-injection "
                             "experiments (all fields complete, core fields complete) and all images, and the other "
                             "images. Categorical fields, the class mix and acquisition use the multi-category SMD "
                             "(Yang and Dalton 2012). |SMD| > 0.1 is used as the mark of imbalance, a common "
                             "convention (Austin 2009) rather than a statistical test. Fields are compared on "
                             "observed values only.",
    "null_distributions.png": "Balanced accuracy of the same pipeline on permuted labels, for both permutation "
                              "schemes, with the real-label result as a vertical line. With few permutations the "
                              "histograms are coarse.",
}


from .mechanism import FIGURE_TEXT as _MECHANISM_TEXT  # noqa: E402
FIGURE_TEXT["missingness_mechanism.png"] = _MECHANISM_TEXT

# headings that the file name does not give well
FIGURE_TITLE = {"shortcut_cv.png": "Shortcut CV", "complete_case_bias.png": "Complete-case bias",
                "complete_case_smd.png": "Complete-case SMD", "missingness_confusion.png": "Missingness confusion"}


def _swap_text(gw: str) -> str:
    """The group swap of the missingness tests, in the words of the CV group."""
    if gw == "image":
        return "images swapping labels within strata of acquisition x age band"
    return f"whole {gw}s swapping labels within strata of images per {gw} x acquisition x age band"


def _figure_text(fig: str, gw: str) -> tuple[str, str]:
    """Heading and caption of one figure."""
    name = fig.replace(".png", "")
    cap = FIGURE_TEXT.get(fig)
    if fig == "missingness_mechanism.png":
        cap = cap.replace("whole patients swapping labels within strata of images per patient x acquisition x age band",
                          _swap_text(gw))
    if cap is None and name.endswith("_by_class"):              # one figure per numeric field
        field = name[:-len("_by_class")]
        return f"`{field}` by class", f"Distribution of `{field}` within each class."
    return FIGURE_TITLE.get(fig, name.replace("_", " ").capitalize()), cap or name.replace("_", " ") + "."

# full references for the methods cited (author year) in the generated README
REFERENCES = [
    "Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups "
    "in propensity-score matched samples. Stat Med. 2009;28(25):3083-3107. doi:10.1002/sim.3697",
    "Bergsma W. A bias-correction for Cramér's V and Tschuprow's T. J Korean Stat Soc. 2013;42(3):323-328. "
    "doi:10.1016/j.jkss.2012.10.002",
    "Bouckaert RR, Frank E. Evaluating the replicability of significance tests for comparing learning algorithms. "
    "In: Advances in Knowledge Discovery and Data Mining (PAKDD 2004). Lecture Notes in Computer Science, vol 3056. "
    "Springer; 2004:3-12. doi:10.1007/978-3-540-24775-3_3",
    "Candès E, Fan Y, Janson L, Lv J. Panning for gold: 'model-X' knockoffs for high dimensional controlled "
    "variable selection. J R Stat Soc Series B Stat Methodol. 2018;80(3):551-577. doi:10.1111/rssb.12265",
    "Holm S. A simple sequentially rejective multiple test procedure. Scand J Stat. 1979;6(2):65-70.",
    "Nadeau C, Bengio Y. Inference for the generalization error. Mach Learn. 2003;52(3):239-281. "
    "doi:10.1023/A:1024068626366",
    "Phipson B, Smyth GK. Permutation p-values should never be zero: calculating exact p-values when permutations "
    "are randomly drawn. Stat Appl Genet Mol Biol. 2010;9(1):Article 39. doi:10.2202/1544-6115.1585",
    "Tomczak M, Tomczak E. The need to report effect size estimates revisited. An overview of some recommended "
    "measures of effect size. Trends Sport Sci. 2014;1(21):19-25.",
    "Yang D, Dalton JE. A unified approach to measuring the effect size between two groups using SAS. "
    "SAS Global Forum 2012; paper 335-2012.",
]


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _gain(r: dict, key: str, set_name: str) -> str:
    """Gain with its 95% interval, e.g. '0.195 [0.170, 0.221]'."""
    ci = (r.get("bacc_gain_ci") or {}).get(set_name)
    txt = _fmt(r.get(key))
    if ci and None not in ci:
        txt += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return txt


def write_readme(out_dir: str, gallery: dict | None = None, reproduce: str = "", notes: str = "") -> str:
    with open(os.path.join(out_dir, "summary.json")) as f:
        s = json.load(f)
    d, lic, r, st = s["dataset"], s["license"], s["rubric"], s["structure"]
    gw = str(d["group_level"]).split(" ")[0]   # one word for the CV group in running text
    perm_units = s.get("permutation", {}).get("units", gw)
    redundant = r.get("redundant_shortcut_sets", [])
    skipped = s.get("permutation", {}).get("sensitivity_skipped", {})
    fallback = sorted(k for k, v in s.get("permutation", {}).get("used_per_set", {}).items() if v != "unit")
    lines = [f"# {d['title']}: metadata audit", "",
             f"Audit of the metadata that comes with the {d['title']} skin-lesion images: how complete it is, "
             "whether missingness carries label information, and which non-image shortcuts exist. "
             "No image model is trained here.", ""]
    if gallery:
        lines += [f"![Example images]({gallery['grid']})", "",
                  f"Sample images are shown under {lic['name']}; see [samples/ATTRIBUTION.md]({gallery['attribution']}).", ""]
    lines += ["## Key numbers", "", "| Item | Value |", "|---|---|",
              f"| Images | {_fmt(st['n_images'])} |",
              f"| Groups used for CV (group level: {d['group_level']}) | {_fmt(st['n_groups'])} |"]
    if "n_patients" in st:
        lines.append(f"| Patients | {_fmt(st['n_patients'])} |")
    if "n_lesions" in st:
        lines.append(f"| Lesions | {_fmt(st['n_lesions'])} |")
    cc = s["complete_case"]
    lines += [f"| Metadata fields audited | {', '.join(d['fields']['numeric'] + d['fields']['categorical'])} |",
              f"| Rows with every field present | {_fmt(cc['complete_n'])} ({cc['complete_share']:.1%}) |",
              f"| Rows with core fields present ({', '.join(d['fields']['core'])}) | "
              f"{_fmt(cc['complete_core_n'])} ({cc['complete_core_share']:.1%}) |",
              f"| Smallest class in the model-based tests, {gw}s with core fields | "
              f"{_fmt(r['min_class_groups_complete_core'])} |",
              f"| Highest missing rate of any field | {r['max_missing_rate']:.1%} |",
              f"| Balanced-accuracy gain over null, missing indicators only | {_gain(r, 'informative_missingness_bacc_gain', 'A_missing_indicators')} |",
              f"| Balanced-accuracy gain over null, acquisition only | {_gain(r, 'acquisition_shortcut_bacc_gain', 'B_acquisition')} |",
              f"| Balanced-accuracy gain over null, image size only | {_gain(r, 'image_size_shortcut_bacc_gain', 'C_image_size')} |",
              f"| Balanced-accuracy gain over null, image border / file statistics | {_gain(r, 'image_border_shortcut_bacc_gain', 'I_image_border_stats')} |",
              f"| Balanced-accuracy gain over null, core metadata values | {_gain(r, 'metadata_signal_bacc_gain', 'G_core_values_complete_core')} |",
              f"| License | {lic['name']} |", "",
              _gain_reference_text(s, perm_units, gw, fallback), ""]
    sens = r.get("permutation_sensitivity") or {}
    if sens:
        lines += ["### Sensitivity to the permutation scheme", "",
                  f"Balanced-accuracy gain over the two nulls. {perm_units.capitalize()} null: one "
                  f"label per {perm_units} is shuffled. Size-matched null: whole {gw}s swap labels with "
                  f"{gw}s of the same size, which keeps the link between {gw} size and "
                  f"label. Gap = difference of the two null means, with its standard error. A gap well above "
                  f"about 2 SE suggests that part of the signal runs through how many images a {gw} has.", "",
                  f"| Feature set | Gain, {perm_units} null | Gain, size-matched {gw} null | Gap | SE of gap "
                  "| Null used for the test |",
                  "|---|---|---|---|---|---|"]
        for k in sorted(sens):
            v = sens[k]
            test = {"group": "size-matched", "unit": perm_units}.get(v.get("test_null"), v.get("test_null") or "")
            lines.append(f"| {k} | {v['gain_main_null']:.3f} | {v['gain_group_null']:.3f} | {v['gap']:.3f} | "
                         f"{_fmt(v['gap_se'])} | {test} |")
        lines += [""]
    lines += ["## Figures", ""]
    for fig in s.get("figures", []):
        title, cap = _figure_text(fig, gw)
        lines += [f"### {title}", "", cap, "",
                  f"![{fig}](figures/{fig})", ""]
    lines += ["## Method", "",
              "- Missing-like tokens (`unknown`, `na`, empty strings, ...) are converted to missing before counting.",
              "- Associations use bias-corrected Cramer's V (Bergsma 2013); 95% intervals come from a bootstrap that "
              f"resamples whole {gw}s. Numeric fields use the Kruskal-Wallis epsilon squared, H / (n - 1) "
              "(Tomczak and Tomczak 2014). The chi-square and Kruskal-Wallis p-values in the tables are labelled "
              "`p_naive`: they treat images as independent, "
              + (f"while images of one {gw} are not, so they come out too small" if gw != "image" else
                 "while images of one patient may not be (patients cannot be identified here), so they may come out "
                 "too small")
              + f"; the effect sizes and their {'group-' if gw != 'image' else ''}bootstrap intervals are the "
              "quantities to read.",
              f"- Shortcut tests: `HistGradientBoostingClassifier` (balanced class weights), {gw}-grouped "
              "stratified 5-fold CV, balanced accuracy. Nulls are the same pipeline on permuted labels. The "
              + (f"{perm_units} null permutes the image labels" if perm_units == "image" else
                 f"{perm_units} null gives one label per {perm_units} to all of its images")
              + "; its mean is the chance level every gain is measured from. "
              + (f"When several {perm_units}s sit inside one "
                 f"{gw}, the {perm_units} null also breaks the similarity of labels within a "
                 f"{gw}, so its spread can be too narrow for a test (in simulations about 14% false positives at the "
                 f"5% level); the p-values then come from the size-matched null, where whole {gw}s swap labels "
                 f"with {gw}s of the same size (valid when {gw}s of the same size are exchangeable under the null; "
                 "conservative when a feature works through group size). " if perm_units != gw else "")
              + "The null used is in `test_null`. p-values are "
              "(1 + #null >= observed) / (1 + #permutations) (Phipson and Smyth 2010), computed from the first CV "
              "split (the nulls use single splits too), and Holm-adjusted (Holm 1979) over the feature sets of this "
              "dataset, "
              "image-derived reference sets excluded (`p_perm_holm`). With "
              f"{s.get('permutation', {}).get('n_perm', 'n')} permutations of the test null the smallest "
              f"attainable p is {1 / (1 + (s.get('permutation', {}).get('n_perm') or 0)):.3g}.",
              f"- The reported balanced accuracy is the mean of "
              f"{_n_repeats(s)} grouped CV runs with different fold assignments. Its 95% interval, and the "
              "interval of the gain over the null, use the corrected repeated-CV variance "
              "(1/J + n_test/n_train) * S^2 over the J fold scores (Nadeau and Bengio 2003; Bouckaert and Frank "
              "2004) with a t quantile; the gain interval also includes the noise of the null mean. This "
              "correction is approximate; its coverage was close to 95% in the toolkit's simulations.",
              "- The image-size feature set includes the file size in bytes, which also depends on the image "
              "content (texture, compression), so part of its signal may come from the lesion itself rather than "
              "from the camera.",
              "- One model class with fixed settings is used for the shortcut tests, so a feature set without a "
              "significant gain may still carry signal that another model would find; a significant gain shows "
              "an association with the label, not its cause (e.g. an acquisition source may differ from the "
              "others in its patient population, not only in its images).",
              *([f"- Within-{gw} checks use only {gw}s with several images: conflicting "
                 f"values, and fields missing for some images of a {gw} but present for others (if never, "
                 f"missingness is decided per {gw}). Age can legitimately differ between visits."]
                if gw != "image" else []),
              *([f"- No sensitivity null was run for {', '.join(sorted(skipped))}: every {gw} "
                 "is one permutation unit and all have the same number of images, so the size-matched null would "
                 f"be the same random process as the {perm_units} null."] if skipped else []),
              *([f"- Model-based tests (shortcut CV, feature importance, missingness mechanism): classes with "
                 f"fewer than {s['model_labels']['min_class_groups']} {gw}s are merged into OTHER ("
                 + ", ".join(f"{c}: {n}" for c, n in s['model_labels']['merged_into_other'].items()) + ")"
                 + (f"; {s['model_labels']['unspecified_images_left_out']:,} UNSPECIFIED images (only benign / "
                    "malignant known) are left out" if s['model_labels']['unspecified_images_left_out'] else "")
                 + ". Descriptive tables keep every class."]
                if s.get("model_labels") and (s["model_labels"]["merged_into_other"]
                                              or s["model_labels"]["unspecified_images_left_out"]) else []),
              *([f"- {s['duplicates']['pairs_with_conflicting_labels']} confirmed duplicate pair(s) carry different "
                 f"labels on the two copies. Which label is right is unknown, so their "
                 f"{len(s['duplicates']['conflicting_label_images_left_out_of_model_tests'])} images are left out of "
                 "the model-based tests and their permutation units are not merged; descriptive tables keep them."]
                if (s.get("duplicates") or {}).get("pairs_with_conflicting_labels") else []),
              *([f"- Shortcut tests use a subsample of {s['cv_subsample']['n_rows']:,} rows "
                 f"({s['cv_subsample']['share_of_rows']:.1%}); every other table uses all rows. Rule: "
                 f"{s.get('inputs', {}).get('cv_subsample', {}).get('rule', 'see inputs')}."
                 + (" Because the rule depends on the label, every permutation is applied to all eligible rows "
                    "and the rule is applied again to the permuted labels (shortcut nulls and the group swap of "
                    "the missingness tests), so the nulls go through the same selection as the real labels."
                    if s["cv_subsample"].get("rule_reapplied") else "")]
                if s.get("cv_subsample") else []),
              *([f"- Missingness-mechanism p-values are Holm-adjusted over "
                 f"{s['missingness_mechanism_tests']['rows_tested']} tested rows with "
                 f"{s['missingness_mechanism_tests']['n_perm']} draws, so the smallest attainable adjusted p is "
                 f"{s['missingness_mechanism_tests']['smallest_attainable_holm_p']}"
                 + ("; above 0.05, no field can be called label-dependent in this run."
                    if (s['missingness_mechanism_tests']['smallest_attainable_holm_p'] or 0) > 0.05 else ".")]
                if (s.get("missingness_mechanism_tests") or {}).get("rows_tested") else []),
              *(["- Missingness mechanism: for each field, gradient boosting predicts that the field is missing "
                 "from the other fields, the acquisition variable and the group size, without and with the label. "
                 "The label's contribution is tested with a conditional randomization test (Candès et al. 2018), "
                 "where labels are redrawn from a model of the label given the other inputs, and with a test where "
                 + ("labels are swapped between images" if gw == "image" else f"whole {gw}s swap labels")
                 + " within strata. A field's missingness is called label-dependent only "
                 "when both tests reject after Holm adjustment. In the toolkit's simulations this rule claimed "
                 "dependence in about 4% of runs without any, and in about 7% when the missingness depended on a "
                 "label-related field; its power was limited, so a field that is not called "
                 "label-dependent may still be (validation/README.md)."]
                if s.get("missingness_mechanism") else []),
              *([f"- Labels were not constant within {perm_units}s for {', '.join(fallback)}; there the null swaps "
                 "label vectors between groups of equal size instead."] if fallback else []),
              *([f"- {' and '.join(redundant)} measure the same variable here (Cramer's V > 0.9), so they count as "
                 "one shortcut."] if redundant else []),
              *([f"- {s['duplicates']['confirmed_pairs']} image pairs were confirmed as duplicates "
                 f"(see tables/confirmed_duplicates.csv); the {gw}s they link were merged before "
                 f"cross-validation ({s['duplicates']['groups_before']:,} -> {s['duplicates']['groups_after']:,} "
                 "groups), so a duplicate cannot sit in a training and a test fold at the same time."]
                if (s.get("duplicates") or {}).get("action", "merge") == "merge" and s.get("duplicates") else []),
              *([f"- {s['duplicates']['confirmed_pairs']} image pairs were confirmed as copies of the same photo "
                 f"(see tables/confirmed_duplicates.csv), in {s['duplicates']['clusters']:,} clusters. All copies but "
                 "one were removed before any table was computed (the one kept has the fewest missing metadata "
                 "fields, then a patient id): "
                 f"{s['duplicates']['images_before']:,} -> {s['duplicates']['images_after']:,} images "
                 "(tables/dropped_duplicates.csv)"
                 + (f"; {s['duplicates']['clusters_with_conflicting_labels']} cluster(s) whose copies carry different "
                    "labels were removed entirely." if s['duplicates']['clusters_with_conflicting_labels'] else ".")]
                if (s.get("duplicates") or {}).get("action") == "drop" else []),
              "- All tables are in [`tables/`](tables) and every number above is in [`summary.json`](summary.json).", ""]
    lines += ["## References", ""] + [f"- {c}" for c in REFERENCES] + [""]
    if notes:
        lines += ["## Notes", "", notes, ""]
    if reproduce:
        lines += ["## Notebook", "", reproduce, ""]
    lines += ["## Data source and license", "", f"- Source: {d['source_url']}",
              f"- License of the images and metadata: [{lic['name']}]({lic['url']})"
              + (" - non-commercial use only." if lic["non_commercial"] else "."),
              "- This repository does not redistribute the dataset; only aggregate tables, image ids and a few "
              "resized sample images with attribution.", ""]
    if lic.get("citations"):
        lines += ["Dataset reference" + ("s:" if len(lic["citations"]) > 1 else ":"), ""] \
            + [f"- {c}" for c in lic["citations"]] + [""]
    content = content_license(lic)
    lines += ["## License of this repository", "",
              f"- Tables, figures and text: {content['name']}"
              + (f" ([license text]({content['url']}))" if content["url"] else "") + f". {content['reason']}",
              "- Sample images keep the license of the original image (see samples/ATTRIBUTION.md).",
              f"- Code that produced these files: [skin-metadata-audit]({TOOLKIT_URL}), GPL-3.0-only.",
              "- Details: [LICENSE-CONTENT.md](LICENSE-CONTENT.md).", ""]
    lines += ["Software versions: " + ", ".join(f"{k} {v}" for k, v in s["versions"].items()), ""]
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    write_content_license(out_dir, d["title"], lic)
    return path


def _gain_reference_text(s: dict, perm_units: str, gw: str, fallback: list) -> str:
    """Which null the gains in the key-numbers table are measured from."""
    sets = list(s.get("shortcut_cv", {}))
    if fallback and set(fallback) >= set(sets):
        txt = (f"Gains are over the size-matched {gw} null: labels differ within {perm_units}s, so there is no "
               f"{perm_units}-level chance null, and gains measured from this null are conservative. 95% intervals "
               "in brackets.")
    else:
        txt = f"Gains are over the {perm_units} permutation null (chance level), with 95% intervals in brackets."
        if fallback:
            txt += (f" For {', '.join(fallback)} labels differ within {perm_units}s, and the gain is measured from the "
                    f"size-matched {gw} null instead (conservative).")
    if (s.get("cv_subsample") or {}).get("rule_reapplied"):
        txt += (" The null re-applies the label-dependent subsample, so its mean can lie above 1 / number of "
                "classes (for example when patients with more images receive more positive labels); gains are "
                "measured from that null, not from 1 / number of classes.")
    return txt


def _n_repeats(s: dict) -> int:
    reps = [v.get("n_repeats") for v in s.get("shortcut_cv", {}).values() if v.get("n_repeats")]
    return int(reps[0]) if reps else 1


TOOLKIT_URL = "https://github.com/omoumi/skin-metadata-audit"
CC_BY = ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/")
CC_BY_NC = ("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/")


def content_license(lic: dict) -> dict:
    """License for the tables, figures and text derived from a dataset.

    The license of the derived files is chosen to match the dataset's terms: CC BY-NC for
    non-commercial data, CC BY for data that may be shared with credit, and no open license
    (aggregate tables only) when the terms do not clearly allow redistribution.
    """
    if lic.get("non_commercial"):
        name, url = CC_BY_NC
        reason = "Chosen to match the non-commercial terms of the dataset."
    elif lic.get("allows_redistribution"):
        name, url = CC_BY
        reason = "Chosen to match the license of the dataset."
    else:
        name, url = "No open license; aggregate tables only", ""
        reason = ("The dataset's terms do not clearly allow redistribution; no images are included and the "
                  "tables are aggregate statistics. Ask the dataset owners before reusing them.")
    return {"name": name, "url": url, "reason": reason}


def write_content_license(out_dir: str, title: str, lic: dict) -> str:
    """LICENSE-CONTENT.md for a per-dataset repository."""
    c = content_license(lic)
    text = [f"# License of the {title} audit repository", "",
            "Copyright (C) 2026 Mohammad Hossein Omoumi", "",
            f"## Tables, figures and text", "",
            f"{c['name']}" + (f": {c['url']}" if c["url"] else "") + ".", "", c["reason"], "",
            "## Sample images", "",
            "Every image in samples/ keeps the license of the original image, listed per file in "
            "samples/ATTRIBUTION.md, together with the changes made (centre crop and resize).", "",
            "## Dataset", "",
            f"The dataset itself is not included. Its license: {lic['name']}"
            + (f" ({lic['url']})" if lic.get("url") else "") + ".", "",
            "## Code", "",
            f"The code that produced these files is [skin-metadata-audit]({TOOLKIT_URL}), licensed GPL-3.0-only.",
            ""]
    path = os.path.join(out_dir, "LICENSE-CONTENT.md")
    with open(path, "w") as f:
        f.write("\n".join(text))
    return path
