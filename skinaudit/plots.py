# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Static figures (PNG) for README files. One consistent style across all datasets."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Palette: validated categorical order (light mode) + one-hue sequential ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NULL_BAND = "#d9d8d2"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.labelsize": 10, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "legend.fontsize": 9,
    "legend.labelcolor": INK_2, "font.size": 10, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _content_bbox(fig):
    """Bounding box (inches) of everything drawn so far: axes, tick labels, legends, titles."""
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer())


def _legend_below(fig, handles, labels, ncol=None, fontsize=8):
    """Legend under everything already drawn (rotated tick labels, axis labels), left-aligned,
    so it can never overlap them however long the labels are."""
    bb = _content_bbox(fig)
    w, h = fig.get_size_inches()
    ncol = ncol or len(labels)
    return fig.legend(handles, labels, loc="upper left", ncol=ncol, fontsize=fontsize,
                      bbox_to_anchor=(bb.x0 / w, (bb.y0 - 0.08) / h), bbox_transform=fig.transFigure)


def _suptitle(fig, text, rect_bottom=0.0):
    """Figure title a fixed distance (in inches, not a fraction) above the panels, so tall
    and short figures get the same gap."""
    h = fig.get_size_inches()[1]
    fig.suptitle(text, x=0.01, y=1 - 0.12 / h, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, rect_bottom, 1, 1))     # tight_layout keeps room for the suptitle itself


HEADER_GAP = 0.12        # inches between the header and the top of the content
LINE_GAP = 0.05          # inches between the title and the subtitle


def _place_headers(fig):
    """Title and subtitle of every axes that has a subtitle (the figure's header), placed above
    everything the figure draws (tick labels, bar labels, colorbar ticks, legends) with a fixed
    gap, left-aligned with the content and the subtitle wrapped to its width. Placing them
    after the content is drawn means the header can never run into it, whatever the dataset
    puts on the axes."""
    import textwrap
    heads = []
    for ax in fig.axes:
        subs = [t for t in ax.texts if t.get_gid() == "subtitle"]
        if not subs:
            continue
        # a title set with loc="left" lives in ax._left_title, not in ax.title
        titles = [t for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None))
                  if t is not None and t.get_text()]
        heads.append((titles, subs))
    if not heads:
        return
    for titles, subs in heads:
        for t in titles + subs:
            t.set_visible(False)
    bb = _content_bbox(fig)                           # inches; hidden texts are not counted
    w, h = fig.get_size_inches()
    renderer = fig.canvas.get_renderer()
    x, y = bb.x0 / w, bb.y1 + HEADER_GAP
    chars = max(50, int(bb.width * 13.5))             # about 13.5 characters per inch at 9 pt
    for titles, subs in heads:
        for t in subs:
            txt = fig.text(x, y / h, textwrap.fill(" ".join(t.get_text().split()), chars), ha="left", va="bottom",
                           fontsize=9, color=INK_2, linespacing=1.3)
            y += txt.get_window_extent(renderer).height / fig.dpi + LINE_GAP
        for t in titles:
            txt = fig.text(x, y / h, t.get_text(), ha="left", va="bottom", fontsize=12, fontweight="bold",
                           color=INK)
            y += txt.get_window_extent(renderer).height / fig.dpi + LINE_GAP


def _finish(fig, path, note=None):
    """Save; the header goes above the content (_place_headers), and a note goes under
    everything else, wrapped to the width of the content."""
    _place_headers(fig)
    if note:
        import textwrap
        bb = _content_bbox(fig)
        w, h = fig.get_size_inches()
        chars = max(60, int(bb.width * 15.5))            # about 15 characters per inch at 8 pt
        text = textwrap.fill(" ".join(note.split()), chars)
        fig.text(bb.x0 / w, (bb.y0 - 0.12) / h, text, ha="left", va="top", fontsize=8, color=MUTED,
                 transform=fig.transFigure)
    fig.savefig(path)
    plt.close(fig)
    return os.path.basename(path)


def _num(v, fmt="{:.2f}"):
    """Format without a negative zero ('-0.00')."""
    out = fmt.format(v)
    return out[1:] if out.startswith("-") and float(out) == 0 else out


def _short_block(name: str, max_fields: int = 2) -> str:
    """'a+b+c+d' -> 'a + 3 more' for axis labels; the full list goes in the note."""
    parts = name.split("+")
    return name.replace("+", " + ") if len(parts) <= max_fields else f"{parts[0]} + {len(parts) - 1} more"


def _gw(spec) -> str:
    """One word for the CV group ('patient', 'lesion', 'image') to use in running text; the
    group level can be a longer description such as 'patient (lesion or image where missing)'."""
    return str(spec.group_level).split(" ")[0]


def _subtitle(ax, text):
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=9, color=INK_2, va="bottom", gid="subtitle")


def class_counts(tables, spec, path):
    cc = tables["complete_case_by_class"]
    series = [("All images", cc[("all", "images")]),
              (f"Core fields complete ({', '.join(spec.core_fields)})", cc[("complete_core", "images")]),
              ("All fields complete", cc[("complete", "images")])]
    classes = spec.classes
    y = np.arange(len(classes))
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 0.48 * len(classes) + 1.4))
    for i, (name, s) in enumerate(series):
        vals = s.reindex(classes).fillna(0).to_numpy()
        ax.barh(y + (i - 1) * h, vals, height=h - 0.03, color=SERIES[i], label=name)
        for yy, v in zip(y + (i - 1) * h, vals):
            ax.text(v, yy, f" {int(v):,}", va="center", fontsize=7, color=INK_2)
    ax.set_yticks(y, classes)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.08)
    ax.set_xlabel("Images")
    ax.set_title(f"{spec.title}: images per class", pad=22)
    _subtitle(ax, "How many images survive if rows with missing metadata are dropped")
    h_, l_ = ax.get_legend_handles_labels()
    _legend_below(fig, h_, l_, ncol=1, fontsize=9)
    return _finish(fig, path)


def heatmap(mat: pd.DataFrame, path, title, subtitle, fmt="{:.0%}", vmax=None, cbar_label="", counts=None,
            xrot=None, nan_text="", note=None, tiny_text=None):
    """nan_text: shown in empty cells (on a light grey fill) instead of leaving them blank;
    a cell whose row and column name are the same field shows a dash. xrot=None rotates the
    column labels only when they would not fit their cells. tiny_text (e.g. "<1%"): shown for
    values above 0 that the format would round to 0, so a small class does not read as absent
    (and ">99%" for shares below 1 that would round to 100%)."""
    fig, ax = plt.subplots(figsize=(0.75 * mat.shape[1] + 3.2, 0.45 * mat.shape[0] + 1.8))
    vals = mat.to_numpy(dtype=float)
    vmax = vmax if vmax is not None else max(float(np.nanmax(vals)) if np.isfinite(vals).any() else 0, 1e-9)
    if nan_text and np.isnan(vals).any():
        from matplotlib.colors import ListedColormap
        ax.imshow(np.isnan(vals).astype(float), cmap=ListedColormap([SURFACE, "#efeee9"]), vmin=0, vmax=1,
                  aspect="auto")
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=SEQ, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.isna(v):
                if str(mat.index[i]) == str(mat.columns[j]):
                    ax.text(j, i, "\u2013", ha="center", va="center", fontsize=8, color=MUTED)
                elif nan_text:
                    ax.text(j, i, nan_text, ha="center", va="center", fontsize=7, color=MUTED)
                continue
            txt = fmt.format(v)
            if tiny_text and v > 0 and float(txt.rstrip("%")) == 0:
                txt = tiny_text
            elif tiny_text and v < 1 and txt == "100%":
                txt = ">99%"
            if counts is not None:
                txt += f"\n({int(counts.iat[i, j]):,})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.55 * vmax else INK)
    if xrot is None:      # about 0.075 inch per character at 9 pt; a cell is 0.75 inch wide
        xrot = 30 if max(len(str(c)) for c in mat.columns) * 0.075 > 0.7 else 0
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=xrot, ha="right" if xrot else "center",
                  rotation_mode="anchor" if xrot else "default")
    import textwrap
    ax.set_yticks(range(mat.shape[0]), [textwrap.fill(str(i), 40) for i in mat.index])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # separators on every cell edge, the outer ones too (otherwise a thin dark seam of the
    # image shows along the right and bottom edges of dark cells)
    ax.set_xticks(np.arange(-.5, mat.shape[1] + .5), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0] + .5), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    if cbar_label:
        cb.set_label(cbar_label, color=INK_2, fontsize=9)
    ax.set_title(title, pad=22)
    _subtitle(ax, subtitle)
    return _finish(fig, path, note=note)


def missing_patterns(df, spec, path, top=12):
    pats = df["miss_pattern"].value_counts()
    shown = pats.head(top)
    fields = spec.fields
    fig, (ax_bar, ax_dot) = plt.subplots(2, 1, figsize=(max(6.5, 0.55 * len(shown) + 2.5), 2.6 + 0.32 * len(fields)),
                                         gridspec_kw={"height_ratios": [2.2, 0.25 * len(fields) + 0.4]}, sharex=True)
    x = np.arange(len(shown))
    ax_bar.bar(x, shown.to_numpy(), color=SERIES[0], width=0.7)
    for xx, v in zip(x, shown.to_numpy()):
        ax_bar.text(xx, v, f"{v:,}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax_bar.set_ylabel("Images")
    ax_bar.yaxis.grid(True)
    ax_bar.set_axisbelow(True)
    other = int(pats.iloc[top:].sum())
    ax_bar.set_title(f"{spec.title}: which fields are missing together", pad=22)
    _subtitle(ax_bar, f"Top {len(shown)} missingness patterns" + (f"; {other:,} images in rarer patterns" if other else ""))
    for j, f in enumerate(fields):
        ax_dot.axhline(j, color=GRID, linewidth=0.8, zorder=0)
        for xx, p in zip(x, shown.index):
            missing = p[j] == "1"
            ax_dot.scatter(xx, j, s=46, color=SERIES[1] if missing else SURFACE,
                           edgecolor=SERIES[1] if missing else AXIS, linewidth=1.2, zorder=2)
    ax_dot.set_yticks(range(len(fields)), fields)
    ax_dot.set_ylim(len(fields) - 0.5, -0.5)
    ax_dot.set_xticks([])
    ax_dot.spines["bottom"].set_visible(False)
    ax_dot.spines["left"].set_visible(False)
    ax_dot.tick_params(length=0)
    return _finish(fig, path, note="Filled dot = field missing in that pattern. A column with no filled dots = complete metadata.")


def numeric_by_class(df, spec, field, path, effect=None):
    classes = [c for c in spec.classes if (df["label"] == c).any()]
    data = [df.loc[df["label"] == c, field].dropna().to_numpy() for c in classes]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.6},
                    boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0], "linewidth": 1},
                    whiskerprops={"color": SERIES[0]}, capprops={"color": SERIES[0]})
    ax.set_xticks(range(1, len(classes) + 1), [f"{c}\nn={len(d):,}" for c, d in zip(classes, data)])
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel(field)
    ax.set_title(f"{spec.title}: {field} by class", pad=22)
    if effect is not None:
        _subtitle(ax, f"Kruskal-Wallis epsilon squared = {effect:.3f} (share of rank variance explained by class)")
    return _finish(fig, path, note="Box = interquartile range, line = median, whiskers = 1.5 IQR; outliers hidden.")


def association_bars(miss_assoc, value_assoc, spec, path):
    m = miss_assoc[miss_assoc["versus"].isin(["label", "acquisition"]) & miss_assoc["indicator"].str.startswith("miss_")]
    fields = spec.fields
    cat_vals = value_assoc.set_index("field")
    y = np.arange(len(fields))
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(fields) + 1.6))
    series = [("Missingness vs label", m[m["versus"] == "label"].set_index("indicator"), SERIES[0]),
              (f"Missingness vs {spec.acquisition_name}", m[m["versus"] == "acquisition"].set_index("indicator"), SERIES[1])]
    for i, (name, t, color) in enumerate(series):
        vals = t.reindex([f"miss_{f}" for f in fields])
        v = vals["v_corrected"].fillna(0).to_numpy()
        lo = vals["v_corrected_ci_low"].to_numpy()
        hi = vals["v_corrected_ci_high"].to_numpy()
        ax.barh(y + (i - 1) * h, v, height=h - 0.03, color=color, label=name)
        ok = ~np.isnan(lo) & ~np.isnan(hi)
        # interval drawn from its own ends: the point estimate can fall outside a percentile interval
        ax.hlines((y + (i - 1) * h)[ok], lo[ok], hi[ok], color=_darker(color), linewidth=1.0, zorder=3)
    vv = np.array([cat_vals.loc[f, "effect"] if f in spec.cat_fields and f in cat_vals.index else np.nan for f in fields])
    ok = ~np.isnan(vv)
    ax.barh((y + h)[ok], vv[ok], height=h - 0.03, color=SERIES[2], label="Observed value vs label")
    ax.set_yticks(y, fields)
    ax.invert_yaxis()
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(0.1, float(np.nanmax(np.r_[vv, ax.get_xlim()[1]]))))
    ax.set_xlabel("Bias-corrected Cramer's V (0 = no association)")
    ax.set_title(f"{spec.title}: is missingness informative?", pad=22)
    _subtitle(ax, "Error bars: 95% bootstrap interval, resampling whole " + _gw(spec) +
              "s. Numeric fields have no value bar (see their own figure).")
    h_, l_ = ax.get_legend_handles_labels()
    _legend_below(fig, h_, l_, ncol=3)
    return _finish(fig, path, note="Each interval line has the colour of its bar; a line without a bar is an "
                                   "estimate of 0 with its interval.")


def _darker(color, f=0.6):
    from matplotlib.colors import to_rgb
    return tuple(c * f for c in to_rgb(color))


LABELS = {
    "A_missing_indicators": "Missing-field indicators only",
    "B_acquisition": "Acquisition ({acq}) only",
    "C_image_size": "Image size / resolution only",
    "D_acquisition_plus_indicators": "Acquisition + missing indicators",
    "E_values_with_missing": "Metadata values + missing indicators",
    "F_values_complete": "Metadata values, complete rows",
    "G_core_values_complete_core": "Core values, core-complete rows",
    "I_image_border_stats": "Image border / frame / compression stats",
    "H_monet_concepts_reference": "MONET image concepts (reference, image-derived)",
    "J_tbp_measurements_reference": "TBP lesion measurements (reference, image-derived)",
}


def _ci(r) -> tuple[float, float]:
    """95% interval of the observed balanced accuracy (corrected repeated-CV variance);
    the fold SD when the interval is not available."""
    if "bacc_ci_low" in r and not pd.isna(r["bacc_ci_low"]):
        return float(r["bacc_ci_low"]), float(r["bacc_ci_high"])
    sd = 0.0 if pd.isna(r["bacc_sd"]) else float(r["bacc_sd"])
    return float(r["bacc_mean"]) - sd, float(r["bacc_mean"]) + sd


def shortcut_plot(cv: pd.DataFrame, spec, path, perm_units: str | None = None):
    cv = cv.copy()
    order = [k for k in LABELS if k in cv.index] + [k for k in cv.index if k not in LABELS]
    cv = cv.loc[order]
    y = np.arange(len(cv))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(cv) + 1.8))
    for yy, (_, r) in zip(y, cv.iterrows()):
        if not pd.isna(r["null_q025"]):
            ax.plot([r["null_q025"], r["null_q975"]], [yy, yy], color=NULL_BAND, linewidth=9,
                    solid_capstyle="round", zorder=1)
        if "null_group_q025" in r and not pd.isna(r["null_group_q025"]):
            # sensitivity null (size-matched group permutation): thin bracket under the band
            yb = yy + 0.27
            ax.plot([r["null_group_q025"], r["null_group_q975"]], [yb, yb], color=MUTED, linewidth=1.4, zorder=2)
            ax.plot([r["null_group_mean"]], [yb], marker="|", color=MUTED, markersize=7, zorder=2)
        lo_b, hi_b = _ci(r)
        ax.errorbar(r["bacc_mean"], yy, xerr=[[r["bacc_mean"] - lo_b], [hi_b - r["bacc_mean"]]], fmt="o",
                    color=SERIES[0], ecolor=SERIES[0], markersize=7, elinewidth=1.2, capsize=0, zorder=3)
        p_raw = r["p_perm"]
        has_holm = "p_perm_holm" in r and not pd.isna(r["p_perm_holm"])
        p, lab = (r["p_perm_holm"], "Holm p") if has_holm else (p_raw, "p")
        warn = "" if r["every_fold_has_all_classes"] else "  (some folds lack classes)"
        n_p = int(r["n_perm"]) if not pd.isna(r["n_perm"]) else 0
        # the smallest p a permutation test can give is 1 / (n_perm + 1)
        at_floor = not pd.isna(p_raw) and p_raw <= 1 / (n_p + 1) + 1e-12 and p < 1
        ptxt = "" if pd.isna(p) else (f"{lab}\u2264{p:.3g} ({n_p} perm.)" if at_floor else f"{lab}={p:.3g}")
        ax.text(max(hi_b, r["null_q975"] if not pd.isna(r["null_q975"]) else 0) + 0.01, yy,
                f"{r['bacc_mean']:.3f}  {ptxt}  n={int(r['n_rows']):,}{warn}", va="center", fontsize=8, color=INK_2)
    chance = float(cv["chance"].iloc[0])
    ax.axvline(chance, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(chance, -0.9, " chance", fontsize=8, color=MUTED, va="center")
    ax.set_yticks(y, [LABELS.get(k, k).format(acq=spec.acquisition_name) for k in cv.index])
    ax.invert_yaxis()
    ax.set_ylim(len(cv) - 0.5, -1.3)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    hi = float(np.nanmax([_ci(r)[1] for _, r in cv.iterrows()]))
    ax.set_xlim(0, min(1.0, hi + 0.25))
    ax.set_xlabel(f"Balanced accuracy, {_gw(spec)}-grouped 5-fold CV")
    ax.set_title(f"{spec.title}: what can predict the label without the image?", pad=22)
    n_rep = int(cv["n_repeats"].iloc[0]) if "n_repeats" in cv.columns and not pd.isna(cv["n_repeats"].iloc[0]) else 1
    _subtitle(ax, f"Dot = mean of {n_rep} CV repeats (line = 95% interval); grey band = 95% range of the same "
                  "pipeline on permuted labels")
    has_sens = "null_group_q025" in cv.columns and cv["null_group_q025"].notna().any()
    note = None
    if has_sens:
        unit = perm_units or _gw(spec)
        tests = set(cv["test_null"].dropna()) if "test_null" in cv.columns else {"unit"}
        which = (f"the size-matched {_gw(spec)} null (several {unit}s per {_gw(spec)}: labels of one "
                 f"{_gw(spec)} are alike, which the {unit} null ignores)" if tests == {"group"} else
                 f"the {unit} null" if tests <= {"unit", "image"} else "the null named in shortcut_cv.csv (test_null)")
        note = (f"Grey band: one label per {unit} shuffled (chance level of the gain). Thin bracket below: whole "
                f"{_gw(spec)}s swap labels with {_gw(spec)}s of the same size\n"
                f"(keeps the clustering and the link between {_gw(spec)} size and label). p-values come from "
                f"{which},\nHolm-adjusted over the feature sets.")
    return _finish(fig, path, note=note)


def image_stats_plot(assoc: pd.DataFrame, spec, path):
    t = assoc.sort_values("epsilon2_vs_label")
    y = np.arange(len(t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(t) + 1.6))
    colors = {"border": SERIES[1], "file": SERIES[3], "whole image": SERIES[0]}
    ax.barh(y + h / 2, t["epsilon2_vs_label"], height=h - 0.03, color=[colors[r] for r in t["region"]])
    handles = [Patch(color=c, label=f"vs label, {r}") for r, c in colors.items()]
    if t["epsilon2_vs_acquisition"].notna().any():
        ax.barh(y - h / 2, t["epsilon2_vs_acquisition"].fillna(0), height=h - 0.03, color=NULL_BAND)
        handles.append(Patch(color=NULL_BAND, label=f"vs {spec.acquisition_name}"))
    ax.set_yticks(y, t.index)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Kruskal-Wallis epsilon squared")
    ax.set_title(f"{spec.title}: image statistics by class", pad=22)
    _subtitle(ax, "Border / file statistics mostly reflect the camera and frame rather than the lesion, so a CNN could "
                  "use them as a shortcut")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _finish(fig, path, note="Whole-image colour reflects the lesion itself, so it is not used as a shortcut test. "
                                   "Border and file statistics can still include part of a large lesion\n"
                                   "and its texture, so the border shortcut test may overstate the pure camera/frame shortcut.")


def complete_case_bias(df, spec, tables, path):
    """Is the complete-case subset (used for missingness injection) like the whole dataset?"""
    share = tables["complete_case_class_share"].astype(float)
    fields = list(spec.core_fields)
    n = 1 + len(fields)
    first_w = 4.2 + 0.3 * max(0, len(share) - 4)          # the class-mix panel grows with the classes
    fig, axes = plt.subplots(1, n, figsize=(first_w + 4.2 * (n - 1), 4.2), squeeze=False,
                             gridspec_kw={"width_ratios": [first_w] + [4.2] * (n - 1)})
    ax = axes[0, 0]
    cols = [("all", "All images"), ("complete", "All fields complete"), ("complete_core", "Core fields complete"),
            ("incomplete", "Some field missing")]
    x = np.arange(len(share))
    w = 0.8 / len(cols)
    for i, (c, lab) in enumerate(cols):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, share[c].to_numpy(), width=w - 0.02, color=SERIES[i], label=lab)
    ax.set_xticks(x, share.index, rotation=45, ha="right")
    ax.set_ylabel("Share of images")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_title("Class mix", fontsize=10, loc="left")
    mix_handles = ax.get_legend_handles_labels()
    groups = [("all", df), ("complete", df[df["complete"] == 1]), ("incomplete", df[df["complete"] == 0])]
    for j, f in enumerate(fields, start=1):
        ax = axes[0, j]
        if f in spec.num_fields:
            data = [g[f].dropna().to_numpy() for _, g in groups]
            ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]},
                       medianprops={"color": INK})
            ax.set_xticks([1, 2, 3], ["All", "Complete", "Incomplete"])
            ax.yaxis.grid(True)
        else:
            t = tables[f"complete_case_{f}"][["all", "complete", "incomplete"]].astype(float).fillna(0)
            top = t["all"].sort_values(ascending=False).index[:7]
            t = pd.concat([t.loc[top], t.drop(index=top).sum().to_frame("other").T]) if len(t) > 7 else t.loc[top]
            bottom = np.zeros(3)
            for k, (cat, row) in enumerate(t.iterrows()):
                vals = row.to_numpy(dtype=float)
                ax.bar([0, 1, 2], vals, bottom=bottom, color=SERIES[k % len(SERIES)], label=str(cat), width=0.6)
                bottom += vals
            ax.set_xticks([0, 1, 2], ["All", "Complete", "Incomplete"])
            ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax.set_ylabel("Share (observed values)")
        ax.set_title(f, fontsize=10, loc="left")
    _suptitle(fig, f"{spec.title}: is the complete-case subset representative?")
    # class-mix legend under the whole figure, so it never meets rotated class names
    _legend_below(fig, *mix_handles, ncol=4)
    return _finish(fig, path, note="Missingness-injection experiments start from complete rows; differences here "
                                   "indicate a selection bias that those experiments inherit.")


def group_size_plot(df, spec, tables, path):
    """Images per CV group by class, and missing rate by group size; not drawn when every
    group is a single image (nothing to show)."""
    if df["group_size"].max() <= 1:
        return None
    classes = [c for c in spec.classes if (df["label"] == c).any()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    data = [df.loc[df["label"] == c, "group_size"].to_numpy() for c in classes]
    a1.boxplot(data, widths=0.5, showfliers=True, patch_artist=True,
               boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]}, medianprops={"color": INK},
               flierprops={"markersize": 2, "markeredgecolor": MUTED})
    a1.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    # ticks at 1, 2, 5, 10, 20, 50 ... printed as whole numbers, so small ranges still get labels
    a1.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.set_xticks(range(1, len(classes) + 1), classes, rotation=45, ha="right")
    a1.set_ylabel(f"Images in the image's {spec.group_level.split(' ')[0]} (log)")
    a1.yaxis.grid(True)
    a1.set_title(f"{spec.group_level.split(' ')[0].capitalize()} size by class (one point per image)",
                 fontsize=10, loc="left")
    t = tables["missing_rate_by_group_size"]
    t = t.loc[:, t.loc["images"] > 0]                      # size bins with no image are not drawn
    counts = t.loc["images"]
    rates = t.drop(index="images")
    rates = rates[rates.max(axis=1) > 0]
    # fields that go missing together have identical lines: draw them once
    merged = {}
    for f, row in rates.iterrows():
        key = tuple(np.round(row.to_numpy(dtype=float), 6))
        merged.setdefault(key, []).append(str(f))
    xs = np.arange(len(t.columns))
    for k, (key, names) in enumerate(merged.items()):
        lab = names[0] if len(names) == 1 else f"{names[0]} + {len(names) - 1} more (same rate)"
        a2.plot(xs, np.array(key), marker="o", color=SERIES[k % len(SERIES)], label=lab)
    a2.set_xticks(xs, [f"{c}\nn={int(n):,}" for c, n in counts.items()])
    a2.set_xlabel(f"Images per {spec.group_level.split(' ')[0]}")
    a2.set_ylabel("Missing rate")
    a2.yaxis.grid(True)
    a2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    a2.set_title("Missing rate by group size", fontsize=10, loc="left")
    _suptitle(fig, f"{spec.title}: group size, label and missingness")
    return _finish(fig, path, note="If group size differs by class and missingness differs by group size, "
                                   "missingness could predict the label through group size alone.")


def importance_plot(tables, spec, path):
    """Permutation importance of each feature in the missing-indicator and full-metadata models."""
    keys = [k for k in ["importance_A_missing_indicators", "importance_E_values_with_missing"] if k in tables]
    titles = {"importance_A_missing_indicators": "Missing-field indicators only",
              "importance_E_values_with_missing": "Metadata values + missing indicators"}
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 0.4 * max(len(tables[k]) for k in keys) + 1.8),
                             squeeze=False)
    for ax, k in zip(axes[0], keys):
        t = tables[k].iloc[::-1]
        colors = [SERIES[1] if str(i).startswith("miss_") else SERIES[0] for i in t.index]
        ax.barh(t.index.astype(str), t["importance_mean"], xerr=t["importance_sd"], color=colors,
                ecolor=INK_2, capsize=2)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("Drop in balanced accuracy when shuffled")
        ax.set_title(titles[k], fontsize=10, loc="left")
    _suptitle(fig, f"{spec.title}: which fields carry the signal?")
    return _finish(fig, path, note="Orange = missing indicator (miss_*), blue = observed value. Held-out folds, "
                                   "10 shuffles per fold; bars = mean, lines = SD across folds.")


def confusion_plot(tables, spec, path):
    conf = tables["confusion_A_missing_indicators"]
    rate = conf.div(conf.sum(axis=1).replace(0, np.nan), axis=0)
    return heatmap(rate, path, f"{spec.title}: which classes does missingness identify?",
                   "Out-of-fold predictions of the missing-indicator model; row share (true class), counts in brackets",
                   vmax=1.0, counts=conf, cbar_label="Share of true class", xrot=45)


def categorical_by_class(tables, spec, path):
    fields = [f for f in spec.cat_fields if f"{f}_share_within_class" in tables]
    if not fields:
        return None
    ncol = min(3, len(fields))
    nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
    for ax, f in zip(axes.ravel(), fields):
        t = tables[f"{f}_share_within_class"].astype(float).fillna(0)
        if len(t) > 7:
            top = t.sum(axis=1).sort_values(ascending=False).index[:7]
            t = pd.concat([t.loc[top], t.drop(index=top).sum().to_frame("other").T])
        x = np.arange(t.shape[1])
        bottom = np.zeros(t.shape[1])
        for k, (cat, row) in enumerate(t.iterrows()):
            ax.bar(x, row.to_numpy(dtype=float), bottom=bottom, color=SERIES[k % len(SERIES)], label=str(cat), width=0.7)
            bottom += row.to_numpy(dtype=float)
        ax.set_xticks(x, t.columns, rotation=45, ha="right", fontsize=8)
        ax.set_title(f, fontsize=10, loc="left")
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for ax in axes.ravel()[len(fields):]:
        ax.axis("off")
    _suptitle(fig, f"{spec.title}: categorical fields by class (observed values only)")
    return _finish(fig, path)


def null_distributions(tables, spec, path, perm_units: str | None = None):
    unit = perm_units or _gw(spec)
    nulls, cv = tables["permutation_nulls"], tables["shortcut_cv"]
    sets = [k for k in cv.index if (nulls["feature_set"] == k).any()]
    if not sets:
        return None
    ncol = min(4, len(sets))
    nrow = int(np.ceil(len(sets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
    for ax, k in zip(axes.ravel(), sets):
        main = nulls[(nulls["feature_set"] == k) & (nulls["null"] == "unit")]["bacc"]
        grp = nulls[(nulls["feature_set"] == k) & (nulls["null"] != "unit")]["bacc"]
        # the p-value compares the first CV split with the null (built from single splits)
        obs = float(cv.loc[k, "bacc_split1"] if "bacc_split1" in cv.columns else cv.loc[k, "bacc_mean"])
        lo = min(main.min(), grp.min() if len(grp) else main.min(), obs)
        hi = max(main.max(), grp.max() if len(grp) else main.max(), obs)
        bins = np.linspace(lo - 0.01, hi + 0.01, 25)
        ax.hist(main, bins=bins, color=NULL_BAND, label=f"{unit} null (chance level)")
        if len(grp):
            ax.hist(grp, bins=bins, histtype="step", color=INK_2, linewidth=1.2, label="size-matched null")
        ax.axvline(obs, color=SERIES[0], linewidth=2, label="observed")
        ax.set_title(LABELS.get(k, k).format(acq=spec.acquisition_name), fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(sets):]:
        ax.axis("off")
    from matplotlib.ticker import MaxNLocator
    for ax in axes.ravel()[:len(sets)]:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    _suptitle(fig, f"{spec.title}: permutation null distributions")
    _legend_below(fig, handles, labels, ncol=3)
    return _finish(fig, path, note="Balanced accuracy of the same pipeline on permuted labels; blue line = real labels "
                                     "(first CV split, the score the p-value uses).")


def mechanism_plot(tables, spec, path):
    """AUC of predicting a field's missingness without (a) and with (b) the label.
    One panel per input variant, stacked; the test results sit in a column right of each
    panel, so they cannot run into the other panel or the axis labels."""
    from matplotlib.transforms import blended_transform_factory
    mall = tables["missingness_mechanism"]
    m = mall[mall["skipped"].isna()]
    if m.empty:
        return None
    variants = [v for v in ["with_indicators", "values_only"] if (m["variant"] == v).any()]
    titles = {"with_indicators": "Inputs: other fields + their missing indicators (all rows)",
              "values_only": "Inputs: observed values only (rows where those are complete)"}
    heights = [0.42 * int((m["variant"] == v).sum()) + 1.15 for v in variants]
    fig, axes = plt.subplots(len(variants), 1, figsize=(11, sum(heights) + 0.4), squeeze=False,
                             gridspec_kw={"height_ratios": heights})

    def pv(x):
        return "    \u2013" if pd.isna(x) else f"{x:>5.2g}"
    for ax, v in zip(axes[:, 0], variants):
        t = m[m["variant"] == v].reset_index(drop=True)
        y = np.arange(len(t))
        for i, r in t.iterrows():
            ax.plot([r["auc_a"], r["auc_b"]], [i, i], color=AXIS, linewidth=2, zorder=1)
        ax.scatter(t["auc_a"], y, s=55, color=SERIES[0], zorder=3, label="(a) without the label")
        ax.scatter(t["auc_b"], y, s=55, color=SERIES[1], zorder=3, label="(b) with the label")
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.02, -0.9, f"{'gain':>6}  {'CRT p':>5}  {'swap p':>6}  {'label R2':>8}  {'n':>7}", transform=tr,
                fontsize=7.5, color=MUTED, va="center", family="monospace")
        for i, r in t.iterrows():
            r2 = "    \u2013" if pd.isna(r.get("label_r2")) else _num(r["label_r2"])
            sup = r.get("label_dependence_supported")
            star = "*" if (sup is not None and not pd.isna(sup) and bool(sup)) else " "
            ax.text(1.02, i, f"{_num(r['auc_gain'], '{:+.3f}'):>6}  {pv(r.get('crt_p_holm'))}  "
                             f"{pv(r.get('group_swap_p_holm')):>6}{star} {r2:>8}  {int(r['n_rows']):>7,}",
                    transform=tr, fontsize=7.5, color=INK_2, va="center", family="monospace")
        ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_yticks(y, [_short_block(f) for f in t["fields"]])
        ax.set_ylim(len(t) - 0.5, -1.4)
        lo = min(0.4, float(np.nanmin(np.r_[t["auc_a"], t["auc_b"]])) - 0.05)
        ax.set_xlim(lo, 1.0)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_title(titles[v], fontsize=10, loc="left")
    axes[-1, 0].set_xlabel("AUC, predicting that the field is missing (grouped 5-fold CV); 0.5 = chance level")
    _suptitle(fig, f"{spec.title}: what explains that a field is missing?")
    _legend_below(fig, *axes[0, 0].get_legend_handles_labels(), ncol=2)
    note = ("AUC(a) clearly above 0.5 is evidence against MCAR; it cannot tell MAR from MNAR. The gain (b - a) may overstate the dependence on the label: "
            "the label can stand in for what the model did not learn from the other fields (more likely when label "
            "R2 is high). p-values are Holm-adjusted over the rows: CRT = conditional randomization test; swap = "
            + ("images swap labels within strata of acquisition x age" if _gw(spec) == "image" else
               f"whole {_gw(spec)}s swap labels within strata of images per {_gw(spec)} x acquisition x age")
            + " (no age when age is the target). * = both reject at "
            "0.05, the condition for calling the missingness label-dependent; a field without * may still be.")
    blocks = [f for f in m["fields"].unique() if len(str(f).split("+")) > 2]
    if blocks:
        note += " Fields missing together, modelled as one block: " + "; ".join(
            f"{_short_block(b)} = {', '.join(b.split('+'))}" for b in blocks) + "."
    skipped = mall[mall["skipped"].notna() & (mall["variant"] == "-")]["fields"].tolist()
    if skipped:
        note += " Not modelled (too few missing images or groups): " + ", ".join(map(str, skipped)) + "."
    return _finish(fig, path, note=note)


def smd_plot(tables, spec, path):
    """Standardized mean differences between the complete-case subsets and the data."""
    t = tables["complete_case_smd"]
    comps = [("vs_all", "Subset vs all images"), ("vs_rest", "Subset vs the other images")]
    variables = list(dict.fromkeys(t["variable"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(variables) + 2.2), sharey=True)
    for ax, (c, title) in zip(axes, comps):
        for k, (subset, lab) in enumerate([("complete", "All fields complete"),
                                           ("complete_core", "Core fields complete")]):
            d = t[(t["comparison"] == c) & (t["subset"] == subset)].set_index("variable").reindex(variables)
            yy = np.arange(len(variables)) + (k - 0.5) * 0.3
            v = d["smd_corrected"].abs().to_numpy(dtype=float)
            if "ci_low" in d.columns:
                lo, hi = d["ci_low"].to_numpy(dtype=float), d["ci_high"].to_numpy(dtype=float)
                # the interval of |SMD| for a signed numeric SMD whose interval crosses zero starts at 0
                alo = np.where(lo * hi < 0, 0, np.minimum(np.abs(lo), np.abs(hi)))
                ahi = np.maximum(np.abs(lo), np.abs(hi))
                ok = ~np.isnan(alo)
                ax.hlines(yy[ok], alo[ok], ahi[ok], color=SERIES[k], linewidth=1.2, alpha=0.7)
            ax.scatter(v, yy, s=36, color=SERIES[k], zorder=3, label=lab)
        ax.axvline(0.1, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(0.1, -0.9, " 0.1", fontsize=8, color=MUTED, va="center")
        ax.set_yticks(np.arange(len(variables)), variables)
        ax.set_ylim(len(variables) - 0.5, -1.3)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("|SMD| (categorical: multi-category SMD)")
        ax.set_title(title, fontsize=10, loc="left")
    _suptitle(fig, f"{spec.title}: how different are the complete-case rows?")
    _legend_below(fig, *axes[0].get_legend_handles_labels(), ncol=2)
    return _finish(fig, path, note="|SMD| > 0.1 is a common convention for imbalance, not a test. Fields are compared on observed values; "
                                   "lines = 95% bootstrap interval over groups. Categorical SMDs are bias-corrected\n"
                                   "(sampling noise alone pushes them above 0), with basic-bootstrap intervals.")


NA_NOTE = ("n/a = not testable: the row field is never (or always) missing where the column field is observed, "
           "e.g. two fields that always go missing together. A dash marks a field against itself.")


def make_all_figures(df, spec, tables, fig_dir, perm_units: str | None = None) -> list[str]:
    os.makedirs(fig_dir, exist_ok=True)
    out = [class_counts(tables, spec, os.path.join(fig_dir, "class_counts.png"))]
    mr = tables["missing_rate_by_class"][[f"miss_{f}" for f in spec.fields]].T
    mr.index = spec.fields
    out.append(heatmap(mr, os.path.join(fig_dir, "missing_by_class.png"),
                       f"{spec.title}: missing rate per field and class",
                       "Share of images in each class where the field is empty", vmax=max(0.2, float(mr.max().max())),
                       cbar_label="Missing rate"))
    out.append(missing_patterns(df, spec, os.path.join(fig_dir, "missing_patterns.png")))
    out.append(association_bars(tables["missing_association"], tables["value_association"], spec,
                                os.path.join(fig_dir, "association.png")))
    va = tables["value_association"].set_index("field")
    for f in spec.num_fields:
        out.append(numeric_by_class(df, spec, f, os.path.join(fig_dir, f"{f}_by_class.png"),
                                    effect=va.loc[f, "effect"] if f in va.index else None))
    mm = tables.get("missing_vs_other_fields_matrix")
    if mm is not None and mm.notna().any().any():
        mm = mm.dropna(how="all")
        out.append(heatmap(mm, os.path.join(fig_dir, "missing_vs_other_fields.png"),
                           f"{spec.title}: does missingness depend on the other fields?",
                           "Row = field whose missingness is tested; column = observed values of another field "
                           "(numeric fields in quintiles). Bias-corrected Cramer's V",
                           fmt="{:.2f}", vmax=max(0.3, float(np.nanmax(mm.to_numpy(dtype=float)))),
                           cbar_label="Cramer's V (corrected)", xrot=30, nan_text="n/a",
                           note=NA_NOTE))
    ml = tables.get("missing_vs_other_fields_within_label_matrix")
    if ml is not None and ml.notna().any().any():
        ml = ml.dropna(how="all")
        # same colour scale as the unadjusted matrix so the two can be compared cell by cell
        vmax_m = max(0.3, float(np.nanmax(mm.to_numpy(dtype=float)))) if mm is not None else 0.3
        out.append(heatmap(ml, os.path.join(fig_dir, "missing_vs_other_fields_within_label.png"),
                           f"{spec.title}: missingness vs other fields, within each class",
                           "Same check computed inside each class, then averaged (weighted by size). "
                           "Cells that fade here may run through the label",
                           fmt="{:.2f}", vmax=vmax_m, cbar_label="Cramer's V (corrected)", xrot=30,
                           nan_text="n/a", note=NA_NOTE + " Within classes, a cell also needs a class with at least "
                                                          "30 rows that has both missing and observed values."))
    ab = tables["acquisition_by_class"]
    share = ab / ab.sum(axis=0).replace(0, np.nan)
    out.append(heatmap(share, os.path.join(fig_dir, "acquisition_by_class.png"),
                       f"{spec.title}: where each class comes from ({spec.acquisition_name})",
                       f"Column share: fraction of each class's images per {spec.acquisition_name}; image counts in brackets",
                       vmax=1.0, counts=ab, cbar_label="Share of class"))
    if "resolution_by_class" in tables:
        rb = tables["resolution_by_class"]
        out.append(heatmap(rb / rb.sum(axis=0).replace(0, np.nan), os.path.join(fig_dir, "resolution_by_class.png"),
                           f"{spec.title}: image resolution by class",
                           "Column share of the 15 most common resolutions; a class concentrated in one row is a possible shortcut",
                           vmax=1.0, cbar_label="Share of class"))
    if "image_stats_association" in tables:
        out.append(image_stats_plot(tables["image_stats_association"], spec, os.path.join(fig_dir, "image_stats.png")))
    out.append(shortcut_plot(tables["shortcut_cv"], spec, os.path.join(fig_dir, "shortcut_cv.png"),
                             perm_units=perm_units))
    extra = [
        lambda: complete_case_bias(df, spec, tables, os.path.join(fig_dir, "complete_case_bias.png")),
        lambda: group_size_plot(df, spec, tables, os.path.join(fig_dir, "group_size.png")),
        lambda: importance_plot(tables, spec, os.path.join(fig_dir, "feature_importance.png"))
        if any(k.startswith("importance_") for k in tables) else None,
        lambda: confusion_plot(tables, spec, os.path.join(fig_dir, "missingness_confusion.png"))
        if "confusion_A_missing_indicators" in tables else None,
        lambda: categorical_by_class(tables, spec, os.path.join(fig_dir, "categorical_by_class.png")),
        lambda: null_distributions(tables, spec, os.path.join(fig_dir, "null_distributions.png"), perm_units)
        if tables["permutation_nulls"].shape[0] else None,
        lambda: mechanism_plot(tables, spec, os.path.join(fig_dir, "missingness_mechanism.png"))
        if "missingness_mechanism" in tables else None,
        lambda: smd_plot(tables, spec, os.path.join(fig_dir, "complete_case_smd.png"))
        if "complete_case_smd" in tables else None,
    ]
    out += [f for f in (make() for make in extra) if f]
    return out
