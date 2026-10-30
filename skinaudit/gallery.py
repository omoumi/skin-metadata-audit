# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Sample-image gallery for a public README, with per-image attribution.

Only runs when the dataset license allows redistribution. Every shown image is
resized and centre-cropped; the attribution file states this change, as CC
licenses require.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from .plots import INK, INK_2, SURFACE  # noqa: E402


def select_samples(df: pd.DataFrame, spec, n_per_class: int = 3, seed: int = 2026,
                   prefer: pd.Series | None = None, require_core: bool = True) -> pd.DataFrame:
    """Pick n images per class: one per group, core metadata present, preferred rows first.

    ``prefer`` is a boolean Series (e.g. image is dermoscopic) used to rank candidates.
    Selection is deterministic for a given seed.
    """
    d = df.copy()
    if require_core:
        d = d[d[spec.core_fields].notna().all(axis=1)]
    d["_pref"] = 0 if prefer is None else (~prefer.reindex(d.index).fillna(False)).astype(int)
    d["_rand"] = np.random.default_rng(seed).random(len(d))
    picks = []
    for c in spec.classes:
        g = d[d["label"] == c].sort_values(["_pref", "_rand"])
        g = g.drop_duplicates("group_id")
        picks.append(g.head(n_per_class))
    return pd.concat(picks).drop(columns=["_pref", "_rand"])


def _thumb(path: str, size: int) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return im.resize((size, size), Image.LANCZOS)


def _caption(r, spec) -> str:
    parts = []
    for f in spec.core_fields:
        v = r.get(f)
        if pd.isna(v):
            continue
        parts.append(f"{int(v)} y" if f.startswith("age") else str(v))
    return " | ".join(parts)


def make_gallery(samples: pd.DataFrame, spec, out_dir: str, size: int = 320) -> dict:
    """Write thumbnails, a grid figure and ATTRIBUTION.md. Returns file names."""
    lic = spec.license
    assert lic.allows_redistribution, f"{spec.name}: license '{lic.name}' does not allow showing images"
    assert "image_path" in samples.columns, "samples need an image_path column"
    img_dir = os.path.join(out_dir, "samples")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for _, r in samples.iterrows():
        name = f"{r['label']}_{r['image_id']}.jpg"
        _thumb(r["image_path"], size).save(os.path.join(img_dir, name), quality=88)
        rows.append({"file": f"samples/{name}", "image_id": r["image_id"], "label": r["label"],
                     "license": r.get("license", lic.name) if pd.notna(r.get("license", np.nan)) else lic.name,
                     "attribution": r.get("attribution", "") if pd.notna(r.get("attribution", np.nan)) else "",
                     "caption": _caption(r, spec)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(img_dir, "manifest.csv"), index=False)

    classes = [c for c in spec.classes if (manifest["label"] == c).any()]
    nrow = int(manifest.groupby("label").size().max())
    # Classes as columns, samples as rows: a wide grid reads better in a README.
    fig, axes = plt.subplots(nrow, len(classes), figsize=(1.75 * len(classes), 2.15 * nrow + 0.9), squeeze=False)
    for j, c in enumerate(classes):
        m = manifest[manifest["label"] == c].reset_index(drop=True)
        for i in range(nrow):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i < len(m):
                ax.imshow(Image.open(os.path.join(out_dir, m.loc[i, "file"])))
                cap = m.loc[i, "caption"].replace(" | ", "\n", 1)
                ax.set_xlabel(f"{m.loc[i, 'image_id']}\n{cap}", fontsize=6.3, color=INK_2, linespacing=1.15)
            if i == 0:
                ax.set_title(c, fontsize=11, color=INK, fontweight="bold", loc="center", pad=6)
    fig.suptitle(f"{spec.title}: example images per class", x=0.01, y=0.995, ha="left", va="top",
                 fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.0, f"Images: {lic.name}. Resized and centre-cropped. Per-image credit in samples/ATTRIBUTION.md.",
             fontsize=7.5, color=INK_2, ha="left", va="top")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.01, 1, 0.965))
    grid_name = "sample_grid.jpg"
    fig.savefig(os.path.join(out_dir, grid_name), dpi=130, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)

    lines = [f"# Image attribution: {spec.title}", "",
             f"License: [{lic.name}]({lic.url})" + (" (non-commercial use only)" if lic.non_commercial else ""), "",
             "Changes: every image below was centre-cropped to a square and resized to "
             f"{size}x{size} pixels. No other changes were made.", ""]
    if lic.citations:
        lines += ["Required citations:", ""] + [f"- {c}" for c in lic.citations] + [""]
    lines += ["| File | ISIC / image id | Class | License | Attribution |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{os.path.basename(r['file'])}` | {r['image_id']} | {r['label']} | {r['license']} | {r['attribution']} |")
    with open(os.path.join(img_dir, "ATTRIBUTION.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"grid": grid_name, "manifest": "samples/manifest.csv", "attribution": "samples/ATTRIBUTION.md",
            "n_images": len(rows)}
