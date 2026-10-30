# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 Mohammad Hossein Omoumi
"""Near-duplicate detection with perceptual hashes, within and across datasets.

Within a dataset: near-duplicate pairs that fall in different groups would leak
between CV folds. Across datasets: shared images must be removed before any
cross-dataset evaluation (many datasets are subsets of the ISIC Archive).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image


def phash_table(df: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    """64-bit pHash for each row with an image_path. Needs `pip install imagehash`."""
    import imagehash

    def one(p):
        with Image.open(p) as im:
            return int(str(imagehash.phash(im.convert("RGB"))), 16)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        hashes = list(ex.map(one, df["image_path"]))
    return pd.DataFrame({"image_id": df["image_id"].to_numpy(), "group_id": df["group_id"].to_numpy(),
                         "phash": np.array(hashes, dtype=np.uint64)})


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two uint64 vectors -> (len(a), len(b)) uint8."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POP[x.view(np.uint8).reshape(*x.shape, 8)].sum(-1, dtype=np.uint8)


def near_pairs(h1: pd.DataFrame, h2: pd.DataFrame | None = None, max_dist: int = 6,
               chunk: int = 2000) -> pd.DataFrame:
    """All pairs with Hamming distance <= max_dist (within h1 when h2 is None)."""
    same = h2 is None
    h2 = h1 if same else h2
    a, b = h1["phash"].to_numpy(np.uint64), h2["phash"].to_numpy(np.uint64)
    out = []
    for s in range(0, len(a), chunk):
        d = _hamming(a[s:s + chunk], b)
        i, j = np.nonzero(d <= max_dist)
        i = i + s
        if same:
            keep = j > i
            i, j = i[keep], j[keep]
        out.append(pd.DataFrame({"i": i, "j": j, "dist": d[i - s, j]}))
    pairs = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["i", "j", "dist"])
    res = pd.DataFrame({
        "image_id_1": h1["image_id"].to_numpy()[pairs["i"]], "group_1": h1["group_id"].to_numpy()[pairs["i"]],
        "image_id_2": h2["image_id"].to_numpy()[pairs["j"]], "group_2": h2["group_id"].to_numpy()[pairs["j"]],
        "hamming": pairs["dist"].to_numpy()})
    if same:
        res["crosses_groups"] = res["group_1"] != res["group_2"]
    return res.sort_values("hamming").reset_index(drop=True)


def cross_pairs(h1: pd.DataFrame, h2: pd.DataFrame, max_dist: int = 4, chunk: int = 5000) -> pd.DataFrame:
    """Pairs between two hash tables with Hamming distance <= max_dist; the same pairs as
    near_pairs(h1, h2, max_dist) without comparing every pair, so it scales to datasets with
    hundreds of thousands of images. Pigeonhole: the 64 bits are split into max_dist + 1
    blocks, and two hashes within max_dist bits agree exactly on at least one block, so only
    hashes that share a block are compared."""
    a, b = h1["phash"].to_numpy(np.uint64), h2["phash"].to_numpy(np.uint64)
    edges = np.linspace(0, 64, max_dist + 2).astype(int)
    hits = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = np.uint64(((1 << int(hi - lo)) - 1) << int(lo))
        order = np.argsort(b & mask, kind="stable")
        kb = (b & mask)[order]
        for s in range(0, len(a), chunk):
            ka = a[s:s + chunk] & mask
            left, right = np.searchsorted(kb, ka, "left"), np.searchsorted(kb, ka, "right")
            n = right - left
            if not n.any():
                continue
            i = np.repeat(np.arange(len(ka)), n)
            j = order[np.concatenate([np.arange(x, y) for x, y in zip(left[n > 0], right[n > 0])])]
            x = np.bitwise_xor(a[s + i], b[j])
            d = _POP[x.view(np.uint8).reshape(-1, 8)].sum(-1)
            ok = d <= max_dist
            hits.append(np.c_[s + i[ok], j[ok], d[ok]].astype(np.int64))
    h = np.unique(np.vstack(hits), axis=0) if hits else np.zeros((0, 3), np.int64)
    res = pd.DataFrame({"image_id_1": h1["image_id"].to_numpy()[h[:, 0]], "group_1": h1["group_id"].to_numpy()[h[:, 0]],
                        "image_id_2": h2["image_id"].to_numpy()[h[:, 1]], "group_2": h2["group_id"].to_numpy()[h[:, 1]],
                        "hamming": h[:, 2].astype(np.uint8)})
    return res.sort_values(["hamming", "image_id_1", "image_id_2"]).reset_index(drop=True)


def overlap_table(tables: dict[str, pd.DataFrame], max_dist: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Overlap between every pair of datasets: images with the same id (e.g. the same ISIC id,
    a certain overlap) and pHash pairs within max_dist (candidates only: look-alike images, for
    example dark-framed dermoscopy or low-detail 3D tiles, also hash close together, so these
    need a look before they are called duplicates). Returns (summary per dataset pair, all
    candidate pairs)."""
    import itertools
    rows, pairs = [], []
    for a, b in itertools.combinations(tables, 2):
        p = cross_pairs(tables[a], tables[b], max_dist)
        same_id = p["image_id_1"].to_numpy() == p["image_id_2"].to_numpy()
        rows.append({"dataset_1": a, "dataset_2": b,
                     "shared_ids": len(set(tables[a]["image_id"]) & set(tables[b]["image_id"])),
                     "phash_pairs": len(p), "phash_pairs_distance_0": int((p["hamming"] == 0).sum()),
                     "phash_pairs_other_id": int((~same_id).sum()),
                     "images_1_in_pairs": int(p["image_id_1"].nunique()), "images_2_in_pairs": int(p["image_id_2"].nunique())})
        pairs.append(p.assign(dataset_1=a, dataset_2=b, same_id=same_id))
    cols = ["dataset_1", "dataset_2", "image_id_1", "image_id_2", "hamming", "same_id"]
    allp = pd.concat(pairs, ignore_index=True)[cols] if pairs else pd.DataFrame(columns=cols)
    return pd.DataFrame(rows), allp


def id_overlap(ids_by_dataset: dict[str, set]) -> pd.DataFrame:
    """Matrix of shared image ids (e.g. ISIC ids) between datasets."""
    names = list(ids_by_dataset)
    return pd.DataFrame([[len(ids_by_dataset[a] & ids_by_dataset[b]) for b in names] for a in names],
                        index=names, columns=names)


def save_hashes(h: pd.DataFrame, path: str) -> None:
    """Write hashes as 16-digit hex strings (uint64 does not survive CSV round-trips)."""
    h.assign(phash=[format(int(x), "016x") for x in h["phash"]]).to_csv(path, index=False)


def load_hashes(path: str) -> pd.DataFrame:
    h = pd.read_csv(path, dtype={"phash": str, "image_id": str, "group_id": str})
    h["phash"] = np.array([int(x, 16) for x in h["phash"]], dtype=np.uint64)
    return h


# --- human-confirmed duplicates ---------------------------------------------------
# pHash distance alone cannot separate true duplicates from look-alike images: the same
# photo saved at two resolutions can be further apart than two different lesions taken
# with the same camera. Candidates are therefore reviewed by eye with
# review_figure(), and the confirmed pairs are kept in confirmed_duplicates.csv
# (columns image_id_1, image_id_2, reason) next to the results.

def review_figure(pairs: pd.DataFrame, df: pd.DataFrame, path: str,
                  fields: tuple = ("group_id", "label", "age_approx", "sex", "anatom_site_1", "anatom_site_general",
                                   "site", "region"), max_pairs: int | None = None,
                  per_page: int = 30) -> list[str]:
    """Side-by-side figures of candidate pairs with id, metadata, file size and distance.

    One file per page of `per_page` pairs, named <path stem>_p01<ext>, _p02<ext>, ...
    (use a .jpg path: the pages are photos, so JPEG keeps them small);
    pair numbers run on across pages. Returns the list of written files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = pairs.reset_index(drop=True)
    if max_pairs is not None:
        p = p.head(max_pairs)
    meta = df.set_index("image_id")
    cols = [c for c in fields if c in meta.columns] + [c for c in ["file_bytes"] if c in meta.columns]
    stem, ext = os.path.splitext(path)
    ext = ext or ".png"
    pages = [p.iloc[k:k + per_page] for k in range(0, max(len(p), 1), per_page)]
    written = []
    for page_no, page in enumerate(pages, start=1):
        n = max(len(page), 1)
        # constrained layout reserves room for the multi-line titles, so a title never
        # runs into the image of the row above
        row_h = 3.0 + 0.14 * (len(cols) + 3)
        fig, axes = plt.subplots(n, 2, figsize=(8, row_h * n), squeeze=False, layout="constrained")
        for row, (i, r) in enumerate(page.iterrows()):
            for j, img in enumerate([r["image_id_1"], r["image_id_2"]]):
                ax = axes[row, j]
                with Image.open(meta.loc[img, "image_path"]) as im:
                    im = im.convert("RGB")
                    im.thumbnail((600, 600))      # a panel is about 400 px wide; full size only adds bytes
                    ax.imshow(im)
                info = [f"{c}: {meta.loc[img, c]}" for c in cols]
                # one short item per line, so long values never run into the neighbouring panel
                extra = [f"pixel corr {r['pixel_corr']:.3f}"] if "pixel_corr" in r and not pd.isna(r["pixel_corr"]) else []
                ax.set_title("\n".join([f"pair {i}: {img}", *info, f"hamming {r['hamming']}", *extra]),
                             fontsize=7, loc="left")
                ax.axis("off")
        if page.empty:
            axes[0, 0].text(0, 0.5, "no candidate pairs", fontsize=10)
            for ax in axes.ravel():
                ax.axis("off")
        out = f"{stem}_p{page_no:02d}{ext}"
        # photos: JPEG is several times smaller than PNG and good enough to judge duplicates by eye
        fig.savefig(out, dpi=120, bbox_inches="tight",
                    **({"pil_kwargs": {"quality": 85}} if ext.lower() in (".jpg", ".jpeg") else {}))
        plt.close(fig)
        written.append(out)
    return written


PIXEL_SIZE = 64
PIXEL_CENTER = 0.5      # share of each side kept around the centre


def _pixels(path: str, size: int = PIXEL_SIZE, center: float = PIXEL_CENTER) -> np.ndarray:
    """Central part of the grey image (center x each side), resized to size x size and
    standardized (mean 0, SD 1), for pixel_corr. The border is left out on purpose: a black
    dermoscope circle or a ruler takes much of the frame and makes two different lesions look
    alike, while the lesion sits in the centre."""
    with Image.open(path) as im:
        im.draft("L", (size * 4, size * 4))
        g = im.convert("L")
        w, h = g.size
        dx, dy = w * (1 - center) / 2, h * (1 - center) / 2
        g = g.crop((round(dx), round(dy), round(w - dx), round(h - dy)))
        a = np.asarray(g.resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    sd = a.std()
    return (a - a.mean()) / sd if sd > 0 else a * 0


def describe_pairs(pairs: pd.DataFrame, df: pd.DataFrame, workers: int = 16) -> pd.DataFrame:
    """Add file size, label, an exact-copy flag (same md5) and a pixel correlation to
    candidate pairs.

    pHash distance cannot tell a copy from a look-alike (two blurred images with the same
    dark frame can have distance 0); identical bytes can, and the same photo saved again
    (other resolution, compression or colour processing) keeps a pixel correlation near 1
    (Pearson correlation of the central half of the two grey images, resized to 64 x 64),
    while two different lesions rarely do. A copy that was cropped, shifted or rotated gets a
    lower value, so the list is a candidate list, checked by eye. md5 and pixels are computed
    only for the images in the pairs. Exact copies come first, then the pairs by pixel
    correlation, highest first."""
    from .io import md5
    meta = df.set_index("image_id")
    ids = sorted(set(pairs["image_id_1"]) | set(pairs["image_id_2"]))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        digests = dict(zip(ids, ex.map(lambda i: md5(meta.loc[i, "image_path"]), ids)))
        pixels = dict(zip(ids, ex.map(lambda i: _pixels(meta.loc[i, "image_path"]), ids)))
    out = pairs.copy()
    for k in ("1", "2"):
        ids_k = out[f"image_id_{k}"]
        out[f"label_{k}"] = ids_k.map(meta["label"]).to_numpy()
        out[f"file_bytes_{k}"] = ids_k.map(lambda i: os.path.getsize(meta.loc[i, "image_path"])).to_numpy()
    out["identical_file"] = out["image_id_1"].map(digests).to_numpy() == out["image_id_2"].map(digests).to_numpy()
    out["labels_differ"] = out["label_1"].astype(str).to_numpy() != out["label_2"].astype(str).to_numpy()
    n_px = PIXEL_SIZE * PIXEL_SIZE
    out["pixel_corr"] = [round(float(pixels[a] @ pixels[b]) / n_px, 4)
                         for a, b in zip(out["image_id_1"], out["image_id_2"])]
    return out.sort_values(["identical_file", "pixel_corr", "hamming"], ascending=[False, False, True],
                           kind="stable").reset_index(drop=True)


def identical_pairs(described: pd.DataFrame) -> pd.DataFrame:
    """Exact copies from describe_pairs(), in the confirmed-duplicates format (image_id_1,
    image_id_2, reason): identical bytes need no review by eye."""
    t = described[described["identical_file"]]
    reason = np.where(t["labels_differ"], "identical file (md5); conflicting labels " +
                      t["label_1"].astype(str) + " vs " + t["label_2"].astype(str), "identical file (md5)")
    return pd.DataFrame({"image_id_1": t["image_id_1"].to_numpy(), "image_id_2": t["image_id_2"].to_numpy(),
                         "reason": reason})


def likely_same_photo(described: pd.DataFrame, min_corr: float = 0.97) -> pd.DataFrame:
    """Pairs that are not byte-identical but whose pixel correlation is at least min_corr, in
    the confirmed-duplicates format. A candidate list: look at them in the review figure
    (they come first there) before copying them into confirmed_duplicates.csv."""
    t = described[~described["identical_file"] & (described["pixel_corr"] >= min_corr)]
    reason = np.where(t["labels_differ"], "same photo (pixel correlation " + t["pixel_corr"].astype(str)
                      + "); conflicting labels " + t["label_1"].astype(str) + " vs " + t["label_2"].astype(str),
                      "same photo (pixel correlation " + t["pixel_corr"].astype(str) + ")")
    return pd.DataFrame({"image_id_1": t["image_id_1"].to_numpy(), "image_id_2": t["image_id_2"].to_numpy(),
                         "reason": reason})


def load_confirmed(path: str) -> pd.DataFrame:
    """Read confirmed_duplicates.csv (image_id_1, image_id_2, reason)."""
    c = pd.read_csv(path, dtype=str)
    assert {"image_id_1", "image_id_2", "reason"} <= set(c.columns), "need image_id_1, image_id_2, reason"
    assert c["reason"].notna().all(), "every confirmed pair needs a reason"
    return c


def merge_duplicate_groups(df: pd.DataFrame, pairs: pd.DataFrame,
                           missing: str = "error", column: str = "group_id") -> tuple[pd.Series, list[list[str]]]:
    """Union of the groups (or any id column, e.g. the permutation unit) linked by
    confirmed duplicate pairs.

    Returns the new id per row (merged ids get 'g1+g2+...', sorted) and the list
    of merged id sets. Images are not removed; they only can no longer fall into
    different CV folds. missing="error" stops on ids that are not in df (typos);
    missing="skip" ignores pairs not fully in df (e.g. a dermoscopic-only subset).
    """
    ids = set(df["image_id"])
    absent = sorted((set(pairs["image_id_1"]) | set(pairs["image_id_2"])) - ids)
    if missing == "error":
        assert not absent, f"confirmed duplicates not in the dataset: {absent[:5]}"
    pairs = pairs[pairs["image_id_1"].isin(ids) & pairs["image_id_2"].isin(ids)]
    group_of = df.set_index("image_id")[column].astype(str)
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(pairs["image_id_1"], pairs["image_id_2"]):
        ra, rb = find(group_of[a]), find(group_of[b])
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    members: dict[str, set] = {}
    for g in parent:
        members.setdefault(find(g), set()).add(g)
    merged = [sorted(m) for m in members.values() if len(m) > 1]
    rename = {g: "+".join(m) for m in merged for g in m}
    new = df[column].astype(str).map(lambda g: rename.get(g, g))
    return new.where(df[column].astype(str).isin(rename), df[column]), sorted(merged)


def duplicate_drops(df: pd.DataFrame, pairs: pd.DataFrame, fields: list[str]) -> tuple[set, list[dict]]:
    """Images to remove when confirmed duplicates are dropped instead of merged.

    Pairs are joined into clusters (copies of one photo). A cluster whose copies carry
    different labels is removed entirely (which label is right is unknown). Otherwise one
    copy is kept: the one with the fewest missing metadata fields, then one whose group is a
    patient (not 'image:' or 'lesion:'), then the smallest image_id. Returns the image ids to
    drop and one record per cluster."""
    ids = set(df["image_id"])
    pairs = pairs[pairs["image_id_1"].isin(ids) & pairs["image_id_2"].isin(ids)]
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(pairs["image_id_1"], pairs["image_id_2"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    clusters: dict[str, list] = {}
    for x in list(parent):
        clusters.setdefault(find(x), []).append(x)
    meta = df.set_index("image_id")
    n_missing = meta[list(fields)].isna().sum(axis=1) if fields else pd.Series(0, index=meta.index)
    drops, records = set(), []
    for members in clusters.values():
        members = sorted(members)
        labels = sorted(set(meta.loc[members, "label"].astype(str)))
        if len(labels) > 1:
            drops |= set(members)
            records.append({"images": members, "kept": None, "reason": "conflicting labels " + " vs ".join(labels)})
            continue
        def rank(i):
            g = str(meta.loc[i, "group_id"])
            return (int(n_missing[i]), g.startswith(("image:", "lesion:")), i)
        keep = min(members, key=rank)
        drops |= set(members) - {keep}
        records.append({"images": members, "kept": keep, "reason": "copies of one photo"})
    return drops, records
