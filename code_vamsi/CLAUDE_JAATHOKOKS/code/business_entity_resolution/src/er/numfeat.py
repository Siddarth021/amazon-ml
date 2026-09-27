"""Near-miss house number features: digit-drop noise (4182 -> 182, true match) vs neighbours (3130 -> 3131).

Same chunking as features.py (parts align row by row).
Output: work/worlds/<w>/numfeat/part_{i:05d}.parquet
Run:    python -m er.numfeat --world B
"""
import argparse
import gc

import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from . import config
from .pairs import chunk, n_parts, pairs
from .utils import get_logger, stage_done, timed, write_parquet

log = get_logger("numfeat")


def lev(a: pl.Series, b: pl.Series) -> np.ndarray:
    """Levenshtein distance, -1 when either side is missing."""
    d = process.cpdist(a.fill_null("").to_list(), b.fill_null("").to_list(), scorer=Levenshtein.distance,
                       workers=-1, dtype=np.int32)
    d[(a.is_null() | b.is_null()).to_numpy()] = -1
    return d


def chunk_numfeat(c: pl.DataFrame, s1n: pl.DataFrame, recn: pl.DataFrame) -> pl.DataFrame:
    s = s1n[c["s1_idx"].to_numpy()]
    r = recn[c["rec_idx"].to_numpy()]
    a, b = s["hnum"], r["hnum"]
    best = np.full(len(c), np.iinfo(np.int32).max, dtype=np.int64)
    for i in range(3):
        for j in range(3):
            d = lev(s["nums"].list.get(i, null_on_oob=True), r["nums"].list.get(j, null_on_oob=True)).astype(np.int64)
            best = np.where(d >= 0, np.minimum(best, d), best)
    best[best == np.iinfo(np.int32).max] = -1
    df = pl.DataFrame({"a": a, "b": b})
    ia = pl.col("a").str.slice(0, 9).cast(pl.Int64, strict=False)
    ib = pl.col("b").str.slice(0, 9).cast(pl.Int64, strict=False)
    missing = pl.col("a").is_null() | pl.col("b").is_null()
    drop1 = lambda x, y: ((pl.col(x).str.slice(1) == pl.col(y)) | (pl.col(x).str.head(-1) == pl.col(y))) \
        & (pl.col(x).str.len_chars() == pl.col(y).str.len_chars() + 1)
    g = df.select(
        hn_affix=pl.when(missing).then(0).otherwise((drop1("a", "b") | drop1("b", "a")).cast(pl.Int8)).cast(pl.Int8),
        hn_reldiff=pl.when(missing | ia.is_null() | ib.is_null()).then(-1.0)
        .otherwise((ia - ib).abs() / pl.max_horizontal(ia, ib, pl.lit(1))).cast(pl.Float32),
        hn_same_len=pl.when(missing).then(-1)
        .otherwise((pl.col("a").str.len_chars() == pl.col("b").str.len_chars()).cast(pl.Int8)).cast(pl.Int8),
    )
    return pl.concat([c.select("s1_idx", "rec_idx"),
                      pl.DataFrame({"hn_ed": lev(a, b).astype(np.int16), "hn_best_ed": best.astype(np.int16)}), g],
                     how="horizontal")


def run(world: str, cand: str = "cand_v1", norm: str = "_v2", out: str = "numfeat", force: bool = False):
    d = config.WORLDS / world / out
    c = pairs(world, cand)
    n = n_parts(len(c))
    outs = [d / f"part_{i:05d}.parquet" for i in range(n)]
    if stage_done(outs, log, force):
        return
    w = config.WORLDS / world
    s1n = pl.read_parquet(w / f"s1_norm{norm}.parquet", columns=["hnum", "nums"])
    recn = pl.read_parquet(w / f"rec_norm{norm}.parquet", columns=["hnum", "nums"])
    for i, path in enumerate(outs):
        if path.exists() and not force:
            continue
        with timed(log, f"{world} part {i + 1}/{n}"):
            write_parquet(chunk_numfeat(chunk(c, i), s1n, recn), path)
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--cand", default="cand_v1")
    ap.add_argument("--norm", default="_v2")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.world, a.cand, a.norm, force=a.force)
