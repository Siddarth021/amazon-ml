"""Initials features: record concat == initials of the S1 name ("PF" <-> "Pecheurs & Freres") and the reverse.

  init_r2s  rec concat == initials of S1 core tokens, or of all folded S1 name tokens (stop words kept)
  init_s2r  S1 concat == initials of the record's core / all name tokens
Both require length >= 2. Same chunking as features.py.
Output: work/worlds/<w>/namefeat/part_{i:05d}.parquet
Run:    python -m er.namefeat --world B
"""
import argparse
import gc

import polars as pl

from . import config
from .normalize import fold
from .pairs import chunk, n_parts, pairs
from .utils import get_logger, stage_done, timed, write_parquet

log = get_logger("namefeat")


def initials(tok: pl.Expr) -> pl.Expr:
    return tok.list.eval(pl.element().str.slice(0, 1)).list.join("")


def side(world: str, which: str, norm: str) -> pl.DataFrame:
    w = config.WORLDS / world
    raw = pl.read_parquet(w / f"{which}.parquet", columns=["name"])
    nrm = pl.read_parquet(w / f"{which}_norm{norm}.parquet", columns=["concat", "core_tok"])
    all_tok = (fold("name").str.replace_all("&", " ").str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
               .str.split(" ").list.eval(pl.element().filter(pl.element() != "")))
    return pl.concat([nrm, raw.select(all_tok=all_tok)], how="horizontal").select(
        "concat", ini_core=initials(pl.col("core_tok")), ini_all=initials(pl.col("all_tok")))


def chunk_namefeat(c: pl.DataFrame, s: pl.DataFrame, r: pl.DataFrame) -> pl.DataFrame:
    a = s[c["s1_idx"].to_numpy()]
    b = r[c["rec_idx"].to_numpy()]
    df = pl.concat([a.rename(lambda x: f"s_{x}"), b.rename(lambda x: f"r_{x}")], how="horizontal")

    def match(cat: str, ini: str) -> pl.Expr:
        return ((pl.col(cat).str.len_chars() >= 2) & (pl.col(cat) == pl.col(ini))).fill_null(False)

    g = df.select(
        init_r2s=(match("r_concat", "s_ini_core") | match("r_concat", "s_ini_all")).cast(pl.Int8),
        init_s2r=(match("s_concat", "r_ini_core") | match("s_concat", "r_ini_all")).cast(pl.Int8),
    )
    return pl.concat([c.select("s1_idx", "rec_idx"), g], how="horizontal")


def run(world: str, cand: str = "cand_v1", norm: str = "_v2", out: str = "namefeat", force: bool = False):
    d = config.WORLDS / world / out
    c = pairs(world, cand)
    n = n_parts(len(c))
    outs = [d / f"part_{i:05d}.parquet" for i in range(n)]
    if stage_done(outs, log, force):
        return
    # an Indic raw name gives Indic "all" initials that never match; its core initials are transliterated
    s, r = side(world, "s1", norm), side(world, "rec", norm)
    for i, path in enumerate(outs):
        if path.exists() and not force:
            continue
        with timed(log, f"{world} part {i + 1}/{n}"):
            write_parquet(chunk_namefeat(chunk(c, i), s, r), path)
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--cand", default="cand_v1")
    ap.add_argument("--norm", default="_v2")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.world, a.cand, a.norm, force=a.force)
