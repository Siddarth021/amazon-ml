"""The pair set every per-pair stage iterates over, in fixed chunks (so feature parts align row by row).

A: candidates of a 20% random sample of A's records (seed 2026); B, test, dev: all candidates.
rec_ncand / s1_ncand are counted on the full candidate set before sampling.
"""
import numpy as np
import polars as pl

from . import config


def pairs(world: str, cand: str = "cand_v1") -> pl.DataFrame:
    c = pl.read_parquet(config.WORLDS / world / f"{cand}.parquet")
    c = c.with_columns(rec_ncand=pl.len().over("rec_idx").cast(pl.Int16),
                       s1_ncand=pl.len().over("s1_idx").cast(pl.Int32))
    if world == "A":
        n_rec = pl.scan_parquet(config.WORLDS / world / "rec.parquet").select(pl.len()).collect().item()
        rng = np.random.default_rng(config.SEED)
        keep = pl.Series(np.flatnonzero(rng.random(n_rec) < config.A_SAMPLE).astype(np.int32))
        c = c.filter(pl.col("rec_idx").is_in(keep.implode()))
    return c


def n_parts(n: int) -> int:
    return -(-n // config.FEAT_CHUNK)


def chunk(c: pl.DataFrame, i: int) -> pl.DataFrame:
    return c.slice(i * config.FEAT_CHUNK, config.FEAT_CHUNK)


def check_aligned(a: pl.DataFrame, b: pl.DataFrame, what: str) -> None:
    assert len(a) == len(b), f"{what}: {len(a)} vs {len(b)} rows"
    assert (a["s1_idx"] == b["s1_idx"]).all() and (a["rec_idx"] == b["rec_idx"]).all(), f"{what}: keys misaligned"
