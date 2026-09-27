"""Load the raw TSVs and build self-contained matching worlds.

  A, B   training S1 split 50/50 by entity (seed 2026); each gets its S1s, all records matched to them and
         ALL training distractors (records matched to no S1)
  dev    10% of A's S1 + their records + 10% sample of distractors (fast iteration)
  test   test S1 and S2+S3, s1_idx null

Outputs: work/worlds/<w>/s1.parquet (s1_idx, entity_id, name, address, country)
         work/worlds/<w>/rec.parquet (rec_idx, entity_id, name, address, country, source, s1_idx)
Row order = idx.

Run:  python -m er.data [--force]
"""
import argparse

import numpy as np

import polars as pl

from . import config
from .utils import get_logger, save_json, stage_done, timed, write_parquet

log = get_logger("data")

COLS = {"entity_id": "entity_id", "business_name": "name", "business_address": "address", "country": "country"}


def read_tsv(path) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                     encoding="utf8-lossy")
    return df.rename({k: v for k, v in COLS.items() if k in df.columns}).with_columns(pl.all().fill_null(""))


def load_split(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = config.DATA / split
    s1 = read_tsv(d / f"{split}_source1.tsv").select("entity_id", "name", "address", "country")
    rec = pl.concat([
        read_tsv(d / f"{split}_source{k}.tsv").select("entity_id", "name", "address", "country")
        .with_columns(source=pl.lit(k, dtype=pl.Int8))
        for k in (2, 3)
    ])
    assert s1["entity_id"].is_unique().all(), "duplicate S1 ids"
    assert rec["entity_id"].is_unique().all(), "duplicate S2/S3 ids"
    assert rec["entity_id"].str.slice(0, 3).is_in(["S2-", "S3-"]).all()
    assert (rec["entity_id"].str.slice(1, 1).cast(pl.Int8) == rec["source"]).all(), "source prefix mismatch"
    return s1, rec


def load_truth(s1: pl.DataFrame, rec: pl.DataFrame) -> pl.DataFrame:
    gt = pl.read_csv(config.DATA / "train" / "train_ground_truth.tsv", separator="\t", quote_char=None,
                     infer_schema=False).fill_null("")
    gt = gt.select(s1_id="source1_entity_id",
                   rec_id=pl.col("matched_entity_ids").str.split(",")).explode("rec_id", empty_as_null=True)
    gt = gt.with_columns(pl.col("rec_id").str.strip_chars()).filter(pl.col("rec_id").fill_null("") != "")
    dup = gt.filter(pl.col("rec_id").is_duplicated())
    assert dup.is_empty(), f"{dup['rec_id'].n_unique()} records linked to more than one S1"
    assert gt["s1_id"].is_in(s1["entity_id"].implode()).all(), "truth S1 id missing from source1"
    assert gt["rec_id"].is_in(rec["entity_id"].implode()).all(), "truth record id missing from source2/3"
    cc = (gt.join(s1.select(s1_id="entity_id", c1="country"), on="s1_id")
          .join(rec.select(rec_id="entity_id", c2="country"), on="rec_id"))
    n_cross = cc.filter(pl.col("c1") != pl.col("c2")).height
    log.info("truth: %d links, %d S1 with >=1 match, %d cross-country links", len(gt), gt["s1_id"].n_unique(), n_cross)
    return gt


def make_world(name: str, s1: pl.DataFrame, rec: pl.DataFrame) -> dict:
    """s1: S1 rows of the world; rec: records with column s1_id (true S1 entity id or null)."""
    s1 = s1.with_row_index("s1_idx").with_columns(pl.col("s1_idx").cast(pl.Int32))
    rec = (rec.with_row_index("rec_idx").with_columns(pl.col("rec_idx").cast(pl.Int32))
           .join(s1.select(s1_id="entity_id", s1_idx="s1_idx"), on="s1_id", how="left", maintain_order="left")
           .drop("s1_id"))
    assert (rec["rec_idx"].to_numpy() == np.arange(len(rec))).all(), "rec_idx not contiguous / row order broken"
    assert len(rec) == rec["entity_id"].n_unique()
    d = config.WORLDS / name
    write_parquet(s1.select("s1_idx", "entity_id", "name", "address", "country"), d / "s1.parquet")
    write_parquet(rec.select("rec_idx", "entity_id", "name", "address", "country", "source", "s1_idx"),
                  d / "rec.parquet")
    return world_stats(name, s1, rec)


def world_stats(name: str, s1: pl.DataFrame, rec: pl.DataFrame) -> dict:
    card = (s1.select("s1_idx", "country")
            .join(rec.filter(pl.col("s1_idx").is_not_null()).group_by("s1_idx").len("k"), on="s1_idx", how="left")
            .with_columns(pl.col("k").fill_null(0)))
    bins = (card.with_columns(b=pl.when(pl.col("k") <= 2).then(pl.col("k").cast(pl.Utf8))
                              .when(pl.col("k") <= 5).then(pl.lit("3-5")).otherwise(pl.lit("6+")))
            .group_by("b").len().sort("b"))
    by_c = {}
    for c in sorted(s1["country"].unique().to_list()):
        ns1 = s1.filter(pl.col("country") == c).height
        nrec = rec.filter(pl.col("country") == c).height
        npos = rec.filter((pl.col("country") == c) & pl.col("s1_idx").is_not_null()).height
        by_c[c] = {"s1": ns1, "rec": nrec, "pos": npos, "rec_per_s1": nrec / max(ns1, 1)}
    st = {"world": name, "s1": len(s1), "rec": len(rec), "pos": int(rec["s1_idx"].is_not_null().sum()),
          "rec_per_s1": len(rec) / max(len(s1), 1), "by_country": by_c,
          "cardinality": dict(bins.iter_rows())}
    log.info("world %s: %d S1, %d rec (%.2f/S1), %d positive links; by country %s; cardinality %s",
             name, st["s1"], st["rec"], st["rec_per_s1"], st["pos"], by_c, st["cardinality"])
    return st


def run(force: bool = False):
    names = ["A", "B", "dev", "test"]
    outs = [config.WORLDS / w / f for w in names for f in ("s1.parquet", "rec.parquet")]
    if stage_done(outs, log, force):
        return
    stats = {}
    with timed(log, "train worlds"):
        s1, rec = load_split("train")
        log.info("train: %d S1, %d rec; S1 by country %s", len(s1), len(rec),
                 dict(s1.group_by("country").len().sort("country").iter_rows()))
        gt = load_truth(s1, rec)
        rec = rec.join(gt.rename({"rec_id": "entity_id"}), on="entity_id", how="left", maintain_order="left")
        is_distr = pl.col("s1_id").is_null()
        log.info("distractors: %d", rec.filter(is_distr).height)

        s1 = s1.with_columns(_u=pl.int_range(pl.len()).shuffle(seed=config.SEED) / pl.len())
        s1_a = s1.filter(pl.col("_u") < 0.5).drop("_u")
        s1_b = s1.filter(pl.col("_u") >= 0.5).drop("_u")
        for w, s1w in (("A", s1_a), ("B", s1_b)):
            recw = rec.filter(is_distr | pl.col("s1_id").is_in(s1w["entity_id"].implode()))
            stats[w] = make_world(w, s1w, recw)

        s1_dev = s1_a.sample(fraction=0.1, seed=config.SEED)
        distr = rec.filter(is_distr)
        rec_dev = pl.concat([
            rec.filter(pl.col("s1_id").is_in(s1_dev["entity_id"].implode())),
            distr.sample(fraction=0.1, seed=config.SEED),
        ])
        rec_dev = rec_dev.join(rec.select("entity_id").with_row_index("_o"), on="entity_id").sort("_o").drop("_o")
        stats["dev"] = make_world("dev", s1_dev, rec_dev)
        del s1, rec, gt, distr

    with timed(log, "test world"):
        s1t, rect = load_split("test")
        stats["test"] = make_world("test", s1t, rect.with_columns(s1_id=pl.lit(None, dtype=pl.Utf8)))
    save_json(stats, config.PREDS / "data_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    run(ap.parse_args().force)
