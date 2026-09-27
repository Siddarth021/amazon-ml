"""Submission files + checks + official validator.

output/matching_results.tsv  source1_entity_id \t matched_entity_ids    one row per test S1, file order
output/candidate_pairs.tsv   source1_entity_id \t candidate_entity_ids  the exact candidate set the stacker scored

Run:  python -m er.write_output [--cand cand_v1] [--matches work/preds/test_matches.parquet]
"""
import argparse
import os
import subprocess
import sys

import polars as pl

from . import config
from .utils import get_logger, timed

log = get_logger("write_output")

S1_BLOCK = 250_000  # S1 rows per block when writing candidate_pairs.tsv (bounds RAM)


def id_lists(pairs: pl.DataFrame, s1: pl.DataFrame, rec_ids: pl.Series, col: str) -> pl.DataFrame:
    g = pairs.unique(["s1_idx", "rec_idx"]).sort(["s1_idx", "rec_idx"])
    g = g.with_columns(rid=rec_ids.gather(g["rec_idx"]))
    g = g.group_by("s1_idx", maintain_order=True).agg(pl.col("rid").str.join(",").alias(col))
    return (s1.select("s1_idx", source1_entity_id="entity_id").join(g, on="s1_idx", how="left", maintain_order="left")
            .select("source1_entity_id", pl.col(col).fill_null("")))


def write_tsv(df: pl.DataFrame, f, header: bool):
    txt = df.write_csv(separator="\t", quote_style="never", include_header=header, line_terminator="\n")
    f.write(txt)


def run(cand: str, matches: str, validate: bool = True):
    w = config.WORLDS / "test"
    s1 = pl.read_parquet(w / "s1.parquet", columns=["s1_idx", "entity_id"])
    rec_ids = pl.read_parquet(w / "rec.parquet", columns=["entity_id"])["entity_id"]
    m = pl.read_parquet(matches).select(pl.col("s1_idx", "rec_idx").cast(pl.Int32))
    assert m["rec_idx"].is_unique().all(), "a record matched to more than one S1"
    cand_path = w / f"{cand}.parquet"
    miss = m.join(pl.scan_parquet(cand_path).select("s1_idx", "rec_idx").collect(), on=["s1_idx", "rec_idx"], how="anti")
    assert miss.is_empty(), f"{len(miss)} matched pairs are not candidates"

    config.OUTPUT.mkdir(parents=True, exist_ok=True)
    mpath, cpath = config.OUTPUT / "matching_results.tsv", config.OUTPUT / "candidate_pairs.tsv"
    with timed(log, "matching_results.tsv"):
        out = id_lists(m, s1, rec_ids, "matched_entity_ids")
        assert len(out) == len(s1)
        with open(mpath.with_suffix(".tmp"), "w", encoding="utf-8", newline="\n") as f:
            write_tsv(out, f, True)
        os.replace(mpath.with_suffix(".tmp"), mpath)
        log.info("%d S1, %d with matches, %d links", len(out), (out["matched_entity_ids"] != "").sum(), len(m))

    with timed(log, "candidate_pairs.tsv"):
        n_pairs = 0
        with open(cpath.with_suffix(".tmp"), "w", encoding="utf-8", newline="\n") as f:
            for lo in range(0, len(s1), S1_BLOCK):
                hi = min(lo + S1_BLOCK, len(s1))
                c = (pl.scan_parquet(cand_path).select("s1_idx", "rec_idx")
                     .filter(pl.col("s1_idx").is_between(lo, hi - 1)).collect())
                n_pairs += len(c)
                write_tsv(id_lists(c, s1.slice(lo, hi - lo), rec_ids, "candidate_entity_ids"), f, lo == 0)
        os.replace(cpath.with_suffix(".tmp"), cpath)
        log.info("candidates: %d pairs, %.2f per S1", n_pairs, n_pairs / len(s1))

    if validate:
        v = config.ROOT / "utils" / "validate_submission.py"
        if v.exists():
            r = subprocess.run([sys.executable, str(v), "--matching", str(mpath), "--candidate", str(cpath),
                                "--test-dir", str(config.DATA / "test")], capture_output=True, text=True)
            log.info("validator:\n%s%s", r.stdout, r.stderr)
            assert r.returncode == 0, "validator FAILED"
        else:
            log.warning("utils/validate_submission.py not found - run the official validator yourself")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="cand_v1")
    ap.add_argument("--matches", default=str(config.PREDS / "test_matches.parquet"))
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    run(a.cand, a.matches, not a.no_validate)
