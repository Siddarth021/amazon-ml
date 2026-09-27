"""Record-record agreement features: a candidate record vs the most confident OTHER record of the same S1.

For pairs with q > Q_MIN (q = a stacker's prediction; B: out-of-fold, test: 2-model average), other = the S1's best
record by q excluding this one (among pairs with q > Q_MIN). Features (NaN when there is no such pair / no other):
  rr_q         q of the other record
  rr_name_tset token_set_ratio(core names), rr_name_ratio ratio(core names)
  rr_addr_tset token_set_ratio(addresses)   (NaN when either address is empty)
  rr_concat    token_set_ratio(concat views), rr_same_src (other record from the same source)
Aligned with the p1 parts.

Run:  python -m er.rrfeat --world B --name ctx_v6      (-> work/worlds/B/rr_ctx_v6/part_*.parquet)
"""
import argparse

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config
from .lgbm_pair import parts
from .utils import get_logger, stage_done, timed, write_parquet

log = get_logger("rrfeat")
Q_MIN = 0.01


def build(world: str, name: str, force: bool = False):
    fparts = parts(world, "p1")
    d = config.WORLDS / world / f"rr_{name}"
    outs = [d / p.name for p in fparts]
    if stage_done(outs, log, force):
        return
    with timed(log, f"rrfeat {world} {name}"):
        lens = [pl.scan_parquet(p).select(pl.len()).collect().item() for p in fparts]
        pr = pl.read_parquet(config.PREDS / f"{name}_{world}.parquet").with_row_index("_row")
        n = len(pr)
        assert n == sum(lens)
        sel = pr.filter(pl.col("p") > Q_MIN)
        log.info("%d of %d pairs with q > %.2f", len(sel), n, Q_MIN)
        top = (sel.sort(["s1_idx", "p"], descending=[False, True])
               .group_by("s1_idx", maintain_order=True)
               .agg(r1=pl.col("rec_idx").first(), q1=pl.col("p").first(),
                    r2=pl.col("rec_idx").get(1, null_on_oob=True), q2=pl.col("p").get(1, null_on_oob=True)))
        j = (sel.join(top, on="s1_idx", how="left")
             .with_columns(o=pl.when(pl.col("r1") == pl.col("rec_idx")).then(pl.col("r2")).otherwise(pl.col("r1")),
                           oq=pl.when(pl.col("r1") == pl.col("rec_idx")).then(pl.col("q2")).otherwise(pl.col("q1")))
             .filter(pl.col("o").is_not_null())
             .select("_row", "rec_idx", "o", "oq"))
        del sel, top
        norm = (pl.read_parquet(config.WORLDS / world / "rec_norm_v2.parquet",
                                columns=["idx", "source", "core", "ad", "concat", "addr_empty"])
                .sort("idx"))
        assert (norm["idx"].to_numpy() == np.arange(len(norm))).all()
        core = norm["core"].fill_null("").to_numpy()
        ad = norm["ad"].fill_null("").to_numpy()
        cc = norm["concat"].fill_null("").to_numpy()
        src = norm["source"].to_numpy()
        emp = norm["addr_empty"].fill_null(True).to_numpy()
        del norm
        r, o = j["rec_idx"].to_numpy(), j["o"].to_numpy()
        log.info("%d pairs with another record", len(j))
        rows = j["_row"].to_numpy()
        cols = {k: np.full(n, np.nan, dtype=np.float32) for k in
                ["rr_q", "rr_name_tset", "rr_name_ratio", "rr_addr_tset", "rr_concat", "rr_same_src"]}
        cols["rr_q"][rows] = j["oq"].to_numpy()
        cols["rr_same_src"][rows] = (src[r] == src[o]).astype(np.float32)
        a, b = core[r].tolist(), core[o].tolist()
        cols["rr_name_tset"][rows] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1)
        cols["rr_name_ratio"][rows] = process.cpdist(a, b, scorer=fuzz.ratio, workers=-1)
        at = process.cpdist(ad[r].tolist(), ad[o].tolist(), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
        cols["rr_addr_tset"][rows] = np.where(emp[r] | emp[o], np.nan, at)
        cols["rr_concat"][rows] = process.cpdist(cc[r].tolist(), cc[o].tolist(), scorer=fuzz.token_set_ratio, workers=-1)
        s1, rec = pr["s1_idx"].to_numpy(), pr["rec_idx"].to_numpy()
        del pr, j
        d.mkdir(parents=True, exist_ok=True)
        off = 0
        for out, ln in zip(outs, lens):
            df = pl.DataFrame({"s1_idx": s1[off:off + ln], "rec_idx": rec[off:off + ln],
                               **{k: v[off:off + ln] for k, v in cols.items()}})
            write_parquet(df, out)
            off += ln


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="B")
    ap.add_argument("--name", default="ctx_v6")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.world, a.name, a.force)
