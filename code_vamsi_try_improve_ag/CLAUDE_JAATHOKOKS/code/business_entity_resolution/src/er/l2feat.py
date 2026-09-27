"""Level-2 competition features on a stacker's predictions q (B: out-of-fold, test: 2-model average).

Aligned with the p1 parts (same part files, same row order) so `context.fit --numfeat ...,l2_<name>` can use them:
  q, logit(q)
  record (over its candidate S1s): rank, best, margin to the best OTHER S1, #q>0.5, #q>0.1, sum of q
  S1 (over its candidate records): rank, #other q>0.5 / >0.9, sum / max of other q, #candidates
  S1 "wins": other records whose argmax S1 is this S1 with q>0.5 (all / same source), their q sum

Run:  python -m er.l2feat --world B --name ctx_v6      (-> work/worlds/B/l2_ctx_v6/part_*.parquet)
"""
import argparse
import gc

import numpy as np
import polars as pl

from . import config
from .context import group_rank, rec_segments, seg_count
from .lgbm_pair import parts
from .utils import get_logger, stage_done, timed, write_parquet

log = get_logger("l2feat")


def build(world: str, name: str, force: bool = False):
    fparts = parts(world, "p1")
    d = config.WORLDS / world / f"l2_{name}"
    outs = [d / p.name for p in fparts]
    if stage_done(outs, log, force):
        return
    with timed(log, f"l2feat {world} {name}"):
        lens = [pl.scan_parquet(p).select(pl.len()).collect().item() for p in fparts]
        pr = pl.read_parquet(config.PREDS / f"{name}_{world}.parquet")
        s1, rec, q = pr["s1_idx"].to_numpy(), pr["rec_idx"].to_numpy(), pr["p"].to_numpy().astype(np.float32)
        del pr
        n, n_s1 = len(q), int(s1.max()) + 1
        assert n == sum(lens)
        qc = np.clip(q, 1e-6, 1 - 1e-6)
        cols = {"q": q, "q_logit": np.log(qc / (1 - qc)).astype(np.float32)}
        del qc
        seg, starts = rec_segments(rec)
        rank, best, other = group_rank(q, seg, 0.0)
        cols["qr_rank"] = np.minimum(rank, 32767).astype(np.int16)
        cols["qr_best"] = best
        cols["qr_margin"] = (q - other).astype(np.float32)
        cols["qr_other"] = other
        cols["qr_n05"] = seg_count(q > 0.5, seg, starts)
        cols["qr_n01"] = seg_count(q > 0.1, seg, starts)
        cols["qr_sum"] = np.add.reduceat(q.astype(np.float64), starts)[seg].astype(np.float32)
        won = (rank == 1) & (q > 0.5)
        del rank, best, other, seg, starts
        gc.collect()

        rank, _, other = group_rank(q, s1, 0.0)
        cols["qs_rank"] = np.minimum(rank, 32767).astype(np.int16)
        cols["qs_max_other"] = other
        del rank, other
        for thr, nm in [(0.5, "qs_n05_other"), (0.9, "qs_n09_other")]:
            m = q > thr
            cols[nm] = (np.bincount(s1[m], minlength=n_s1)[s1] - m).astype(np.int32)
        cols["qs_sum_other"] = (np.bincount(s1, weights=q, minlength=n_s1)[s1] - q).astype(np.float32)
        cols["qs_ncand"] = np.bincount(s1, minlength=n_s1)[s1].astype(np.int32)
        cols["qs_won_other"] = (np.bincount(s1[won], minlength=n_s1)[s1] - won).astype(np.int32)
        cols["qs_wonsum_other"] = (np.bincount(s1[won], weights=q[won], minlength=n_s1)[s1]
                                   - np.where(won, q, 0)).astype(np.float32)
        src = pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["source"])["source"].to_numpy()[rec]
        same = np.zeros(n, dtype=np.int32)
        for s in (2, 3):
            ws = won & (src == s)
            cnt = np.bincount(s1[ws], minlength=n_s1)
            m = src == s
            same[m] = cnt[s1[m]] - ws[m]
        cols["qs_won_same_src"] = same
        cols["qs_won_other_src"] = cols["qs_won_other"] - same
        del src, won, same
        gc.collect()

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
