"""Decision v2: each record -> its highest-scoring S1; accepted if p >= T1 when it is its S1's top record, else p >= T2.

(T1, T2) are tuned per country on a labelled world's out-of-fold predictions (fast numpy F0.5 scorer, coordinate
search); countries without labelled data (e.g. France) use the global (T1, T2) tuned on all S1.

Run:  python -m er.decide2 tune  --world B    --name ctx_v7
      python -m er.decide2 write --world test --name ctx_v7
"""
import argparse

import numpy as np
import polars as pl

from . import config
from .decide import best_per_record
from .evaluate import report, score, truth_pairs
from .utils import get_logger, load_json, save_json, write_parquet

log = get_logger("decide2")
GRID = np.round(np.arange(0.30, 0.99001, 0.005), 3)


def prep(best: pl.DataFrame) -> pl.DataFrame:
    """Adds the rank of each record within its S1 (by p desc) -> 'top' flag."""
    return (best.sort(["s1_idx", "p"], descending=[False, True])
            .with_columns(top=(pl.int_range(pl.len()).over("s1_idx") == 0)))


def apply(b: pl.DataFrame, s1c: pl.DataFrame, th: dict) -> pl.DataFrame:
    g = th["__global__"]
    t = (b.join(s1c, on="s1_idx", how="left")
         .with_columns(t1=pl.col("country").replace_strict({k: v[0] for k, v in th.items()}, default=g[0]),
                       t2=pl.col("country").replace_strict({k: v[1] for k, v in th.items()}, default=g[1])))
    keep = pl.when(pl.col("top")).then(pl.col("p") >= pl.col("t1")).otherwise(pl.col("p") >= pl.col("t2"))
    return t.filter(keep).select("s1_idx", "rec_idx")


class Fast:
    """F0.5 macro over a set of S1 with numpy bincounts."""

    def __init__(self, b: pl.DataFrame, t: np.ndarray, n_true: np.ndarray, s1_mask: np.ndarray):
        s = b["s1_idx"].to_numpy()
        m = s1_mask[s]
        self.s = s[m]
        self.p = b["p"].to_numpy()[m]
        self.top = b["top"].to_numpy()[m]
        self.y = (t[b["rec_idx"].to_numpy()[m]] == self.s).astype(np.float64)
        self.n_true = n_true
        self.mask = s1_mask
        self.n = int(s1_mask.sum())

    def f(self, t1: float, t2: float) -> float:
        acc = np.where(self.top, self.p >= t1, self.p >= t2)
        ns = len(self.n_true)
        tp = np.bincount(self.s[acc], weights=self.y[acc], minlength=ns)
        npred = np.bincount(self.s[acc], minlength=ns).astype(np.float64)
        nt = self.n_true
        with np.errstate(divide="ignore", invalid="ignore"):
            P = np.where(npred > 0, tp / npred, 0.0)
            R = np.where(nt > 0, tp / nt, 0.0)
            F = np.where(P + R > 0, 1.25 * P * R / (0.25 * P + R), 0.0)
        F = np.where(nt == 0, (npred == 0).astype(np.float64), F)
        return float(F[self.mask].sum() / self.n)

    def search(self) -> tuple:
        best = (0.9, 0.9)
        fb = self.f(*best)
        for _ in range(3):  # coordinate search
            for i in (0, 1):
                for v in GRID:
                    c = (v, best[1]) if i == 0 else (best[0], v)
                    fv = self.f(*c)
                    if fv > fb:
                        fb, best = fv, c
        return (float(best[0]), float(best[1])), fb


def tune(world: str, name: str):
    pred = pl.read_parquet(config.PREDS / f"{name}_{world}.parquet")
    s1c = pl.read_parquet(config.WORLDS / world / "s1.parquet", columns=["s1_idx", "country"])
    rec = pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["rec_idx", "s1_idx"])
    tp_ = truth_pairs(rec)
    t = rec["s1_idx"].fill_null(-1).to_numpy()
    n_s1 = len(s1c)
    n_true = np.bincount(t[t >= 0], minlength=n_s1).astype(np.float64)
    b = prep(best_per_record(pred))
    country = s1c.sort("s1_idx")["country"].to_numpy()
    th = {}
    g, fg = Fast(b, t, n_true, np.ones(n_s1, bool)).search()
    th["__global__"] = g
    log.info("global T1=%.3f T2=%.3f fast-F %.5f", g[0], g[1], fg)
    for c in np.unique(country):
        tc, fc = Fast(b, t, n_true, country == c).search()
        th[str(c)] = tc
        log.info("%s T1=%.3f T2=%.3f fast-F %.5f", c, tc[0], tc[1], fc)
    r = score(s1c, tp_, apply(b, s1c, th))
    report(log, f"{world} {name} decide2 per-country (T1,T2)", r)
    rg = score(s1c, tp_, apply(b, s1c, {"__global__": g}))
    report(log, f"{world} {name} decide2 global (T1,T2)", rg)
    save_json({"thresholds": th, "f05": r["f05"], "f05_global": rg["f05"], "score": r},
              config.MODELS / name / "decision2.json")


def write(world: str, name: str, out_name: str):
    th = {k: tuple(v) for k, v in load_json(config.MODELS / name / "decision2.json")["thresholds"].items()}
    s1c = pl.read_parquet(config.WORLDS / world / "s1.parquet", columns=["s1_idx", "country"])
    b = prep(best_per_record(pl.read_parquet(config.PREDS / f"{name}_{world}.parquet")))
    m = apply(b, s1c, th)
    write_parquet(m, config.PREDS / out_name)
    rate = (s1c.join(m.group_by("s1_idx").len(), on="s1_idx", how="left")
            .group_by("country").agg(matched=pl.col("len").is_not_null().mean(), links=pl.col("len").fill_null(0).mean()))
    log.info("%s %s decide2 %s: %d matches; per country %s", world, name, th, len(m), rate.rows())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "write"])
    ap.add_argument("--world", default="B")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="test_matches.parquet")
    a = ap.parse_args()
    tune(a.world, a.name) if a.cmd == "tune" else write(a.world, a.name, a.out)
