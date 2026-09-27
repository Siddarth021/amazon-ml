"""Average of aligned stacker predictions (same pair order): work/preds/<out>_<world>.parquet (s1_idx, rec_idx, p).

Run:  python -m er.blend --world B --names ctx_v7,ctx_v8 --out ctx_b78 [--logit]
"""
import argparse

import numpy as np
import polars as pl

from . import config
from .pairs import check_aligned
from .utils import get_logger, write_parquet

log = get_logger("blend")


def blend(world: str, names: list, out: str, logit: bool):
    base = pl.read_parquet(config.PREDS / f"{names[0]}_{world}.parquet")
    acc = np.zeros(len(base), dtype=np.float64)
    for nm in names:
        d = pl.read_parquet(config.PREDS / f"{nm}_{world}.parquet")
        check_aligned(base, d, f"{world} {names[0]}/{nm}")
        p = np.clip(d["p"].to_numpy().astype(np.float64), 1e-7, 1 - 1e-7)
        acc += np.log(p / (1 - p)) if logit else p
        del d
    acc /= len(names)
    p = 1 / (1 + np.exp(-acc)) if logit else acc
    (config.MODELS / out).mkdir(parents=True, exist_ok=True)
    write_parquet(base.select("s1_idx", "rec_idx").with_columns(p=pl.Series(p.astype(np.float32))),
                  config.PREDS / f"{out}_{world}.parquet")
    log.info("%s %s = mean(%s)%s", out, world, names, " in logit space" if logit else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="B")
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logit", action="store_true")
    a = ap.parse_args()
    blend(a.world, a.names.split(","), a.out, a.logit)
