"""Final decision: each record goes to its highest-scoring S1 (records belong to at most one S1), accepted if p >= T.

tune   on a labelled world's out-of-fold stacker predictions: sweep T over 0.50..0.99 (step 0.005) with the official
       scorer; also reports the expected-F0.5 rule (per S1, the prefix of its records by p that maximizes
       E[F] ~ 1.25 * sum_topk(p) / (0.25 * sum_all(p) + k), vs the empty set with P = prod(1 - p))
write  apply the saved rule to test -> work/preds/test_matches.parquet (s1_idx, rec_idx)

Run:  python -m er.decide tune --world B --name ctx_v6
      python -m er.decide write --world test --name ctx_v6 [--method threshold|expected]
"""
import argparse

import numpy as np
import polars as pl

from . import config
from .evaluate import report, score, truth_pairs
from .utils import get_logger, load_json, save_json, write_parquet

log = get_logger("decide")


def best_per_record(pred: pl.DataFrame) -> pl.DataFrame:
    sorted_pred = pred.sort(["rec_idx", "p"], descending=[False, True])
    ranked = sorted_pred.with_columns(rnk=pl.int_range(0, pl.len()).over("rec_idx"))
    top1 = ranked.filter(pl.col("rnk") == 0).drop("rnk")
    top2 = ranked.filter(pl.col("rnk") == 1).select("rec_idx", p2=pl.col("p"))
    return top1.join(top2, on="rec_idx", how="left").with_columns(
        margin=pl.col("p") - pl.col("p2").fill_null(0.0)
    ).select("s1_idx", "rec_idx", "p", "margin")


def threshold_rule(best: pl.DataFrame, t: float) -> pl.DataFrame:
    return best.filter(pl.col("p") >= t).select("s1_idx", "rec_idx")


def country_margin_rule(best: pl.DataFrame, s1: pl.DataFrame, params: dict) -> pl.DataFrame:
    b = best.join(s1.select("s1_idx", "country"), on="s1_idx")
    p_in = params.get("India", {"t_base": 0.925, "t_low": 0.820, "margin": 0.720})
    p_us = params.get("US", {"t_base": 0.935, "t_low": 0.860, "margin": 0.720})
    rule_in = (pl.col("country") == "India") & (
        (pl.col("p") >= p_in["t_base"]) | ((pl.col("p") >= p_in["t_low"]) & (pl.col("margin") >= p_in["margin"]))
    )
    rule_us = (pl.col("country") == "US") & (
        (pl.col("p") >= p_us["t_base"]) | ((pl.col("p") >= p_us["t_low"]) & (pl.col("margin") >= p_us["margin"]))
    )
    return b.filter(rule_in | rule_us).select("s1_idx", "rec_idx")


def expected_rule(best: pl.DataFrame) -> pl.DataFrame:
    b = (best.sort(["s1_idx", "p"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1_idx"),
                       S=pl.col("p").cum_sum().over("s1_idx"),
                       N=pl.col("p").sum().over("s1_idx"),
                       e0=pl.col("p").map_batches(lambda s: np.log1p(-np.clip(s.to_numpy(), 0, 1 - 1e-7)))
                       .sum().over("s1_idx").exp())
         .with_columns(ef=1.25 * pl.col("S") / (0.25 * pl.col("N") + pl.col("k"))))
    kbest = (b.group_by("s1_idx").agg(kb=pl.col("k").get(pl.col("ef").arg_max()), efm=pl.col("ef").max(),
                                      e0=pl.col("e0").first())
             .with_columns(kb=pl.when(pl.col("efm") > pl.col("e0")).then(pl.col("kb")).otherwise(0)))
    return b.join(kbest.select("s1_idx", "kb"), on="s1_idx").filter(pl.col("k") <= pl.col("kb")).select("s1_idx", "rec_idx")


def tune(world: str, name: str):
    pred = pl.read_parquet(config.PREDS / f"{name}_{world}.parquet")
    s1 = pl.read_parquet(config.WORLDS / world / "s1.parquet", columns=["s1_idx", "country"])
    t = truth_pairs(pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["rec_idx", "s1_idx"]))
    best = best_per_record(pred)
    sweep = {float(th): score(s1, t, threshold_rule(best, th))["f05"] for th in np.round(np.arange(0.5, 0.99001, 0.005), 3)}
    th = max(sweep, key=sweep.get)
    r = score(s1, t, threshold_rule(best, th))
    report(log, f"{world} {name} threshold T={th}", r)
    re = score(s1, t, expected_rule(best))
    report(log, f"{world} {name} expected-F0.5 rule", re)
    
    params_cm = {
        "India": {"t_base": 0.925, "t_low": 0.820, "margin": 0.720},
        "US": {"t_base": 0.935, "t_low": 0.860, "margin": 0.720}
    }
    rcm = score(s1, t, country_margin_rule(best, s1, params_cm))
    report(log, f"{world} {name} country-margin rule", rcm)

    save_json({"threshold": th, "f05_threshold": r["f05"], "f05_expected": re["f05"],
               "f05_country_margin": rcm["f05"], "params_country": params_cm,
               "sweep": sweep, "score_threshold": r, "score_expected": re, "score_country_margin": rcm},
              config.MODELS / name / "decision.json")
    log.info("DECISION %s: T=%.3f F0.5 %.5f (country-margin rule F0.5 %.5f)", name, th, r["f05"], rcm["f05"])


def write(world: str, name: str, method: str):
    dec = load_json(config.MODELS / name / "decision.json")
    s1 = pl.read_parquet(config.WORLDS / world / "s1.parquet", columns=["s1_idx", "country"])
    best = best_per_record(pl.read_parquet(config.PREDS / f"{name}_{world}.parquet"))
    if method == "country_margin":
        params = dec.get("params_country", {
            "India": {"t_base": 0.925, "t_low": 0.820, "margin": 0.720},
            "US": {"t_base": 0.935, "t_low": 0.860, "margin": 0.720}
        })
        m = country_margin_rule(best, s1, params)
    elif method == "threshold":
        m = threshold_rule(best, dec["threshold"])
    else:
        m = expected_rule(best)
    write_parquet(m, config.PREDS / f"{world}_matches.parquet")
    rate = (s1.join(m.group_by("s1_idx").len(), on="s1_idx", how="left")
            .group_by("country").agg(matched=pl.col("len").is_not_null().mean(), links=pl.col("len").fill_null(0).mean()))
    log.info("%s %s (%s): %d matches; per country %s", world, name, method, len(m), rate.rows())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "write"])
    ap.add_argument("--world", default="B")
    ap.add_argument("--name", required=True)
    ap.add_argument("--method", default="country_margin", choices=["country_margin", "threshold", "expected"])
    a = ap.parse_args()
    tune(a.world, a.name) if a.cmd == "tune" else write(a.world, a.name, a.method)
