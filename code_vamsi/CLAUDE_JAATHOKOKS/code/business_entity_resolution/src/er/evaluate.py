"""Official metric: F0.5 per S1, macro-averaged over all S1 (singletons included).

Run:  python -m er.evaluate --world B --pred <parquet with s1_idx, rec_idx>
"""
import argparse

import polars as pl

from . import config
from .utils import get_logger

log = get_logger("evaluate")

CARD_BINS = [(0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 10**9, "6+")]


def f05(tp: pl.Expr, n_pred: pl.Expr, n_true: pl.Expr) -> pl.Expr:
    p = pl.when(n_pred > 0).then(tp / n_pred).otherwise(0.0)
    r = pl.when(n_true > 0).then(tp / n_true).otherwise(0.0)
    f = pl.when(p + r > 0).then(1.25 * p * r / (0.25 * p + r)).otherwise(0.0)
    return (pl.when((n_true == 0) & (n_pred == 0)).then(1.0)
            .when(n_true == 0).then(0.0)
            .otherwise(f))


def truth_pairs(rec: pl.DataFrame) -> pl.DataFrame:
    return rec.filter(pl.col("s1_idx").is_not_null()).select("s1_idx", "rec_idx")


def per_s1(s1: pl.DataFrame, truth: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    key = ["s1_idx", "rec_idx"]
    truth = truth.select(pl.col(key).cast(pl.Int32)).unique()
    pred = pred.select(pl.col(key).cast(pl.Int32)).unique()
    tp = truth.join(pred, on=key).group_by("s1_idx").len("tp")
    nt = truth.group_by("s1_idx").len("n_true")
    npd = pred.group_by("s1_idx").len("n_pred")
    df = (s1.select(pl.col("s1_idx").cast(pl.Int32), "country")
          .join(tp, on="s1_idx", how="left").join(nt, on="s1_idx", how="left").join(npd, on="s1_idx", how="left")
          .with_columns(pl.col("tp", "n_true", "n_pred").fill_null(0).cast(pl.Int64)))
    card = pl.lit(None, dtype=pl.Utf8)
    for lo, hi, name in reversed(CARD_BINS):
        card = pl.when(pl.col("n_true").is_between(lo, hi)).then(pl.lit(name)).otherwise(card)
    p = pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")).otherwise(0.0)
    r = pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")).otherwise(0.0)
    return df.with_columns(f=f05(pl.col("tp"), pl.col("n_pred"), pl.col("n_true")), card=card, p=p, r=r)


def score(s1: pl.DataFrame, truth: pl.DataFrame, pred: pl.DataFrame) -> dict:
    df = per_s1(s1, truth, pred)
    n = len(df)
    loss = 1.0 - pl.col("f")
    fp = pl.col("n_pred") - pl.col("tp")
    fn = pl.col("n_true") - pl.col("tp")
    kind = (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).then(pl.lit("singleton_false_match"))
            .when((pl.col("n_true") > 0) & (pl.col("n_pred") == 0)).then(pl.lit("fully_missed"))
            .when(fp > 0).then(pl.lit("false_matches"))
            .when(fn > 0).then(pl.lit("missed_matches"))
            .otherwise(pl.lit("none")))
    dec = df.with_columns(kind=kind).group_by("kind").agg(loss.sum())
    decomposition = {k: 0.0 for k in ("singleton_false_match", "fully_missed", "false_matches", "missed_matches")}
    for k, v in dec.iter_rows():
        if k in decomposition:
            decomposition[k] = v / n
    return {
        "f05": df["f"].mean(),
        "precision_macro": df["p"].mean(),
        "recall_macro": df["r"].mean(),
        "n_s1": n,
        "by_country": dict(df.group_by("country").agg(pl.col("f").mean()).sort("country").iter_rows()),
        "n_by_country": dict(df.group_by("country").len().sort("country").iter_rows()),
        "by_cardinality": {c: df.filter(pl.col("card") == c)["f"].mean() for _, _, c in CARD_BINS},
        "loss": decomposition,
        "n_pred_pairs": int(df["n_pred"].sum()),
        "tp": int(df["tp"].sum()),
        "fp": int((df["n_pred"] - df["tp"]).sum()),
        "fn": int((df["n_true"] - df["tp"]).sum()),
    }


def ceiling(s1: pl.DataFrame, truth: pl.DataFrame, cand: pl.DataFrame) -> dict:
    """Score of a perfect model restricted to the candidate pairs."""
    key = ["s1_idx", "rec_idx"]
    hit = truth.select(pl.col(key).cast(pl.Int32)).join(cand.select(pl.col(key).cast(pl.Int32)).unique(), on=key)
    r = score(s1, truth, hit)
    r["pair_recall"] = len(hit) / max(len(truth), 1)
    return r


def report(log, title: str, r: dict) -> None:
    lines = [f"== {title}: F0.5 {r['f05']:.5f}  (P {r['precision_macro']:.5f} R {r['recall_macro']:.5f}, {r['n_s1']} S1)"]
    if "pair_recall" in r:
        lines.append(f"   pair recall {r['pair_recall']:.5f}")
    lines.append("   by country: " + ", ".join(f"{k} {v:.5f} (n={r['n_by_country'][k]})" for k, v in r["by_country"].items()))
    lines.append("   by cardinality: " + ", ".join(f"{k} {v:.5f}" if v is not None else f"{k} -" for k, v in r["by_cardinality"].items()))
    lines.append("   loss: " + ", ".join(f"{k} {v:.5f}" for k, v in r["loss"].items()))
    lines.append(f"   pairs pred {r['n_pred_pairs']} tp {r['tp']} fp {r['fp']} fn {r['fn']}")
    log.info("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--pred", required=True)
    a = ap.parse_args()
    d = config.WORLDS / a.world
    s1 = pl.read_parquet(d / "s1.parquet")
    rec = pl.read_parquet(d / "rec.parquet")
    report(log, f"{a.world} {a.pred}", score(s1, truth_pairs(rec), pl.read_parquet(a.pred)))
