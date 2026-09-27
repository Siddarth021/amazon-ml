"""Stage-1 pair model: LightGBM on features.py columns.

train      A's 20% sample: all positives + a fraction of negatives (weight 1/fraction; 0.5 -> 2.0 as in the
           reference, lowered automatically so the matrix fits in --max-train-rows), early stopping on B parts,
           checkpoint every 100 rounds (resumable).
predict    p for every candidate pair of a world -> work/worlds/<w>/p1/part_*.parquet (s1_idx, rec_idx, p)
shortlist  p >= 0.01 and rank <= 3 within the record -> work/worlds/<w>/shortlist.parquet
check      B only: argmax per record + threshold sweep with the official scorer (stage-1-only F0.5)

Run:  python -m er.lgbm_pair train
      python -m er.lgbm_pair predict --world B
      python -m er.lgbm_pair shortlist --world B
      python -m er.lgbm_pair check --world B
"""
import argparse
import gc
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .evaluate import report, score, truth_pairs
from .utils import get_logger, load_json, save_json, stage_done, timed, write_parquet

log = get_logger("lgbm_pair")

NAME = "lgbm_pair"
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, max_bin=255, num_threads=os.cpu_count(), seed=config.SEED,
              verbose=-1)
ROUNDS, EARLY, CKPT_EVERY = 2000, 50, 100
KEYS = ["s1_idx", "rec_idx"]


def parts(world: str, sub: str = "feat_v2") -> list:
    ps = sorted((config.WORLDS / world / sub).glob("part_*.parquet"))
    assert ps, f"no {sub} parts for {world}"
    return ps


def true_s1(world: str) -> np.ndarray:
    return pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["s1_idx"])["s1_idx"].fill_null(-1).to_numpy()


def labels(df: pl.DataFrame, t: np.ndarray) -> np.ndarray:
    return (t[df["rec_idx"].to_numpy()] == df["s1_idx"].to_numpy()).astype(np.float32)


def load_xy(world: str, feats: list | None, neg_frac: float | None, max_rows: int | None, n_parts: int | None = None):
    t = true_s1(world)
    ps = parts(world)[:n_parts] if n_parts else parts(world)
    if neg_frac is None:
        npos = nneg = 0
        for p in ps:
            y = labels(pl.read_parquet(p, columns=KEYS), t)
            npos += int(y.sum())
            nneg += len(y) - int(y.sum())
        neg_frac = min(0.5, max(max_rows - npos, 0) / max(nneg, 1)) if max_rows else 0.5
        log.info("%s: %d positives, %d negatives -> negative fraction %.3f (weight %.2f)",
                 world, npos, nneg, neg_frac, 1 / neg_frac)
    rng = np.random.default_rng(config.SEED)
    X, Y = [], []
    for p in ps:
        df = pl.read_parquet(p)
        y = labels(df, t)
        keep = (y == 1) | (rng.random(len(y)) < neg_frac)
        feats = feats or [c for c in df.columns if c not in KEYS]
        X.append(df.select(feats).filter(pl.Series(keep)).to_numpy().astype(np.float32))
        Y.append(y[keep])
        del df
        gc.collect()
    X, Y = np.concatenate(X), np.concatenate(Y)
    w = np.where(Y == 1, 1.0, 1.0 / neg_frac).astype(np.float32)
    return X, Y, w, feats


def train(max_rows: int, force: bool = False):
    d = config.MODELS / NAME
    out = d / "model.txt"
    if stage_done([out], log, force):
        return
    d.mkdir(parents=True, exist_ok=True)
    ckpt = d / "checkpoint.txt"
    with timed(log, "load A-sample training matrix"):
        X, Y, W, feats = load_xy("A", None, None, max_rows)
    with timed(log, "load B validation parts"):
        Xv, Yv, Wv, _ = load_xy("B", feats, 1.0, None, n_parts=2)
    log.info("train %s (%.4f pos), valid %s, %d features", X.shape, Y.mean(), Xv.shape, len(feats))
    dtr = lgb.Dataset(X, Y, weight=W, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(Xv, Yv, reference=dtr, feature_name=feats, free_raw_data=True)
    init, done = None, 0
    if ckpt.exists():
        init = str(ckpt)
        done = lgb.Booster(model_file=init).current_iteration()
        log.info("resuming from checkpoint at round %d", done)

    def save_ckpt(env):
        if (env.iteration + 1) % CKPT_EVERY == 0:
            tmp = ckpt.with_suffix(".tmp")
            env.model.save_model(str(tmp))
            os.replace(tmp, ckpt)

    with timed(log, "LightGBM train"):
        bst = lgb.train(PARAMS, dtr, num_boost_round=ROUNDS - done, valid_sets=[dva], valid_names=["B"],
                        init_model=init, callbacks=[lgb.early_stopping(EARLY), lgb.log_evaluation(50), save_ckpt])
    tmp = out.with_suffix(".tmp")
    bst.save_model(str(tmp), num_iteration=bst.best_iteration)
    os.replace(tmp, out)
    imp = sorted(zip(feats, bst.feature_importance("gain")), key=lambda x: -x[1])
    save_json({"features": feats, "best_iteration": bst.best_iteration, "best_score": bst.best_score["B"],
               "importance_gain": dict(imp)}, d / "meta.json")
    log.info("best iteration %d, valid %s; top features %s", bst.best_iteration, dict(bst.best_score["B"]),
             [f for f, _ in imp[:10]])


def predict(world: str, force: bool = False):
    bst = lgb.Booster(model_file=str(config.MODELS / NAME / "model.txt"))
    feats = load_json(config.MODELS / NAME / "meta.json")["features"]
    for p in parts(world):
        out = config.WORLDS / world / "p1" / p.name
        if out.exists() and not force:
            continue
        df = pl.read_parquet(p)
        pr = bst.predict(df.select(feats).to_numpy().astype(np.float32), num_threads=os.cpu_count())
        write_parquet(df.select(KEYS).with_columns(p=pl.Series(pr, dtype=pl.Float32)), out)
        log.info("%s %s: %d pairs, mean p %.4f", world, p.name, len(df), pr.mean())
        del df
        gc.collect()


def load_p1(world: str) -> pl.DataFrame:
    return pl.read_parquet(config.WORLDS / world / "p1" / "part_*.parquet")


def shortlist(world: str, pmin: float = 0.01, top: int = 3, force: bool = False):
    out = config.WORLDS / world / "shortlist.parquet"
    if stage_done([out], log, force):
        return
    p = load_p1(world).with_columns(rank=pl.col("p").rank("ordinal", descending=True).over("rec_idx").cast(pl.Int16))
    sl = p.filter((pl.col("p") >= pmin) & (pl.col("rank") <= top))
    write_parquet(sl, out)
    log.info("%s shortlist: %d of %d pairs (%.2f per record with any)", world, len(sl), len(p),
             len(sl) / max(sl["rec_idx"].n_unique(), 1))
    if world != "test":
        t = truth_pairs(pl.read_parquet(config.WORLDS / world / "rec.parquet"))
        log.info("%s shortlist recall of true pairs: %.5f (candidates: %.5f)", world,
                 t.join(sl, on=KEYS).height / len(t), t.join(p, on=KEYS).height / len(t))


def best_per_record(p: pl.DataFrame, col: str = "p") -> pl.DataFrame:
    return p.sort([ "rec_idx", col], descending=[False, True]).group_by("rec_idx", maintain_order=True).first()


def threshold_sweep(world: str, best: pl.DataFrame, col: str = "p", grid=None) -> tuple[float, dict]:
    s1 = pl.read_parquet(config.WORLDS / world / "s1.parquet", columns=["s1_idx", "country"])
    t = truth_pairs(pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["rec_idx", "s1_idx"]))
    grid = grid if grid is not None else np.round(np.arange(0.50, 0.9901, 0.005), 3)
    res = {float(th): score(s1, t, best.filter(pl.col(col) >= th))["f05"] for th in grid}
    th = max(res, key=res.get)
    return th, res


def check(world: str):
    best = best_per_record(load_p1(world))
    th, res = threshold_sweep(world, best, grid=np.round(np.arange(0.30, 0.9901, 0.02), 3))
    s1 = pl.read_parquet(config.WORLDS / world / "s1.parquet", columns=["s1_idx", "country"])
    t = truth_pairs(pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["rec_idx", "s1_idx"]))
    r = score(s1, t, best.filter(pl.col("p") >= th))
    report(log, f"{world} stage-1 only (argmax, T={th})", r)
    save_json({"threshold": th, "sweep": res, "score": r}, config.PREDS / f"stage1_{world}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict", "shortlist", "check"])
    ap.add_argument("--world", default="B")
    ap.add_argument("--max-train-rows", type=int, default=20_000_000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.max_train_rows, a.force)
    elif a.cmd == "predict":
        predict(a.world, a.force)
    elif a.cmd == "shortlist":
        shortlist(a.world, force=a.force)
    else:
        check(a.world)
