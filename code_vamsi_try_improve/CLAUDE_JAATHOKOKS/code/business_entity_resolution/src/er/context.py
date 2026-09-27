"""Stacker: LightGBM over stage-1 p, cross-encoder logits and competition features, for EVERY candidate pair.

context   per-pair table aligned with the feature parts (same part files, same row order):
            p1, logit(p1); per cross-encoder (suffix "", "_2", ...): ce (-20 + ce_missing=1 outside the shortlist),
            ce_rank within record, ce_rec_best, ce_s1_npos (#pairs of the S1 with ce > 0); ce_mean
            record competition on p1 (r_*) and on sigmoid(ce_mean) (rc_*): rank, #cands > 0.1, #cands > 0.5, best,
            margin to the best OTHER S1 of the record
            S1 competition on p1 (s_*): rank of the record, #other cands > 0.5, sum / max of other cands, #S2, #S3
          computed with numpy segment ops on 1-D arrays (~40 bytes per pair)
fit       B, 2-fold cross-fit by hash(S1 entity_id) % 2 -> out-of-fold p for all B pairs; early stopping on 10% held-out
          S1 of the training half; all positives + a negative fraction (weight 1/fraction) capped by --max-train-rows
predict   test: average of the two half-models

Run:  python -m er.context fit --world B --ce ce_v3 --name ctx_v3
      python -m er.context predict --world test --ce ce_v3 --name ctx_v3
      python -m er.context fit --world B --ce ce_v3,ce_v4 --numfeat numfeat,namefeat --name ctx_v6
Out:  work/preds/<name>_<world>.parquet (s1_idx, rec_idx, p)
"""
import argparse
import gc
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .cross_encoder import load_ce
from .lgbm_pair import labels, parts, true_s1
from .pairs import check_aligned
from .utils import get_logger, load_json, save_json, stage_done, timed, write_parquet

log = get_logger("context")

KEYS = ["s1_idx", "rec_idx"]
RAW = ["n_best", "n_token_set", "n_ratio", "n_jw", "a_token_set", "a_ratio", "num_jacc", "num_conflict", "hnum_eq",
       "street_eq", "region_eq", "n_idf_jacc", "a_idf_jacc", "name_freq", "rec_ncand", "nkeys"]
CE_MISSING = -20.0
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, num_threads=os.cpu_count(), seed=config.SEED, verbose=-1)
PARAMS.update(__import__("json").loads(os.environ.get("ER_CTX_PARAMS", "{}")))  # e.g. '{"learning_rate": 0.03}'
ROUNDS, EARLY = 3000, int(os.environ.get("ER_CTX_EARLY", 50))


# ----------------------------------------------------------------------------------------------- segment helpers
def rec_segments(rec: np.ndarray):
    """Candidate rows of a record are contiguous (blocking output order). Returns seg id per row and seg starts."""
    starts = np.r_[0, np.flatnonzero(np.diff(rec) != 0) + 1]
    assert np.bincount(rec[starts]).max() == 1, "a record's candidates are not contiguous"
    seg = np.repeat(np.arange(len(starts), dtype=np.int32), np.diff(np.r_[starts, len(rec)]))
    return seg, starts


def group_rank(x: np.ndarray, grp: np.ndarray, lo: float):
    """Within-group rank by x desc (1 = best), best and best-OTHER value per row, for any grouping."""
    order = np.lexsort((-x, grp))
    gs = grp[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gs) != 0) + 1]
    sizes = np.diff(np.r_[starts, len(x)])
    gid = np.repeat(np.arange(len(starts)), sizes)
    rank = np.empty(len(x), dtype=np.int32)
    rank[order] = (np.arange(len(x)) - starts[gid] + 1).astype(np.int32)
    max1 = x[order[starts]]
    max2 = np.where(sizes > 1, x[order[np.minimum(starts + 1, len(x) - 1)]], lo)
    row_gid = np.empty(len(x), dtype=np.int64)
    row_gid[order] = gid
    del order, gs, gid
    best = max1[row_gid]
    other = np.where(rank == 1, max2[row_gid], max1[row_gid]).astype(np.float32)
    return rank, best.astype(np.float32), other


def seg_count(mask: np.ndarray, seg: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(mask.astype(np.int32), starts)[seg].astype(np.int16)


def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z.astype(np.float64)))).astype(np.float32)


# ----------------------------------------------------------------------------------------------- context table
def ctx_dir(world: str, ces: list) -> "os.PathLike":
    return config.WORLDS / world / f"ctx_{'_'.join(ces) or 'none'}"


def build_context(world: str, ces: list, force: bool = False):
    fparts = parts(world, "p1")
    d = ctx_dir(world, ces)
    outs = [d / p.name for p in fparts]
    if stage_done(outs, log, force):
        return
    with timed(log, f"context {world} ({ces})"):
        lens = [pl.scan_parquet(p).select(pl.len()).collect().item() for p in fparts]
        p1 = pl.read_parquet(config.WORLDS / world / "p1" / "part_*.parquet")
        s1, rec, p = (p1["s1_idx"].to_numpy(), p1["rec_idx"].to_numpy(), p1["p"].to_numpy().astype(np.float32))
        n, n_s1 = len(p), int(s1.max()) + 1
        log.info("%d pairs", n)
        cols = {"p1": p, "p1_logit": np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).astype(np.float32)}
        seg, starts = rec_segments(rec)
        rowid = p1.select(KEYS).with_row_index("_row")
        del p1

        ce_vals = []
        for j, name in enumerate(ces):
            sfx = "" if j == 0 else f"_{j + 1}"
            sc = load_ce(world, name)
            hit = rowid.join(sc, on=KEYS)
            ce = np.full(n, CE_MISSING, dtype=np.float32)
            ce[hit["_row"].to_numpy()] = hit["ce"].to_numpy()
            miss = np.ones(n, dtype=np.int8)
            miss[hit["_row"].to_numpy()] = 0
            del hit, sc
            rank, best, _ = group_rank(ce, seg, CE_MISSING)
            cols[f"ce{sfx}"], cols[f"ce_missing{sfx}"] = ce, miss
            cols[f"ce_rank{sfx}"], cols[f"ce_rec_best{sfx}"] = rank.astype(np.int16), best
            cols[f"ce_s1_npos{sfx}"] = np.bincount(s1[ce > 0], minlength=n_s1)[s1].astype(np.int16)
            ce_vals.append((ce, miss))
        if ce_vals:
            num = sum(np.where(m == 0, c, 0.0) for c, m in ce_vals)
            den = sum((m == 0).astype(np.float32) for _, m in ce_vals)
            cols["ce_mean"] = np.where(den > 0, num / np.maximum(den, 1), CE_MISSING).astype(np.float32)
        del rowid
        gc.collect()

        for pre, x in [("r", p)] + ([("rc", sigmoid(cols["ce_mean"]))] if ce_vals else []):
            rank, best, other = group_rank(x, seg, 0.0)
            cols[f"{pre}_rank"] = rank.astype(np.int16)
            cols[f"{pre}_n01"] = seg_count(x > 0.1, seg, starts)
            cols[f"{pre}_n05"] = seg_count(x > 0.5, seg, starts)
            cols[f"{pre}_best"] = best
            cols[f"{pre}_margin"] = (x - other).astype(np.float32)
            gc.collect()

        src = pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["source"])["source"].to_numpy()[rec]
        rank, _, other = group_rank(p, s1, 0.0)
        gt05 = p > 0.5
        cols["s_rank"] = np.minimum(rank, 32767).astype(np.int16)
        cols["s_n05_other"] = (np.bincount(s1[gt05], minlength=n_s1)[s1] - gt05).astype(np.int32)
        cols["s_sum_other"] = (np.bincount(s1, weights=p, minlength=n_s1)[s1] - p).astype(np.float32)
        cols["s_max_other"] = other
        cols["s_n_s2"] = np.bincount(s1[src == 2], minlength=n_s1)[s1].astype(np.int32)
        cols["s_n_s3"] = np.bincount(s1[src == 3], minlength=n_s1)[s1].astype(np.int32)
        del src, rank, other, seg
        gc.collect()

        off = 0
        for out, ln in zip(outs, lens):
            df = pl.DataFrame({"s1_idx": s1[off:off + ln], "rec_idx": rec[off:off + ln],
                               **{k: v[off:off + ln] for k, v in cols.items()}})
            write_parquet(df, out)
            off += ln
        assert off == n


# ----------------------------------------------------------------------------------------------- assembly
def assemble(world: str, part: str, ces: list, extra: list) -> pl.DataFrame:
    w = config.WORLDS / world
    ctx = pl.read_parquet(ctx_dir(world, ces) / part)
    f = pl.read_parquet(w / "feat_v2" / part, columns=KEYS + RAW)
    check_aligned(ctx, f, f"{world} {part} ctx/feat")
    out = [ctx, f.drop(KEYS)]
    for sub in extra:
        e = pl.read_parquet(w / sub / part)
        check_aligned(ctx, e, f"{world} {part} ctx/{sub}")
        out.append(e.drop(KEYS))
    return pl.concat(out, how="horizontal")


def s1_fold(world: str) -> np.ndarray:
    ids = pl.read_parquet(config.WORLDS / world / "s1.parquet", columns=["entity_id"])["entity_id"]
    return (ids.hash(seed=config.SEED) % 2).cast(pl.Int8).to_numpy()


def s1_holdout(world: str) -> np.ndarray:
    n = pl.scan_parquet(config.WORLDS / world / "s1.parquet").select(pl.len()).collect().item()
    return np.random.default_rng(config.SEED + 1).random(n) < 0.1


def fit(world: str, ces: list, extra: list, name: str, max_rows: int, force: bool = False):
    out = config.PREDS / f"{name}_{world}.parquet"
    mdir = config.MODELS / name
    if stage_done([out, mdir / "h0.txt", mdir / "h1.txt"], log, force):
        return
    build_context(world, ces)
    names = [p.name for p in parts(world, "p1")]
    t, fold, hold = true_s1(world), s1_fold(world), s1_holdout(world)
    feats = [c for c in assemble(world, names[0], ces, extra).columns if c not in KEYS]

    npos = nneg = 0
    for pn in names:
        k = pl.read_parquet(config.WORLDS / world / "p1" / pn, columns=KEYS)
        y = labels(k, t)
        npos += int(y.sum())
        nneg += len(y) - int(y.sum())
    frac = min(0.5, max(max_rows - npos, 0) / max(nneg, 1))
    log.info("%s: %d pos, %d neg -> negative fraction %.4f (weight %.1f) per half-pair of models",
             world, npos, nneg, frac, 1 / frac)

    mdir.mkdir(parents=True, exist_ok=True)
    for h in (0, 1):
        path = mdir / f"h{h}.txt"
        if path.exists() and not force:
            continue
        rng = np.random.default_rng(PARAMS["seed"] + h)
        X, Y, V, VY = [], [], [], []
        with timed(log, f"{name} h{h}: load training half {1 - h}"):
            for pn in names:
                df = assemble(world, pn, ces, extra)
                s = df["s1_idx"].to_numpy()
                y = labels(df, t)
                tr_half = fold[s] == (1 - h)
                keep = tr_half & ((y == 1) | (rng.random(len(y)) < frac))
                Xp = df.select(feats).filter(pl.Series(keep)).to_numpy().astype(np.float32)
                hv = hold[s[keep]]
                X.append(Xp[~hv]), Y.append(y[keep][~hv]), V.append(Xp[hv]), VY.append(y[keep][hv])
                del df, Xp
                gc.collect()
        X, Y, V, VY = map(np.concatenate, (X, Y, V, VY))
        W, VW = np.where(Y == 1, 1.0, 1 / frac), np.where(VY == 1, 1.0, 1 / frac)
        log.info("h%d: train %s (%.4f pos), early-stop %s", h, X.shape, Y.mean(), V.shape)
        dtr = lgb.Dataset(X, Y, weight=W, feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(V, VY, weight=VW, reference=dtr, free_raw_data=True)
        with timed(log, f"{name} h{h} train"):
            bst = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva], valid_names=["hold"],
                            callbacks=[lgb.early_stopping(EARLY), lgb.log_evaluation(100)])
        bst.save_model(str(path.with_suffix(".tmp")), num_iteration=bst.best_iteration)
        os.replace(path.with_suffix(".tmp"), path)
        imp = sorted(zip(feats, bst.feature_importance("gain")), key=lambda z: -z[1])
        save_json({"features": feats, "best_iteration": bst.best_iteration, "best_score": bst.best_score["hold"],
                   "importance_gain": dict(imp), "neg_fraction": frac, "ces": ces, "extra": extra},
                  mdir / f"h{h}.json")
        log.info("h%d best iteration %d %s; top %s", h, bst.best_iteration, dict(bst.best_score["hold"]),
                 [f"{a}:{b:.0f}" for a, b in imp[:8]])
        del X, Y, V, VY, dtr, dva, bst
        gc.collect()

    # out-of-fold: pairs of fold-h S1 are scored by model h (trained on the other half)
    models = [lgb.Booster(model_file=str(mdir / f"h{h}.txt")) for h in (0, 1)]
    res, ll = [], []
    for pn in names:
        df = assemble(world, pn, ces, extra)
        X = df.select(feats).to_numpy().astype(np.float32)
        f_ = fold[df["s1_idx"].to_numpy()]
        pr = np.empty(len(df), dtype=np.float32)
        for h in (0, 1):
            m = f_ == h
            if m.any():
                pr[m] = models[h].predict(X[m], num_threads=os.cpu_count())
        y = labels(df, t)
        ll.append((-(y * np.log(np.clip(pr, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - pr, 1e-7, 1)))).sum())
        res.append(df.select(KEYS).with_columns(p=pl.Series(pr)))
        del df, X
        gc.collect()
    oof = pl.concat(res)
    write_parquet(oof, out)
    log.info("%s OOF logloss %.5f on %d pairs", name, sum(ll) / len(oof), len(oof))


def predict(world: str, ces: list, extra: list, name: str, force: bool = False):
    out = config.PREDS / f"{name}_{world}.parquet"
    if stage_done([out], log, force):
        return
    build_context(world, ces)
    mdir = config.MODELS / name
    feats = load_json(mdir / "h0.json")["features"]
    models = [lgb.Booster(model_file=str(mdir / f"h{h}.txt")) for h in (0, 1)]
    res = []
    for pn in [p.name for p in parts(world, "p1")]:
        df = assemble(world, pn, ces, extra)
        X = df.select(feats).to_numpy().astype(np.float32)
        pr = np.mean([m.predict(X, num_threads=os.cpu_count()) for m in models], axis=0).astype(np.float32)
        res.append(df.select(KEYS).with_columns(p=pl.Series(pr)))
        del df, X
        gc.collect()
    write_parquet(pl.concat(res), out)
    log.info("%s %s predicted", name, world)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["context", "fit", "predict"])
    ap.add_argument("--world", default="B")
    ap.add_argument("--ce", default="ce_v3", help="comma-separated cross-encoder names ('' for none)")
    ap.add_argument("--numfeat", default="", help="extra aligned feature dirs, e.g. numfeat,namefeat")
    ap.add_argument("--name", required=False)
    ap.add_argument("--max-train-rows", type=int, default=20_000_000)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ces = [c for c in a.ce.split(",") if c]
    extra = [c for c in a.numfeat.split(",") if c]
    if a.cmd == "context":
        build_context(a.world, ces, a.force)
    elif a.cmd == "fit":
        fit(a.world, ces, extra, a.name, a.max_train_rows, a.force)
    else:
        predict(a.world, ces, extra, a.name, a.force)
