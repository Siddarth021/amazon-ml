"""XLM-R base cross-encoder over raw "name | address" text pairs (FacebookAI/xlm-roberta-base, MIT).

pairs     ce_v3 training pairs from A: every positive in A-sample candidates + each record's top-3 wrong
          candidates by stage-1 p (any p)                          -> models/<name>/train_pairs.parquet
pseudo    ce_v4 pairs: test pseudo-labels from a stacker (positive: record's best S1 with p >= 0.995 and every
          other S1 < 0.02; negative: shortlist pairs with p <= 0.005), 1.2M sampled with France ~40%, plus 1.2M
          replayed ce_v3 A pairs                                   -> models/<name>/train_pairs.parquet
train     1 epoch, bf16 autocast, AdamW, 5% warmup + linear decay, grad clip 1.0; validation on 100k random B
          shortlist pairs every --eval-every steps; checkpoint (model, optimizer, scheduler, step, RNG) for resume;
          best-on-validation weights in models/<name>/best
predict   logits for a world's shortlist, sorted by length, resumable parts; pairs already scored are reused
                                                                   -> worlds/<w>/<name>/part_*.parquet (s1_idx, rec_idx, ce)
Run:
  python -m er.cross_encoder pairs   --name ce_v3
  python -m er.cross_encoder train   --name ce_v3
  python -m er.cross_encoder predict --name ce_v3 --world B
  python -m er.cross_encoder pseudo  --name ce_v4 --stacker ctx_v3 --replay ce_v3
  python -m er.cross_encoder train   --name ce_v4 --init ce_v3 --lr 1e-5
"""
import argparse
import gc
import math
import os
import random

import numpy as np
import polars as pl

from . import config
from .utils import get_logger, load_json, save_json, stage_done, timed, write_parquet

log = get_logger("cross_encoder")

MODEL = "FacebookAI/xlm-roberta-base"
MAX_LEN = 128
TRANSLIT_FROM = {"A": ["A"], "B": ["A"], "dev": ["A"], "test": ["A", "B"]}
KEYS = ["s1_idx", "rec_idx"]
PRED_PART = 500_000


# ----------------------------------------------------------------------------------------------- text
def texts(world: str, p: pl.DataFrame) -> tuple[list, list]:
    """Raw 'name | address' for both sides of pairs p; Indic record names get '(latin)' appended."""
    from .translit import INDIC, load_dict, to_latin
    w = config.WORLDS / world
    s1 = pl.read_parquet(w / "s1.parquet", columns=["name", "address"])
    rec = pl.read_parquet(w / "rec.parquet", columns=["name", "address"])
    need = p.select("rec_idx").unique().sort("rec_idx")["rec_idx"].to_numpy()
    r = rec[need].with_columns(rec_idx=pl.Series(need))
    ind = r["name"].str.contains(INDIC)
    if ind.any():
        lat = to_latin(r["name"], load_dict(TRANSLIT_FROM.get(world, ["A", "B"])))
        r = r.with_columns(name=pl.when(ind).then(pl.format("{} ({})", pl.col("name"), lat)).otherwise(pl.col("name")))
    r = r.select("rec_idx", t2=pl.format("{} | {}", "name", "address"))
    t1 = s1[p["s1_idx"].to_numpy()].select(t1=pl.format("{} | {}", "name", "address"))["t1"]
    t2 = p.select("rec_idx").join(r, on="rec_idx", how="left", maintain_order="left")["t2"]
    return t1.to_list(), t2.to_list()


# ----------------------------------------------------------------------------------------------- pairs
def label(world: str, p: pl.DataFrame) -> pl.DataFrame:
    t = pl.read_parquet(config.WORLDS / world / "rec.parquet", columns=["s1_idx"])["s1_idx"].fill_null(-1).to_numpy()
    return p.with_columns(label=pl.Series((t[p["rec_idx"].to_numpy()] == p["s1_idx"].to_numpy()).astype(np.float32)))


def build_pairs(name: str, force: bool = False):
    out = config.MODELS / name / "train_pairs.parquet"
    if stage_done([out], log, force):
        return
    from .lgbm_pair import load_p1
    p = label("A", load_p1("A")).with_columns(
        wrong_rank=pl.when(pl.col("label") == 0).then(pl.col("p")).otherwise(None)
        .rank("ordinal", descending=True).over("rec_idx"))
    tr = p.filter((pl.col("label") == 1) | (pl.col("wrong_rank") <= 3)).select(*KEYS, "label")
    tr = tr.sample(fraction=1.0, shuffle=True, seed=config.SEED).with_columns(world=pl.lit("A"))
    write_parquet(tr, out)
    log.info("%s: %d training pairs, %.3f positive", name, len(tr), tr["label"].mean())


def build_pseudo(name: str, stacker: str, replay: str, n_pseudo: int, n_replay: int, france_share: float,
                 force: bool = False):
    out = config.MODELS / name / "train_pairs.parquet"
    if stage_done([out], log, force):
        return
    sp = pl.read_parquet(config.PREDS / f"{stacker}_test.parquet").select(*KEYS, "p")
    ctry = pl.read_parquet(config.WORLDS / "test" / "s1.parquet", columns=["s1_idx", "country"])
    sl = pl.read_parquet(config.WORLDS / "test" / "shortlist.parquet", columns=KEYS)
    rk = sp.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("rec_idx"),
                         second=pl.col("p").sort(descending=True).get(1, null_on_oob=True).over("rec_idx"))
    pos = rk.filter((pl.col("r") == 1) & (pl.col("p") >= 0.995) & (pl.col("second").fill_null(0) < 0.02))
    neg = sl.join(sp, on=KEYS).filter(pl.col("p") <= 0.005)
    cand = pl.concat([pos.select(*KEYS).with_columns(label=pl.lit(1.0, pl.Float32)),
                      neg.select(*KEYS).with_columns(label=pl.lit(0.0, pl.Float32))]).join(ctry, on="s1_idx")
    rp = pl.read_parquet(config.MODELS / replay / "train_pairs.parquet")
    pos_rate = rp["label"].mean()
    log.info("pseudo pool: %d pos, %d neg; target positive rate %.3f (from %s)", len(pos), len(neg), pos_rate, replay)
    picks = []
    for is_fr, share in ((True, france_share), (False, 1 - france_share)):
        g = cand.filter((pl.col("country") == "France") == is_fr)
        n = int(n_pseudo * share)
        for lab, frac in ((1.0, pos_rate), (0.0, 1 - pos_rate)):
            gg = g.filter(pl.col("label") == lab)
            picks.append(gg.sample(n=min(len(gg), int(n * frac)), seed=config.SEED))
    ps = pl.concat(picks).drop("country").with_columns(world=pl.lit("test"))
    rp = rp.sample(n=min(n_replay, len(rp)), seed=config.SEED)
    tr = pl.concat([ps, rp.select(ps.columns)]).sample(fraction=1.0, shuffle=True, seed=config.SEED)
    write_parquet(tr, out)
    log.info("%s: %d pseudo (%.3f pos) + %d replay pairs", name, len(ps), ps["label"].mean(), len(rp))


# ----------------------------------------------------------------------------------------------- training
def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batches(tok, t1, t2, bs):
    for i in range(0, len(t1), bs):
        yield tok(t1[i:i + bs], t2[i:i + bs], truncation="longest_first", max_length=MAX_LEN, padding=True,
                  return_tensors="pt")


def valid_set(n: int = 100_000) -> pl.DataFrame:
    path = config.MODELS / "ce_valid_B.parquet"
    if not path.exists():
        sl = pl.read_parquet(config.WORLDS / "B" / "shortlist.parquet", columns=KEYS)
        write_parquet(label("B", sl.sample(n=min(n, len(sl)), seed=config.SEED)), path)
    return pl.read_parquet(path)


def evaluate(model, tok, t1, t2, y, bs):
    import torch
    dev = device()
    model.eval()
    logits = []
    with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16):
        for b in make_batches(tok, t1, t2, bs):
            logits.append(model(**{k: v.to(dev) for k, v in b.items()}).logits.float().squeeze(-1).cpu())
    model.train()
    z = torch.cat(logits)
    yt = torch.tensor(y)
    ll = torch.nn.functional.binary_cross_entropy_with_logits(z, yt).item()
    return ll, ((z > 0).float() == yt).float().mean().item()


def train(name: str, init: str | None, lr: float, bs: int, eval_every: int, force: bool = False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    d = config.MODELS / name
    best_dir, ckpt = d / "best", d / "checkpoint.pt"
    if stage_done([d / "done.json"], log, force):
        return
    torch.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    dev = device()
    src = str(config.MODELS / init / "best") if init else MODEL
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(src, num_labels=1).to(dev)

    tr = pl.read_parquet(d / "train_pairs.parquet")
    with timed(log, f"texts for {len(tr)} training pairs"):
        T1, T2, Y = [None] * len(tr), [None] * len(tr), tr["label"].to_numpy().astype(np.float32)
        tr = tr.with_row_index("_i")
        for w in tr["world"].unique().to_list():
            g = tr.filter(pl.col("world") == w)
            a, b = texts(w, g)
            for i, x, z in zip(g["_i"].to_list(), a, b):
                T1[i], T2[i] = x, z
    va = valid_set()
    V1, V2 = texts("B", va)
    VY = va["label"].to_numpy().astype(np.float32)

    steps = math.ceil(len(T1) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    step, best = 0, float("inf")
    if ckpt.exists():
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sch.load_state_dict(st["sch"])
        step, best = st["step"], st["best"]
        torch.set_rng_state(st["rng"])
        log.info("resumed at step %d/%d (best valid logloss %.5f)", step, steps, best)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()

    def save_ckpt():
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sch": sch.state_dict(), "step": step,
                    "best": best, "rng": torch.get_rng_state()}, tmp)
        os.replace(tmp, ckpt)

    run_loss = 0.0
    for b_i, b in enumerate(make_batches(tok, T1[step * bs:], T2[step * bs:], bs)):
        y = torch.tensor(Y[(step) * bs:(step + 1) * bs], device=dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            z = model(**{k: v.to(dev) for k, v in b.items()}).logits.float().squeeze(-1)
        loss = lossf(z, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        run_loss += loss.item()
        if step % 200 == 0:
            log.info("step %d/%d loss %.4f lr %.2e", step, steps, run_loss / 200, sch.get_last_lr()[0])
            run_loss = 0.0
        if step % eval_every == 0 or step == steps:
            ll, acc = evaluate(model, tok, V1, V2, VY, bs * 4)
            log.info("step %d valid logloss %.5f acc %.4f", step, ll, acc)
            if ll < best:
                best = ll
                model.save_pretrained(best_dir)
                tok.save_pretrained(best_dir)
            save_ckpt()
    save_json({"steps": step, "best_valid_logloss": best, "init": init, "lr": lr, "batch": bs, "pairs": len(T1)},
              d / "done.json")
    log.info("%s done: best valid logloss %.5f", name, best)


# ----------------------------------------------------------------------------------------------- inference
def predict(name: str, world: str, bs: int, force: bool = False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = config.WORLDS / world / name
    if force and d.exists():
        for f in d.glob("part_*.parquet"):
            f.unlink()
    sl = pl.read_parquet(config.WORLDS / world / "shortlist.parquet", columns=KEYS)
    done = pl.read_parquet(d / "part_*.parquet") if list(d.glob("part_*.parquet")) else None
    todo = sl.join(done, on=KEYS, how="anti") if done is not None else sl
    log.info("%s %s: %d shortlist pairs, %d to score", name, world, len(sl), len(todo))
    if len(todo):
        dev = device()
        src = config.MODELS / name / "best"
        tok = AutoTokenizer.from_pretrained(src)
        model = AutoModelForSequenceClassification.from_pretrained(src).to(dev).eval()
        first = len(list(d.glob("part_*.parquet")))
        for j, st in enumerate(range(0, len(todo), PRED_PART)):
            p = todo.slice(st, PRED_PART)
            t1, t2 = texts(world, p)
            order = np.argsort([len(a) + len(b) for a, b in zip(t1, t2)], kind="stable")
            t1, t2 = [t1[i] for i in order], [t2[i] for i in order]
            out = np.empty(len(p), dtype=np.float32)
            z = []
            with timed(log, f"{name} {world} part {first + j} ({len(p)} pairs)"), torch.inference_mode(), \
                    torch.autocast(dev.type, dtype=torch.bfloat16):
                for b in make_batches(tok, t1, t2, bs):
                    z.append(model(**{k: v.to(dev) for k, v in b.items()}).logits.float().squeeze(-1).cpu().numpy())
            out[order] = np.concatenate(z)
            write_parquet(p.with_columns(ce=pl.Series(out)), d / f"part_{first + j:05d}.parquet")
            gc.collect()


def load_ce(world: str, name: str) -> pl.DataFrame:
    """Scores of the current shortlist (older parts may hold pairs that left the shortlist)."""
    sl = pl.read_parquet(config.WORLDS / world / "shortlist.parquet", columns=KEYS)
    sc = pl.read_parquet(config.WORLDS / world / name / "part_*.parquet").unique(KEYS)
    out = sl.join(sc, on=KEYS, how="left")
    assert out["ce"].null_count() == 0, f"{name} {world}: {out['ce'].null_count()} shortlist pairs unscored"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pairs", "pseudo", "train", "predict"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--world", default="B")
    ap.add_argument("--init", default=None, help="fine-tune from this model's best weights (ce_v4: ce_v3)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pred-batch", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--stacker", default="ctx_v3")
    ap.add_argument("--replay", default="ce_v3")
    ap.add_argument("--n-pseudo", type=int, default=1_200_000)
    ap.add_argument("--n-replay", type=int, default=1_200_000)
    ap.add_argument("--france-share", type=float, default=0.4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "pairs":
        build_pairs(a.name, a.force)
    elif a.cmd == "pseudo":
        build_pseudo(a.name, a.stacker, a.replay, a.n_pseudo, a.n_replay, a.france_share, a.force)
    elif a.cmd == "train":
        train(a.name, a.init, a.lr, a.batch, a.eval_every, a.force)
    else:
        predict(a.name, a.world, a.pred_batch, a.force)
