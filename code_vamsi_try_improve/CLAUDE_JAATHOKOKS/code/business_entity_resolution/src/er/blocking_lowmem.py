"""Low-RAM driver for blocking.py (which stays as is): same candidates, one subprocess per shard.

blocking.run keeps every shard in one process; on a 16 GB machine the heap grows until allocation fails. Here each
(country, shard) is built by blocking.build in a fresh process with the SAME slicing as blocking.run, written to
work/worlds/<w>/<out>_shards/, skipped if it exists (resumable), then streamed into <out>.parquet in blocking.run's
order. The recall / ceiling report is computed from truth ∩ candidates shard by shard (the ceiling only depends on
that intersection), giving the same numbers as blocking.recall_report.

Run:  python -m er.blocking_lowmem --world B --norm _v2 --out cand_v1 --shard 120000 --tf-chunk 2000 --max-cand 50
"""
import argparse
import subprocess
import sys

import polars as pl

from . import blocking, config
from .evaluate import ceiling, report, truth_pairs
from .utils import get_logger, load_json, save_json, stage_done, timed

log = get_logger("blocking")


def shard_plan(world: str, norm: str, shard: int) -> list[tuple[str, int, int]]:
    recn = pl.scan_parquet(config.WORLDS / world / f"rec_norm{norm}.parquet").select("country")
    counts = dict(recn.group_by("country").len().collect().iter_rows())
    s1c = pl.scan_parquet(config.WORLDS / world / f"s1_norm{norm}.parquet").select("country").unique().collect()
    plan = []
    for c in sorted(s1c["country"].to_list()):
        n_sh = max(1, -(-counts.get(c, 0) // shard))
        plan += [(c, i, n_sh) for i in range(n_sh)]
    return plan


def build_one(world: str, norm: str, c: str, i: int, n_sh: int, path):
    d = config.WORLDS / world
    s1n = pl.read_parquet(d / f"s1_norm{norm}.parquet")
    sc = s1n.filter(pl.col("country") == c)
    del s1n
    recn = pl.read_parquet(d / f"rec_norm{norm}.parquet")
    rc = recn.filter(pl.col("country") == c)
    del recn
    afreq, tfreq = blocking.freqs(sc, rc)
    n = len(rc)
    log.info("country %s shard %d/%d", c, i + 1, n_sh)
    cc, st = blocking.build(sc, rc.slice(i * n // n_sh, (i + 1) * n // n_sh - i * n // n_sh), afreq, tfreq)
    tmp = path.with_suffix(".tmp")
    cc.write_parquet(tmp)
    tmp.replace(path)
    save_json(st, path.with_suffix(".json"))


def report_lowmem(world: str, norm: str, files: list, out_name: str):
    d = config.WORLDS / world
    s1 = pl.read_parquet(d / "s1.parquet")
    rec = pl.read_parquet(d / "rec.parquet", columns=["rec_idx", "s1_idx", "name", "address"])
    truth = truth_pairs(rec)
    hits, n_pairs, stats = [], 0, {}
    for f in files:
        c = pl.read_parquet(f, columns=["s1_idx", "rec_idx", "keys"])
        n_pairs += len(c)
        hits.append(truth.join(c, on=["s1_idx", "rec_idx"]))
        for k, v in load_json(f.with_suffix(".json")).items():
            stats[k] = stats.get(k, 0) + v
    hit = pl.concat(hits)
    tk = truth.join(hit, on=["s1_idx", "rec_idx"], how="left")
    recall = tk["keys"].is_not_null().mean()
    only = {name: tk.filter(pl.col("keys") == (1 << j)).height / len(truth) for j, name in enumerate(stats)}
    by_c = (tk.join(s1.select("s1_idx", "country"), on="s1_idx").group_by("country")
            .agg(pl.col("keys").is_not_null().mean()).sort("country").rows())
    n_rec = len(rec)
    s1c = s1.group_by("country").len("a")
    rcc = pl.read_parquet(d / "rec.parquet", columns=["country"]).group_by("country").len("b")
    n_all = s1c.join(rcc, on="country").select((pl.col("a").cast(pl.Float64) * pl.col("b")).sum()).item()
    out = {"recall": recall, "recall_by_country": dict(by_c), "pairs": n_pairs, "pairs_per_rec": n_pairs / n_rec,
           "reduction_ratio": 1 - n_pairs / n_all, "only_this_key": only, "pairs_by_key": stats}
    log.info("RECALL %.5f  by country %s  pairs %d (%.2f per rec)  reduction ratio %.6f",
             recall, {c: round(v, 5) for c, v in by_c}, n_pairs, out["pairs_per_rec"], out["reduction_ratio"])
    log.info("true pairs found ONLY by key: %s", {n: round(v, 5) for n, v in only.items()})
    r = ceiling(s1, truth, hit)
    report(log, f"{world} blocking CEILING (perfect model on these candidates)", r)
    out["ceiling"] = r
    miss = (truth.join(hit, on=["s1_idx", "rec_idx"], how="anti").sample(min(15, len(truth)), seed=0)
            .join(s1.select("s1_idx", s1_name="name", s1_addr="address"), on="s1_idx")
            .join(rec.select("rec_idx", "name", "address"), on="rec_idx"))
    log.info("sample of missed true pairs:\n%s", miss.select("s1_name", "name", "s1_addr", "address"))
    save_json(out, config.PREDS / f"blocking_{world}{norm}_{out_name}.json")


def run(a):
    d = config.WORLDS / a.world
    out = d / f"{a.out}.parquet"
    if stage_done([out], log, a.force):
        return
    sd = d / f"{a.out}_shards"
    sd.mkdir(parents=True, exist_ok=True)
    plan = shard_plan(a.world, a.norm, a.shard)
    files = [sd / f"{c}_{i:03d}.parquet" for c, i, _ in plan]
    with timed(log, f"blocking {a.world}: {len(plan)} shards (MAX_CAND {a.max_cand}, FR_EXTRA {not a.no_fr_extra})"):
        for (c, i, n_sh), f in zip(plan, files):
            if f.exists():
                continue
            cmd = [sys.executable, "-m", "er.blocking_lowmem", "--one", c, str(i), str(n_sh)] + sys.argv[1:]
            for attempt in range(3):
                if subprocess.run(cmd).returncode == 0:
                    break
                log.warning("shard %s %d failed (attempt %d)", c, i, attempt + 1)
            else:
                raise RuntimeError(f"shard {c} {i} failed 3 times")
        tmp = out.with_suffix(".tmp")
        pl.scan_parquet(files).sink_parquet(tmp, compression="zstd")
        tmp.replace(out)
    if a.world != "test":
        report_lowmem(a.world, a.norm, files, a.out)
    else:
        n = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
        log.info("pairs %d", n)


if __name__ == "__main__":
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_fmt_str_lengths(70)
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--norm", default="")
    ap.add_argument("--out", default="cand_v1")
    ap.add_argument("--shard", type=int, default=120_000)
    ap.add_argument("--tf-chunk", type=int, default=2000)
    ap.add_argument("--max-cand", type=int, default=blocking.MAX_CAND)
    ap.add_argument("--no-fr-extra", action="store_true")
    ap.add_argument("--tfidf-all", action="store_true")
    ap.add_argument("--one", nargs=3, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    blocking.MAX_CAND, blocking.TF_CHUNK = a.max_cand, a.tf_chunk
    blocking.FR_EXTRA, blocking.TFIDF_ALL = not a.no_fr_extra, a.tfidf_all
    if a.one:
        c, i, n_sh = a.one[0], int(a.one[1]), int(a.one[2])
        build_one(a.world, a.norm, c, i, n_sh, config.WORLDS / a.world / f"{a.out}_shards" / f"{c}_{i:03d}.parquet")
    else:
        run(a)
