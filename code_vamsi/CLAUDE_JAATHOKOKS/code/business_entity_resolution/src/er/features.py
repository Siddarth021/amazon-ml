"""Pair features (~55) for every candidate pair, in fixed chunks of config.FEAT_CHUNK rows.

Output: work/worlds/<w>/feat_v2/part_{i:05d}.parquet  (s1_idx, rec_idx, features...)
Run:    python -m er.features --world B [--cand cand_v1] [--norm _v2] [--force]
"""
import argparse
import gc

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from . import config
from .pairs import chunk, n_parts, pairs
from .utils import get_logger, save_json, stage_done, timed, write_parquet

log = get_logger("features")

S1_COLS = ["country", "core", "alias", "concat", "core_tok", "alias_tok", "rare1", "rare1_freq", "name_freq",
           "addr_freq", "ad", "ad_comps", "ad_tok", "city", "street1", "nums", "hnum", "region", "addr_empty"]
REC_COLS = ["country", "core", "alias", "concat", "core_tok", "alias_tok", "rare1_freq", "ad", "ad_comps", "ad_tok", "city",
            "street1", "nums", "hnum", "region", "addr_empty", "source", "is_web", "is_indic", "has_alias"]
NO_SHARED_F = 1e7


def load_norm(world: str, norm: str):
    d = config.WORLDS / world
    s1n = pl.read_parquet(d / f"s1_norm{norm}.parquet", columns=["idx"] + S1_COLS)
    recn = pl.read_parquet(d / f"rec_norm{norm}.parquet", columns=["idx"] + REC_COLS)
    assert (s1n["idx"].to_numpy() == np.arange(len(s1n))).all() and (recn["idx"].to_numpy() == np.arange(len(recn))).all()
    return s1n.drop("idx"), recn.drop("idx")


def idf_tables(world: str, norm: str, s1n: pl.DataFrame, recn: pl.DataFrame):
    """Per-country token frequency and N for names (tok_freq) and address tokens (computed like it)."""
    d = config.WORLDS / world
    nf = pl.read_parquet(d / f"tok_freq{norm}.parquet").rename({"tok_freq": "f"})
    path = d / f"atok_freq{norm}.parquet"
    if not path.exists():
        af = (pl.concat([s1n.select("country", tok="ad_tok"), recn.select("country", tok="ad_tok")])
              .explode("tok", empty_as_null=True).drop_nulls().group_by("country", "tok").len("f"))
        write_parquet(af, path)
    af = pl.read_parquet(path)
    n = pl.concat([s1n.select("country"), recn.select("country")]).group_by("country").len("N")
    return (nf.with_columns(pl.col("f").cast(pl.Float64)).join(n, on="country"),
            af.with_columns(pl.col("f").cast(pl.Float64)).join(n, on="country"))


def sim(a: pl.Series, b: pl.Series, scorer, scale: float = 100.0) -> np.ndarray:
    """Pairwise similarity in [0,1]; 0 when either side is empty/null."""
    a = a.fill_null("")
    b = b.fill_null("")
    out = process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32) / scale
    empty = ((a == "") | (b == "")).to_numpy()
    out[empty] = 0.0
    return out.astype(np.float32)


def idf_feats(df: pl.DataFrame, a: str, b: str, freq: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """IDF-weighted overlap of list columns a, b (per country): jaccard, shared sum, min shared freq."""
    base = df.select(pl.int_range(pl.len(), dtype=pl.Int32).alias("_i"), "country",
                     inter=pl.col(a).list.set_intersection(pl.col(b)),
                     union=pl.col(a).list.set_union(pl.col(b)))

    def agg(col, what):
        ex = (base.select("_i", "country", tok=col).explode("tok", empty_as_null=True).drop_nulls("tok")
              .join(freq, on=["country", "tok"], how="left")
              .with_columns(f=pl.col("f").fill_null(1.0))
              .with_columns(idf=(pl.col("N").cast(pl.Float64) / pl.col("f")).log()))
        return ex.group_by("_i").agg(what)

    inter = agg("inter", [pl.col("idf").sum().alias("si"), pl.col("f").min().alias("mf")])
    union = agg("union", [pl.col("idf").sum().alias("su")])
    r = (base.select("_i").join(inter, on="_i", how="left").join(union, on="_i", how="left").sort("_i")
         .with_columns(pl.col("si", "su").fill_null(0.0), pl.col("mf").fill_null(NO_SHARED_F)))
    return r.select(
        (pl.when(pl.col("su") > 0).then(pl.col("si") / pl.col("su")).otherwise(0.0)).cast(pl.Float32).alias(f"{prefix}_idf_jacc"),
        pl.col("si").cast(pl.Float32).alias(f"{prefix}_idf_inter"),
        pl.col("mf").cast(pl.Float32).alias(f"{prefix}_min_shared_f"),
    )


def jacc(inter: pl.Expr, a: pl.Expr, b: pl.Expr) -> pl.Expr:
    u = a.list.set_union(b).list.len()
    return pl.when(u > 0).then(inter / u).otherwise(0.0).cast(pl.Float32)


def chunk_features(c: pl.DataFrame, s1n: pl.DataFrame, recn: pl.DataFrame, nf, af) -> pl.DataFrame:
    si, ri = c["s1_idx"].to_numpy(), c["rec_idx"].to_numpy()
    s = s1n[si].rename(lambda x: f"s_{x}")
    r = recn[ri].rename(lambda x: f"r_{x}")
    df = pl.concat([c, s, r], how="horizontal").rename({"s_country": "country"}).drop("r_country")
    del s, r

    # string similarities (rapidfuzz, vectorized)
    f = {}
    f["n_ratio"] = sim(df["s_core"], df["r_core"], fuzz.ratio)
    f["n_token_set"] = sim(df["s_core"], df["r_core"], fuzz.token_set_ratio)
    f["n_token_sort"] = sim(df["s_core"], df["r_core"], fuzz.token_sort_ratio)
    f["n_partial"] = sim(df["s_core"], df["r_core"], fuzz.partial_ratio)
    f["n_jw"] = sim(df["s_core"], df["r_core"], JaroWinkler.normalized_similarity, 1.0)
    f["n_lev"] = sim(df["s_core"], df["r_core"], Levenshtein.normalized_similarity, 1.0)
    alias_combos = [("s_core", "r_alias"), ("s_alias", "r_core"), ("s_alias", "r_alias")]
    f["n_alias_set"] = np.max([sim(df[a], df[b], fuzz.token_set_ratio) for a, b in alias_combos], axis=0)
    f["n_alias_ratio"] = np.max([sim(df[a], df[b], fuzz.ratio) for a, b in alias_combos], axis=0)
    f["n_cat_ratio"] = sim(df["s_concat"], df["r_concat"], fuzz.ratio)
    f["n_cat_partial"] = sim(df["s_concat"], df["r_concat"], fuzz.partial_ratio)
    f["n_best"] = np.maximum.reduce([f["n_token_set"], f["n_alias_set"], f["n_cat_ratio"]])
    f["a_token_set"] = sim(df["s_ad"], df["r_ad"], fuzz.token_set_ratio)
    f["a_ratio"] = sim(df["s_ad"], df["r_ad"], fuzz.ratio)
    f["a_partial"] = sim(df["s_ad"], df["r_ad"], fuzz.partial_ratio)
    f["a_city_ratio"] = sim(df["s_city"], df["r_city"], fuzz.ratio)
    f["a_street_ratio"] = sim(df["s_street1"], df["r_street1"], fuzz.ratio)
    out = pl.DataFrame(f)

    # set / numeric / flag features (polars list ops, vectorized)
    rtok = pl.col("r_core_tok").list.concat(pl.col("r_alias_tok")).list.unique()
    tok_inter = pl.col("s_core_tok").list.set_intersection(rtok).list.len()
    num_inter = pl.col("s_nums").list.set_intersection(pl.col("r_nums")).list.len()
    at_inter = pl.col("s_ad_tok").list.set_intersection(pl.col("r_ad_tok")).list.len()
    has_nums = (pl.col("s_nums").list.len() > 0) & (pl.col("r_nums").list.len() > 0)
    g = df.select(
        n_tok_inter=tok_inter.cast(pl.Int16),
        n_tok_jacc=jacc(tok_inter, pl.col("s_core_tok"), rtok),
        n_first_eq=(pl.col("s_core_tok").list.first() == pl.col("r_core_tok").list.first()).fill_null(False).cast(pl.Int8),
        n_s1rare_in_rec=rtok.list.contains(pl.col("s_rare1")).fill_null(False).cast(pl.Int8),
        n_len_s1=pl.col("s_core_tok").list.len().cast(pl.Int16),
        n_len_rec=pl.col("r_core_tok").list.len().cast(pl.Int16),
        s1_rare1_freq=pl.col("s_rare1_freq").fill_null(0).cast(pl.Int32),
        rec_rare1_freq=pl.col("r_rare1_freq").fill_null(0).cast(pl.Int32),
        name_freq=pl.col("s_name_freq").cast(pl.Int32),
        num_inter=num_inter.cast(pl.Int16),
        num_jacc=jacc(num_inter, pl.col("s_nums"), pl.col("r_nums")),
        num_conflict=(has_nums & (num_inter == 0)).cast(pl.Int8),
        hnum_eq=(pl.col("s_hnum") == pl.col("r_hnum")).fill_null(False).cast(pl.Int8),
        hnum_in_other=(pl.col("r_nums").list.contains(pl.col("s_hnum")).fill_null(False)
                       | pl.col("s_nums").list.contains(pl.col("r_hnum")).fill_null(False)).cast(pl.Int8),
        street_eq=(pl.col("s_street1") == pl.col("r_street1")).fill_null(False).cast(pl.Int8),
        comp_overlap=pl.col("s_ad_comps").list.set_intersection(pl.col("r_ad_comps")).list.len().cast(pl.Int16),
        s1_city_in_rec_comps=pl.col("r_ad_comps").list.contains(pl.col("s_city")).fill_null(False).cast(pl.Int8),
        region_eq=pl.when(pl.col("s_region").is_null() | pl.col("r_region").is_null()).then(-1)
        .otherwise((pl.col("s_region") == pl.col("r_region")).cast(pl.Int8)).cast(pl.Int8),
        at_inter=at_inter.cast(pl.Int16),
        at_jacc=jacc(at_inter, pl.col("s_ad_tok"), pl.col("r_ad_tok")),
        a_empty_s1=pl.col("s_addr_empty").cast(pl.Int8),
        a_empty_rec=pl.col("r_addr_empty").cast(pl.Int8),
        addr_freq=pl.col("s_addr_freq").cast(pl.Int32),
        source=pl.col("r_source").cast(pl.Int8),
        is_web=pl.col("r_is_web").cast(pl.Int8),
        is_indic=pl.col("r_is_indic").cast(pl.Int8),
        has_alias=pl.col("r_has_alias").cast(pl.Int8),
        keys=pl.col("keys").cast(pl.Int32),
        nkeys=pl.col("keys").cast(pl.UInt32).bitwise_count_ones().cast(pl.Int8),
        kscore=pl.col("kscore").cast(pl.Float32),
        rec_ncand=pl.col("rec_ncand").cast(pl.Int16),
        s1_ncand=pl.col("s1_ncand").cast(pl.Int32),
    )
    nidf = idf_feats(df.with_columns(_rt=rtok), "s_core_tok", "_rt", nf, "n")
    aidf = idf_feats(df, "s_ad_tok", "r_ad_tok", af, "a")
    return pl.concat([c.select("s1_idx", "rec_idx"), out, g, nidf, aidf], how="horizontal")


def run(world: str, cand: str = "cand_v1", norm: str = "_v2", out: str = "feat_v2", force: bool = False):
    d = config.WORLDS / world / out
    c = pairs(world, cand)
    n = n_parts(len(c))
    outs = [d / f"part_{i:05d}.parquet" for i in range(n)]
    if stage_done(outs, log, force):
        return
    s1n, recn = load_norm(world, norm)
    nf, af = idf_tables(world, norm, s1n, recn)
    log.info("%s: %d pairs -> %d parts of %d", world, len(c), n, config.FEAT_CHUNK)
    for i, path in enumerate(outs):
        if path.exists() and not force:
            continue
        with timed(log, f"{world} part {i + 1}/{n}"):
            fe = chunk_features(chunk(c, i), s1n, recn, nf, af)
            write_parquet(fe, path)
        del fe
        gc.collect()
    save_json({"world": world, "pairs": len(c), "parts": n, "chunk": config.FEAT_CHUNK,
               "columns": list(pl.read_parquet_schema(outs[0]).keys())},
              config.PREDS / f"features_{world}_{out}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--cand", default="cand_v1")
    ap.add_argument("--norm", default="_v2")
    ap.add_argument("--out", default="feat_v2")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.world, a.cand, a.norm, a.out, a.force)
