"""Candidate generation (blocking): S2/S3 record -> S1, within country, union of rule keys.

Each key joins a record to every S1 sharing the key value; key values held by more than `cap` S1s
are dropped (too generic). S1 side uses its two rarest name tokens; the record side uses *every*
token of its core name and alias, so a typo in the record's rarest token can't hide the match.

Keys (bit in `keys` column); "any number" = every number in the address, not just the first one
(S2/S3 often prepend an extra "PLOT 704" / "H.NO 947"); "any city part" = every digit-free address
component except the region (handles reordered addresses):
     1 hs    number + the word that follows it (street name)
     2 hr    any number + name token
     4 core  full core name, words sorted (word-order swaps)
     8 nums  full numeric signature (>= 2 numbers)
    16 ph    any number + phonetic code (metaphone) of a name token
    32 cat   concatenated-name prefix of 7 (websites / handles)
    64 cr    any city part + name token
   128 r     rare name token alone (only when very specific)
   256 hc    any number + any city part
   512 at    rare address token alone (only when very specific; finds Indic-name records by address)
  1024 atn   rare address token + any number
  2048 rp    pair of name tokens (S1: pairs among its 3 rarest; record: all pairs) - name-only, extra/missing words
  4096 atc   address token + any city part (no number needed: typo'd / missing house numbers)
  8192 tf    char-3-gram TF-IDF name kNN (top 5, cosine >= 0.5) for records without any address number
 16384 hsf   France only: number + street NAME, skipping street-type words / articles ("43 rue de la paix" -> "43 paix")
 32768 sfc   France only: street-name word (first or last meaningful word of the street component) + any city part (no number)
 65536 hsfl  France only: number + LAST meaningful street word ("43 rue jean bedouret" -> "43 bedouret")

Output: work/worlds/<w>/cand_v1.parquet  (s1_idx, rec_idx, keys)
Run:    python -m er.blocking --world B
"""
import argparse

import jellyfish
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import ADDR_CANON

from . import config
from .evaluate import ceiling, report, truth_pairs
from .utils import get_logger, save_json, stage_done, timed, write_parquet

log = get_logger("blocking")

# Caps are sized for a ~1M-S1 world (fold A/B, test); candidates are ranked by key specificity
# (sum over agreeing key types of 1 / #S1 sharing the key value) and cut to MAX_CAND per record.
CAP = 100             # max S1 per key value
CAP_NAME_ONLY = 25    # stricter cap for name-only keys
CAP_PAIR = 30         # cap for the token-pair key
AT_MAX_FREQ = 300     # an S1 address token is "rare" only if it occurs <= this often in its country
PAIR_MAX_FREQ = 20000 # name tokens more common than this are left out of token pairs
MAX_CAND = 50         # hard limit of candidates per record (most specific kept)
SHARD = 1_500_000     # records per blocking shard (bounds peak memory)
FR_EXTRA = True       # --no-fr-extra: France keys hsf only (leaderboard-best v8); True adds sfc + hsfl (v9)


def k(*parts) -> pl.Expr:
    """Null-safe composite key: null if any part is null/empty."""
    ps = [pl.col(p) if isinstance(p, str) else p for p in parts]
    ok = pl.all_horizontal([x.is_not_null() & (x.cast(pl.String) != "") for x in ps])
    return pl.when(ok).then(pl.concat_str(ps, separator="|")).otherwise(None)


def metaphone_map(tokens: pl.Series) -> dict:
    out = {}
    for t in tokens.unique().drop_nulls().to_list():
        if t.isascii() and t.isalpha() and len(t) >= 3:
            out[t] = jellyfish.metaphone(t) or None
    return out


def side_tables(df: pl.DataFrame, is_s1: bool, afreq: pl.DataFrame, tfreq: pl.DataFrame) -> dict:
    """Exploded helper tables (idx, country, value) for one side."""
    base = df.select("idx", "country")
    num = df.select("idx", "country", v="nums").explode("v", empty_as_null=True).drop_nulls("v").unique(["idx", "v"])
    if is_s1:
        tok = (pl.concat([df.select("idx", "country", v="rare1"), df.select("idx", "country", v="rare2")])
               .drop_nulls("v").unique(["idx", "v"]))
    else:
        tok = (df.select("idx", "country", v=pl.concat_list("core_tok", "alias_tok"))
               .explode("v", empty_as_null=True).drop_nulls("v").unique(["idx", "v"]))
    city = (df.select("idx", "country", v="ad_comps").explode("v", empty_as_null=True)
            .drop_nulls("v")
            .with_columns(v=pl.col("v").str.replace_all(r"\b(city|town|village|township|twp|of|cdp|cpd)\b", " ")
                          .str.split(" ").list.eval(pl.element().replace(ADDR_CANON)).list.join(" ")
                          .str.replace_all(r"\s+", " ").str.strip_chars())
            .filter(~pl.col("v").str.contains(r"\d") & (pl.col("v").str.len_chars() >= 3))
            .unique(["idx", "v"]))
    nw = (df.select("idx", "country", v=pl.col("ad").str.extract_all(r"\d+[a-z]?\s+(?:[a-z]\s+)?[\p{L}\p{M}]{2,}"))
          .explode("v", empty_as_null=True).drop_nulls("v")
          .with_columns(v=pl.col("v").str.replace(r"^0+(\d)", "$1").str.replace_all(r"\s+", " "))
          .unique(["idx", "v"]))
    at = (df.select("idx", "country", v="ad_tok").explode("v", empty_as_null=True).drop_nulls("v")
          .filter(pl.col("v").str.len_chars() >= 3).unique(["idx", "v"]))
    at3 = at
    if is_s1:  # keep the two rarest address tokens per S1 (and the 3 rarest, any frequency, for atc)
        ranked = at.join(afreq, on=["country", "v"], how="left").sort("idx", "af", "v")
        at = ranked.filter(pl.col("af") <= AT_MAX_FREQ).group_by("idx", maintain_order=True).head(2).drop("af")
        at3 = ranked.group_by("idx", maintain_order=True).head(3).drop("af")
    # token pairs: S1 -> pairs among its 3 rarest tokens, record -> all pairs of its (<= 8) tokens
    if is_s1:
        pt = (df.select("idx", "country", v="core_tok").explode("v", empty_as_null=True).drop_nulls("v")
              .unique(["idx", "v"]).join(tfreq, on=["country", "v"], how="left")
              .filter(pl.col("tf") <= PAIR_MAX_FREQ).sort("idx", "tf", "v")
              .group_by("idx", maintain_order=True).head(3).drop("tf"))
    else:
        pt = (tok.join(tfreq, on=["country", "v"], how="left").filter(pl.col("tf") <= PAIR_MAX_FREQ).drop("tf")
              .filter(pl.len().over("idx") <= 8))
    pairs = (pt.join(pt.drop("country"), on="idx", suffix="2").filter(pl.col("v") < pl.col("v2")))
    return {"base": base, "num": num, "tok": tok, "city": city, "nw": nw, "at": at, "at3": at3, "pairs": pairs}


FR_SKIP = ("rue|r|avenue|av|ave|allee|all|boulevard|bd|blvd|place|pl|chemin|ch|impasse|imp|route|rte|cours|crs|"
           "quai|square|sq|passage|sentier|ruelle|residence|res|rond|point|voie|cite|lotissement|lot|hameau|parc|"
           "promenade|esplanade|faubourg|fg|bis|ter|quater|de|du|des|la|le|les|l|d|et|saint|st|ste")
FR_NW = rf"\d+[a-z]?\s+(?:(?:{FR_SKIP})\s+)*[\p{{L}}\p{{M}}]{{2,}}"


def fr_numword(df: pl.DataFrame) -> pl.DataFrame:
    """(idx, country, v = "<number> <first non-type word>") for French-style addresses."""
    return (df.select("idx", "country", v=pl.col("ad").str.extract_all(FR_NW)).explode("v", empty_as_null=True)
            .drop_nulls("v").with_columns(parts=pl.col("v").str.split(" "))
            .with_columns(v=pl.concat_str([pl.col("parts").list.first().str.replace(r"^0+(\d)", "$1").str.extract(r"^(\d+)", 1),
                                           pl.col("parts").list.last()], separator=" "))
            .drop("parts").drop_nulls("v").unique(["idx", "v"]))


FR_TYPES = ("rue|r|avenue|av|ave|allee|all|boulevard|bd|blvd|place|pl|chemin|ch|impasse|imp|route|rte|cours|crs|quai|"
            "square|sq|passage|sentier|ruelle|residence|res|rond|voie|cite|lotissement|lot|hameau|parc|promenade|esplanade|"
            "faubourg|fg")
FR_SKIP_SET = set(FR_SKIP.split("|")) | {"n", "no", "nd"}


def fr_street(df: pl.DataFrame) -> pl.DataFrame:
    """(idx, country, num, first, last) from address components that look like a French street line:
    an optional leading number, then words; street-type words and articles are dropped."""
    ex = (df.select("idx", "country", comp="ad_comps").explode("comp", empty_as_null=True).drop_nulls("comp")
          .with_columns(num=pl.col("comp").str.extract(r"^(?:no?\s+)?0*(\d+)", 1),
                        typed=pl.col("comp").str.contains(rf"^(?:no?\s+)?\d*[a-z]?\s*(?:{FR_TYPES})\b"))
          .filter(pl.col("num").is_not_null() | pl.col("typed"))
          .with_columns(words=pl.col("comp").str.split(" ").list.eval(
              pl.element().filter(~pl.element().is_in(sorted(FR_SKIP_SET))
                                  & pl.element().str.contains(r"^[\p{L}\p{M}]{3,}$"))))
          .filter(pl.col("words").list.len() > 0)
          .with_columns(first=pl.col("words").list.first(), last=pl.col("words").list.last())
          .select("idx", "country", "num", "first", "last"))
    return ex


def cross(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Per-row combinations of two exploded tables: (idx, country, v1, v2)."""
    return a.join(b.drop("country"), on="idx", suffix="2")


def freqs(s1n: pl.DataFrame, recn: pl.DataFrame):
    """Address-token and name-token frequencies over the whole (country) world."""
    afreq = (pl.concat([s1n.select("country", v="ad_tok"), recn.select("country", v="ad_tok")])
             .explode("v", empty_as_null=True).drop_nulls().group_by("country", "v").len("af"))
    tfreq = (pl.concat([s1n.select("country", v="core_tok"), recn.select("country", v="core_tok")])
             .explode("v", empty_as_null=True).drop_nulls().group_by("country", "v").len("tf"))
    return afreq, tfreq


TF_K, TF_MIN, TF_CHUNK = 5, 0.5, 20_000
TFIDF_ALL = False  # --tfidf-all: name TF-IDF kNN for every record, not only records without address numbers


def tfidf_knn(s1n: pl.DataFrame, q: pl.DataFrame) -> pl.DataFrame:
    """Top-TF_K S1 by char-3-gram TF-IDF cosine on the core name, for query records q (one country)."""
    empty = pl.DataFrame(schema={"s1_idx": pl.Int32, "rec_idx": pl.Int32, "w": pl.Float32, "bit": pl.Int32})
    if len(q) == 0 or len(s1n) == 0:
        return empty
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2, max_df=0.05, sublinear_tf=True,
                          dtype=np.float32)
    A = vec.fit_transform(s1n["core"].fill_null("").to_list())
    AT = A.T.tocsr()
    Q = vec.transform(q["core"].fill_null("").to_list())
    s1_ids, q_ids = s1n["idx"].to_numpy(), q["idx"].to_numpy()
    out_s, out_r, out_w = [], [], []
    for st in range(0, Q.shape[0], TF_CHUNK):
        M = (Q[st:st + TF_CHUNK] @ AT).tocsr()
        for i in range(M.shape[0]):
            a, b = M.indptr[i], M.indptr[i + 1]
            if a == b:
                continue
            d, ix = M.data[a:b], M.indices[a:b]
            if len(d) > TF_K:
                top = np.argpartition(-d, TF_K)[:TF_K]
                d, ix = d[top], ix[top]
            keep = d >= TF_MIN
            out_s.append(s1_ids[ix[keep]])
            out_r.append(np.full(keep.sum(), q_ids[st + i]))
            out_w.append(d[keep])
    if not out_s:
        return empty
    return pl.DataFrame({"s1_idx": np.concatenate(out_s).astype(np.int32), "rec_idx": np.concatenate(out_r).astype(np.int32),
                         "w": np.concatenate(out_w).astype(np.float32)}).with_columns(bit=pl.lit(8192, pl.Int32))


def build(s1n: pl.DataFrame, recn: pl.DataFrame, afreq: pl.DataFrame, tfreq: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """Candidates for the records in `recn` against all of `s1n` (records can be passed in shards)."""
    S, R = side_tables(s1n, True, afreq, tfreq), side_tables(recn, False, afreq, tfreq)
    mp = metaphone_map(pl.concat([S["tok"]["v"], R["tok"]["v"]]))
    ph = lambda c: pl.col(c).replace_strict(mp, default=None, return_dtype=pl.String)
    sorted_core = lambda: pl.col("core_tok").list.sort().list.join(" ")
    numsig = lambda: pl.col("nums").list.sort().list.join(" ")
    cat7 = lambda: pl.col("concat").str.slice(0, 7)

    specs = [  # bit, name, (S1 frame, key), (rec frame, key), cap
        (1, "hs", S["nw"], k("country", "v"), R["nw"], k("country", "v"), CAP),
        (2, "hr", cross(S["num"], S["tok"]), k("country", "v", "v2"), cross(R["num"], R["tok"]), k("country", "v", "v2"), CAP),
        (4, "core", s1n, k("country", sorted_core()), recn, k("country", sorted_core()), CAP),
        (8, "nums", s1n.filter(pl.col("nums").list.len() >= 2), k("country", numsig()),
         recn.filter(pl.col("nums").list.len() >= 2), k("country", numsig()), CAP),
        (16, "ph", cross(S["num"], S["tok"]), k("country", "v", ph("v2")), cross(R["num"], R["tok"]), k("country", "v", ph("v2")), CAP),
        (32, "cat", s1n.filter(pl.col("concat").str.len_chars() >= 6), k("country", cat7()),
         recn.filter(pl.col("concat").str.len_chars() >= 6), k("country", cat7()), CAP),
        (64, "cr", cross(S["city"], S["tok"]), k("country", "v", "v2"), cross(R["city"], R["tok"]), k("country", "v", "v2"), CAP),
        (128, "r", S["tok"], k("country", "v"), R["tok"], k("country", "v"), CAP_NAME_ONLY),
        (256, "hc", cross(S["num"], S["city"]), k("country", "v", "v2"), cross(R["num"], R["city"]), k("country", "v", "v2"), CAP),
        (512, "at", S["at"], k("country", "v"), R["at"], k("country", "v"), CAP_NAME_ONLY),
        (1024, "atn", cross(S["at"], S["num"]), k("country", "v", "v2"), cross(R["at"], R["num"]), k("country", "v", "v2"), CAP),
        (2048, "rp", S["pairs"], k("country", "v", "v2"), R["pairs"], k("country", "v", "v2"), CAP_PAIR),
        (4096, "atc", cross(S["at3"], S["city"]), k("country", "v", "v2"), cross(R["at3"], R["city"]), k("country", "v", "v2"), CAP),
    ]
    if len(s1n) and (s1n["country"] == "France").all():
        specs.append((16384, "hsf", fr_numword(s1n), k("country", "v"), fr_numword(recn), k("country", "v"), CAP))
    if FR_EXTRA and len(s1n) and (s1n["country"] == "France").all():
        fs, fr = fr_street(s1n), fr_street(recn)
        word = lambda f: pl.concat([f.select("idx", "country", v="first"), f.select("idx", "country", v="last")]).unique()
        specs.append((65536, "hsfl", fs.filter(pl.col("num").is_not_null()), k("country", "num", "last"),
                      fr.filter(pl.col("num").is_not_null()), k("country", "num", "last"), CAP))
        specs.append((32768, "sfc", cross(word(fs), S["city"]), k("country", "v", "v2"),
                      cross(word(fr), R["city"]), k("country", "v", "v2"), CAP))
    parts, stats = [], {}
    for bit, name, sdf, skey, rdf, rkey, cap in specs:
        s_ = sdf.select(s1_idx="idx", key=skey).drop_nulls("key").unique()
        s_ = s_.with_columns(_n=pl.len().over("key")).filter(pl.col("_n") <= cap)
        r_ = rdf.select(rec_idx="idx", key=rkey).drop_nulls("key").unique()
        p = (r_.join(s_, on="key").group_by("s1_idx", "rec_idx").agg(w=(1.0 / pl.col("_n").min()).cast(pl.Float32))
             .with_columns(bit=pl.lit(bit, pl.Int32)))
        stats[name] = len(p)
        log.info("key %-5s pairs %11d", name, len(p))
        parts.append(p)
    tf = tfidf_knn(s1n, recn if TFIDF_ALL else recn.filter(pl.col("nums").list.len() == 0))
    stats["tf"] = len(tf)
    log.info("key tf    pairs %11d", len(tf))
    parts.append(tf)
    cand = (pl.concat(parts).group_by("s1_idx", "rec_idx").agg(keys=pl.col("bit").sum(), kscore=pl.col("w").sum())
            .with_columns(pl.col("s1_idx").cast(pl.Int32), pl.col("rec_idx").cast(pl.Int32)))
    del parts
    n0 = len(cand)
    cand = (cand.sort(["rec_idx", "kscore"], descending=[False, True])
            .group_by("rec_idx", maintain_order=True).head(MAX_CAND))
    log.info("per-record cap %d: %d -> %d pairs", MAX_CAND, n0, len(cand))
    return cand, stats


def recall_report(world: str, cand: pl.DataFrame, s1n: pl.DataFrame, recn: pl.DataFrame, stats: dict) -> dict:
    s1 = pl.read_parquet(config.WORLDS / world / "s1.parquet")
    rec = pl.read_parquet(config.WORLDS / world / "rec.parquet")
    truth = truth_pairs(rec)
    hit = truth.join(cand, on=["s1_idx", "rec_idx"], how="left")
    recall = hit["keys"].is_not_null().mean()
    # unique contribution of each key: true pairs found only by that key
    only = {name: hit.filter(pl.col("keys") == bit).height / len(truth)
            for bit, name in [(1 << i, n) for i, n in enumerate(stats)]}
    by_c = (hit.join(s1.select("s1_idx", "country"), on="s1_idx")
            .group_by("country").agg(pl.col("keys").is_not_null().mean()).sort("country").rows())
    n_all = (s1n.group_by("country").len("a").join(recn.group_by("country").len("b"), on="country")
             .select((pl.col("a").cast(pl.Float64) * pl.col("b").cast(pl.Float64)).sum()).item())
    out = {
        "recall": recall, "recall_by_country": dict(by_c),
        "pairs": len(cand), "pairs_per_rec": len(cand) / len(recn),
        "reduction_ratio": 1 - len(cand) / n_all,
        "only_this_key": only, "pairs_by_key": stats,
    }
    log.info("RECALL %.5f  by country %s  pairs %d (%.2f per rec)  reduction ratio %.6f",
             recall, {c: round(v, 5) for c, v in by_c}, len(cand), out["pairs_per_rec"], out["reduction_ratio"])
    log.info("true pairs found ONLY by key: %s", {n: round(v, 5) for n, v in only.items()})
    r = ceiling(s1, truth, cand)
    report(log, f"{world} blocking CEILING (perfect model on these candidates)", r)
    out["ceiling"] = r
    miss = (truth.join(cand, on=["s1_idx", "rec_idx"], how="anti").sample(min(15, len(truth)), seed=0)
            .join(s1.select("s1_idx", s1_name="name", s1_addr="address"), on="s1_idx")
            .join(rec.select("rec_idx", "name", "address"), on="rec_idx"))
    log.info("sample of missed true pairs:\n%s", miss.select("s1_name", "name", "s1_addr", "address"))
    return out


def run(world: str, force: bool = False, norm: str = "", out_name: str = "cand_v1"):
    d = config.WORLDS / world
    out = d / f"{out_name}.parquet"
    if stage_done([out], log, force):
        return
    s1n = pl.read_parquet(d / f"s1_norm{norm}.parquet")
    recn = pl.read_parquet(d / f"rec_norm{norm}.parquet")
    with timed(log, f"blocking {world}"):
        # keys never cross countries, so process one country at a time (same result, ~half the peak RAM)
        cands, stats = [], {}
        # and within a country, records in shards of <= SHARD (exact: keys only depend on the record's own fields)
        for c in sorted(s1n["country"].unique().to_list()):
            sc, rc = s1n.filter(pl.col("country") == c), recn.filter(pl.col("country") == c)
            afreq, tfreq = freqs(sc, rc)
            n_sh = max(1, -(-len(rc) // SHARD))
            for i in range(n_sh):
                log.info("country %s shard %d/%d", c, i + 1, n_sh)
                cc, st = build(sc, rc.slice(i * len(rc) // n_sh, (i + 1) * len(rc) // n_sh - i * len(rc) // n_sh),
                               afreq, tfreq)
                cands.append(cc)
                for kname, v in st.items():
                    stats[kname] = stats.get(kname, 0) + v
        cand = pl.concat(cands)
        del cands
    write_parquet(cand, out)
    if world != "test":
        save_json(recall_report(world, cand, s1n, recn, stats), config.PREDS / f"blocking_{world}{norm}_{out_name}.json")
    else:
        log.info("pairs %d (%.2f per rec)", len(cand), len(cand) / len(recn))


if __name__ == "__main__":
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_fmt_str_lengths(70)
    pl.Config.set_tbl_rows(20)
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--norm", default="", help="normalization suffix, e.g. _v2")
    ap.add_argument("--out", default="cand_v1")
    ap.add_argument("--tfidf-all", action="store_true")
    ap.add_argument("--no-fr-extra", action="store_true", help="France: hsf key only (v8), skip sfc/hsfl")
    ap.add_argument("--max-cand", type=int, default=MAX_CAND, help="candidates kept per record")
    ap.add_argument("--shard", type=int, default=SHARD, help="records per shard (lower for less RAM)")
    ap.add_argument("--tf-chunk", type=int, default=TF_CHUNK, help="TF-IDF queries per sparse product (RAM only)")
    a = ap.parse_args()
    TFIDF_ALL = a.tfidf_all
    FR_EXTRA = not a.no_fr_extra
    MAX_CAND, SHARD, TF_CHUNK = a.max_cand, a.shard, a.tf_chunk
    log.info("MAX_CAND %d SHARD %d FR_EXTRA %s TFIDF_ALL %s", MAX_CAND, SHARD, FR_EXTRA, TFIDF_ALL)
    run(a.world, a.force, a.norm, a.out)
