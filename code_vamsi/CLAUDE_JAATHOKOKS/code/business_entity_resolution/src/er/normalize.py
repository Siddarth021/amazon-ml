"""Name and address normalization (country-agnostic, vectorized with polars).

Outputs per world:  work/worlds/<w>/s1_norm.parquet, rec_norm.parquet  (row order = idx)

Name columns
  core        cleaned name without legal suffixes/honorifics (string)      core_tok  its tokens
  alias       core of the other side of "X dba Y" / aka / formerly (null if none)
  concat      core without spaces (matches websites/handles like firstnetworks.com)
  rare1/rare2 the two rarest core tokens in this world (blocking keys)
  is_web, is_indic, has_alias
Address columns
  ad          cleaned address with the region component removed          ad_tok  alpha tokens, abbreviations canonicalized
  nums        unique numbers (leading zeros stripped)                     hnum    first number
  street1     first word after a number (street name / locality head)
  city        component just before the region (or last component)
  region      canonical region (learned: S2/S3 codes, native scripts and French departments are mapped to
              the S1 spelling by majority vote over high-precision exact-key matches in this world)
  addr_empty

Run:  python -m er.normalize --world B [--force]
"""
import argparse

import polars as pl

from . import config
from .utils import get_logger, stage_done, timed, write_parquet

log = get_logger("normalize")

INDIC = r"[ऀ-෿]"
ALIAS = (r"(?i)\s*\b(?:doing business as|d/b/a|dba|also known as|a/k/a|aka|formerly known as|formerly|"
         r"f/k/a|fka|nee|trading as|t/a)\b[\s:.-]*")
NAME_STOP = sorted({
    "inc", "incorporated", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation", "co", "company",
    "llp", "lp", "pllc", "plc", "pc", "the", "dr", "mr", "mrs", "ms", "smt", "shri", "sri", "m", "s",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "eirl", "snc", "and", "of", "lnc", "l",
})
ADDR_CANON = {
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "drive": "dr",
    "lane": "ln", "court": "ct", "place": "pl", "boulevard": "blvd", "bd": "blvd", "bld": "blvd",
    "circle": "cir", "highway": "hwy", "parkway": "pkwy", "terrace": "ter", "trail": "trl", "square": "sq",
    "apartment": "apt", "suite": "ste", "north": "n", "south": "s", "east": "e", "west": "w", "mount": "mt",
    "fort": "ft", "number": "no", "sector": "sec", "plot": "plt", "building": "bldg", "floor": "fl",
    "rue": "r", "allee": "all", "route": "rte", "chemin": "ch", "impasse": "imp", "cours": "crs",
    "city": "", "town": "", "of": "", "village": "", "null": "", "na": "", "n/a": "",
}


def fold(col: str) -> pl.Expr:
    """NFKD + drop Latin combining accents only (Indic vowel signs are kept) + lowercase + drop U+FFFD."""
    return (pl.col(col).str.normalize("NFKD").str.replace_all(r"[\u0300-\u036f]", "")
            .str.replace_all("�", "").str.to_lowercase())


def core_expr(e: pl.Expr) -> pl.Expr:
    """Folded name string -> core name (no punctuation, suffixes, honorifics)."""
    return (e.str.replace_all(r"\(\s*id\s*:?\s*\d+\s*\)", " ")
            .str.replace_all("&", " and ")
            .str.replace_all(r"\b(www\.|https?://)", " ")
            .str.replace_all(r"\.(com|net|org|in|co|biz|info|fr|us)\b", " ")
            .str.replace_all(r"\bl\.?\s*l\.?\s*[cp]\b\.?", " ")
            .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
            .str.split(" ")
            .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(NAME_STOP)))
            .list.join(" "))


def norm_names(df: pl.DataFrame) -> pl.DataFrame:
    f = fold("name")
    parts = f.str.replace(ALIAS, "\x01").str.split("\x01")
    return df.select(
        core=core_expr(parts.list.first()),
        alias=pl.when(parts.list.len() > 1).then(core_expr(parts.list.get(1, null_on_oob=True))).otherwise(None),
        has_alias=parts.list.len() > 1,
        is_web=pl.col("name").str.contains(r"(?i)(\.com|\.net|\.org|\.in\b|www\.|^\s*[@#]|\s[@#]\w)"),
        is_indic=pl.col("name").str.contains(INDIC),
    ).with_columns(
        # a name that is only an alias clause ("dba X") keeps X as its core
        core=pl.when(pl.col("core") == "").then(pl.col("alias").fill_null("")).otherwise(pl.col("core")),
    ).with_columns(
        core_tok=pl.col("core").str.split(" ").list.eval(pl.element().filter(pl.element() != "")),
        alias_tok=pl.col("alias").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")),
        concat=pl.col("core").str.replace_all(" ", ""),
    )


def norm_addresses(df: pl.DataFrame) -> pl.DataFrame:
    f = fold("address")
    comps = (f.str.replace_all(r"\b(null|n/a)\b", " ").str.split(",")
             .list.eval(pl.element().str.replace_all(r"[^\p{L}\p{M}\p{N}/ -]+", " ")
                        .str.replace_all(r"\s+", " ").str.strip_chars(" -/"))
             .list.eval(pl.element().filter(pl.element() != "")))
    return df.select(
        comps=comps,
        addr_empty=pl.col("address").str.strip_chars() == "",
    ).with_columns(
        last=pl.col("comps").list.last(),
        full=pl.col("comps").list.join(" "),
    ).with_columns(
        nums=pl.col("full").str.extract_all(r"\d+").list.eval(
            pl.element().str.strip_chars_start("0").replace("", "0")).list.unique(maintain_order=True),
        street1=pl.col("full").str.extract(r"\d+[a-z]?\s+(?:[a-z]\s+)?([\p{L}]{2,})", 1)
        .replace(ADDR_CANON),
    ).with_columns(hnum=pl.col("nums").list.first())


def rare_tokens(s1n: pl.DataFrame, recn: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Token frequency over S1+rec core tokens; attach the two rarest tokens per row."""
    toks = pl.concat([s1n.select("country", "core_tok"), recn.select("country", "core_tok")])
    freq = (toks.explode("core_tok", empty_as_null=True).drop_nulls().group_by("country", "core_tok").len("tok_freq")
            .rename({"core_tok": "tok"}))

    def attach(df, col, prefix):
        ex = (df.select("idx", "country", col).explode(col, empty_as_null=True).drop_nulls()
              .rename({col: "tok"}).unique(["idx", "tok"])
              .join(freq, on=["country", "tok"], how="left")
              .sort(["idx", "tok_freq", "tok"])
              .group_by("idx", maintain_order=True)
              .agg(pl.col("tok").first().alias(f"{prefix}1"),
                   pl.col("tok").get(1, null_on_oob=True).alias(f"{prefix}2"),
                   pl.col("tok_freq").first().alias(f"{prefix}1_freq")))
        return df.join(ex, on="idx", how="left")

    recn = attach(attach(recn, "core_tok", "rare"), "alias_tok", "arare")
    return attach(s1n, "core_tok", "rare"), recn, freq


def split_region(df: pl.DataFrame, region_vals: pl.DataFrame) -> pl.DataFrame:
    """Region = the last component that is a frequent region value of its country (any position, so
    reordered addresses work); city = last digit-free component of the rest; ad = the rest joined."""
    ex = (df.select("idx", "country", "comps").with_row_index("_r").explode("comps", empty_as_null=True)
          .with_columns(_pos=pl.int_range(pl.len()).over("_r"))
          .join(region_vals.rename({"last": "comps"}).with_columns(_hit=pl.lit(True)), on=["country", "comps"], how="left"))
    reg = (ex.filter(pl.col("_hit").fill_null(False)).sort("_r", "_pos")
           .group_by("_r").agg(region_raw=pl.col("comps").last(), _rpos=pl.col("_pos").last()))
    city = (ex.join(reg, on="_r", how="left")
            .filter((pl.col("_pos") != pl.col("_rpos").fill_null(-1)) & pl.col("comps").is_not_null()
                    & ~pl.col("comps").str.contains(r"\d"))
            .sort("_r", "_pos").group_by("_r").agg(city=pl.col("comps").last()))
    df = (df.with_row_index("_r").join(reg, on="_r", how="left").join(city, on="_r", how="left").sort("_r"))
    return df.with_columns(
        body=pl.when(pl.col("_rpos").is_not_null())
        .then(pl.concat_list(pl.col("comps").list.head(pl.col("_rpos")), pl.col("comps").list.slice(pl.col("_rpos") + 1)))
        .otherwise(pl.col("comps")),
    ).with_columns(
        ad=pl.col("body").list.join(" "),
        ad_comps=pl.col("body"),
    ).with_columns(
        ad_tok=pl.col("ad").str.split(" ")
        .list.eval(pl.element().replace(ADDR_CANON))
        .list.eval(pl.element().filter((pl.element() != "") & pl.element().str.contains(r"^[\p{L}\p{M}]+$")))
        .list.unique(maintain_order=True),
    ).drop("_r", "_rpos", "body", "comps", "last", "full")


def learn_region_map(s1n: pl.DataFrame, recn: pl.DataFrame) -> pl.DataFrame:
    """Map rec region spellings (codes, native script, departments) to the S1 spelling.

    Uses exact (core name, house number) matches whose key is unique on the S1 side: ~97% precise
    with no labels, so this works on the test world (France) exactly as on train.
    """
    key = ["country", "core", "hnum"]
    a = s1n.filter((pl.col("core") != "") & pl.col("hnum").is_not_null() & pl.col("region_raw").is_not_null())
    a = a.filter(pl.len().over(key) == 1).select(*key, s1_region="region_raw")
    b = recn.filter(pl.col("region_raw").is_not_null()).select(*key, "region_raw")
    votes = b.join(a, on=key).group_by("country", "region_raw", "s1_region").len("n")
    best = (votes.sort("n", descending=True).group_by("country", "region_raw", maintain_order=True)
            .agg(pl.col("s1_region").first(), share=pl.col("n").first() / pl.col("n").sum(), n=pl.col("n").sum()))
    return best.filter((pl.col("n") >= 20) & (pl.col("share") >= 0.5)).select("country", "region_raw", "s1_region")


def run(world: str, force: bool = False, translit_from: str | None = None, suffix: str = ""):
    """translit_from: comma-separated training worlds whose learned Indic dictionary is applied to names
    (v2+). suffix: output name suffix, e.g. "_v2" -> s1_norm_v2.parquet."""
    d = config.WORLDS / world
    outs = [d / f"s1_norm{suffix}.parquet", d / f"rec_norm{suffix}.parquet", d / f"tok_freq{suffix}.parquet"]
    if stage_done(outs, log, force):
        return
    s1 = pl.read_parquet(d / "s1.parquet")
    rec = pl.read_parquet(d / "rec.parquet")
    if translit_from:
        from .translit import load_dict, to_latin
        with timed(log, f"Indic -> Latin names with dictionary from {translit_from}"):
            dct = load_dict(translit_from.split(","))
            rec = rec.with_columns(name_orig=pl.col("name"), name=to_latin(rec["name"], dct))
            log.info("transliterated %d Indic names", rec.filter(pl.col("name") != pl.col("name_orig")).height)

    with timed(log, f"normalize names+addresses ({world}: {len(s1)} S1, {len(rec)} rec)"):
        s1n = pl.concat([s1.select(idx="s1_idx", country="country"), norm_names(s1), norm_addresses(s1)], how="horizontal")
        recn = pl.concat([rec.select(idx="rec_idx", country="country", source="source"),
                          norm_names(rec), norm_addresses(rec)], how="horizontal")
        if translit_from:  # is_indic must describe the original record
            recn = recn.with_columns(is_indic=rec["name_orig"].str.contains(INDIC))

    with timed(log, "regions: frequent last components -> region, learned rec->S1 map"):
        lasts = pl.concat([s1n.select("country", "last"), recn.select("country", "last")]).drop_nulls()
        n_by_c = lasts.group_by("country").len("nc")
        region_vals = (lasts.group_by("country", "last").len("n").join(n_by_c, on="country")
                       .filter((pl.col("n") >= 200) & (pl.col("n") >= 0.001 * pl.col("nc")))
                       .select("country", "last"))
        s1n = split_region(s1n, region_vals)
        recn = split_region(recn, region_vals)
        rmap = learn_region_map(s1n, recn)
        log.info("region values: %d; learned rec->S1 region mappings: %d (e.g. %s)", len(region_vals), len(rmap),
                 rmap.sample(min(8, len(rmap)), seed=0).select("region_raw", "s1_region").rows())
        s1n = s1n.with_columns(region=pl.col("region_raw"))
        recn = (recn.join(rmap, on=["country", "region_raw"], how="left")
                .with_columns(region=pl.coalesce("s1_region", "region_raw")).drop("s1_region"))

    with timed(log, "rare tokens + name/address frequency"):
        s1n, recn, freq = rare_tokens(s1n, recn)
        s1n = s1n.with_columns(
            name_freq=pl.len().over("country", "core").cast(pl.Int32),
            addr_freq=pl.len().over("country", "ad").cast(pl.Int32),
        )
        s1n = s1n.sort("idx")
        recn = recn.sort("idx")

    write_parquet(s1n, outs[0])
    write_parquet(recn, outs[1])
    write_parquet(freq, outs[2])
    ex = recn.filter(pl.col("has_alias") | pl.col("is_web")).head(3).select("core", "alias", "concat", "ad", "nums", "street1", "city", "region")
    log.info("examples:\n%s", ex)
    log.info("rec: alias %.3f web %.3f indic %.3f empty-addr %.3f; region found %.3f",
             recn["has_alias"].mean(), recn["is_web"].mean(), recn["is_indic"].mean(),
             recn["addr_empty"].mean(), recn["region"].is_not_null().mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--translit-from", default=None, help="e.g. A (validation) or A,B (test)")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    run(a.world, a.force, a.translit_from, a.suffix)
