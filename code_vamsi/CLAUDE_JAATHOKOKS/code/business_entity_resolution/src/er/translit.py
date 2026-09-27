"""Learned Indic -> Latin word dictionary for record names.

learn(world): on the world's true pairs whose record name is in an Indian script and whose S1 name is Latin,
align the two token lists position by position (only when they have the same length) and keep
indic -> latin when seen >= MIN_COUNT times with share >= MIN_SHARE. Legal words are kept on purpose
(प्राइवेट -> private, लिमिटेड -> limited), normalization drops them afterwards.

to_latin(names, dct): replaces every Indic run in a name by its dictionary word, else by a rule-based
transliteration (script from the Unicode block -> IAST -> ASCII, trailing inherent "a" dropped).

Output: work/models/translit_<A|A_B>.json
Run:    python -m er.translit --worlds A        (validation worlds)
        python -m er.translit --worlds A,B      (test)
"""
import argparse
import re
import unicodedata

import polars as pl

from . import config
from .utils import get_logger, load_json, save_json, timed

log = get_logger("translit")

INDIC = r"[ऀ-෿]"
INDIC_RUN = re.compile(r"[ऀ-෿]+")
MIN_COUNT, MIN_SHARE = 3, 0.6
SCRIPTS = [  # (first code point of the Unicode block, sanscript scheme)
    (0x0900, "devanagari"), (0x0980, "bengali"), (0x0A00, "gurmukhi"), (0x0A80, "gujarati"), (0x0B00, "oriya"),
    (0x0B80, "tamil"), (0x0C00, "telugu"), (0x0C80, "kannada"), (0x0D00, "malayalam"),
]


def tokens(col: str) -> pl.Expr:
    """NFKD fold (Latin accents only) + lowercase + & -> and, split on non letters/marks/digits."""
    return (pl.col(col).str.normalize("NFKD").str.replace_all(r"[̀-ͯ]", "").str.to_lowercase()
            .str.replace_all("&", " and ").str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ").str.strip_chars()
            .str.split(" ").list.eval(pl.element().filter(pl.element() != "")))


def learn(worlds: list[str]) -> dict[str, str]:
    pairs = []
    for w in worlds:
        d = config.WORLDS / w
        s1 = pl.read_parquet(d / "s1.parquet", columns=["s1_idx", "name"])
        rec = pl.read_parquet(d / "rec.parquet", columns=["s1_idx", "name"])
        pairs.append(rec.filter(pl.col("s1_idx").is_not_null() & pl.col("name").str.contains(INDIC))
                     .join(s1.rename({"name": "s1_name"}), on="s1_idx")
                     .filter(~pl.col("s1_name").str.contains(INDIC))
                     .select(rt=tokens("name"), st=tokens("s1_name")))
    p = pl.concat(pairs).filter(pl.col("rt").list.len() == pl.col("st").list.len())
    al = (p.with_row_index("_p").explode("rt", "st", empty_as_null=True).drop_nulls()
          .filter(pl.col("rt").str.contains(INDIC)))
    tot = al.group_by("rt").len("tot")
    cnt = (al.filter(pl.col("st").str.contains(r"^[a-z]+$")).group_by("rt", "st").len("n")
           .sort(["n", "st"], descending=[True, False]).group_by("rt", maintain_order=True).first()
           .join(tot, on="rt")
           .filter((pl.col("n") >= MIN_COUNT) & (pl.col("n") / pl.col("tot") >= MIN_SHARE)))
    log.info("learned %d Indic words from %d aligned pairs (worlds %s); e.g. %s", len(cnt), len(p), worlds,
             cnt.sort("n", descending=True).head(10).select("rt", "st", "n").rows())
    return dict(cnt.select("rt", "st").iter_rows())


def load_dict(worlds: list[str]) -> dict[str, str]:
    path = config.MODELS / f"translit_{'_'.join(worlds)}.json"
    if not path.exists():
        with timed(log, f"learn Indic dictionary from {worlds}"):
            save_json(learn(worlds), path)
    return load_json(path)


def rule_translit(tok: str) -> str:
    from indic_transliteration import sanscript
    cp = ord(tok[0])
    scheme = next((s for lo, s in reversed(SCRIPTS) if cp >= lo), None)
    if scheme is None or cp >= 0x0D80:  # outside the supported blocks (e.g. Sinhala)
        return ""
    out = sanscript.transliterate(tok, scheme, sanscript.IAST)
    out = unicodedata.normalize("NFKD", out).encode("ascii", "ignore").decode().lower()
    out = re.sub(r"[^a-z]", "", out)
    if len(out) > 3 and re.search(r"[^aeiou]a$", out):
        out = out[:-1]
    return out


def to_latin(names: pl.Series, dct: dict[str, str]) -> pl.Series:
    """Names with Indic characters get every Indic run replaced (dictionary first, rules as fallback)."""
    uniq = names.filter(names.str.contains(INDIC)).unique().to_list()
    cache, hits, seen = {}, 0, 0

    def word(m: re.Match) -> str:
        nonlocal hits, seen
        t = unicodedata.normalize("NFKD", m.group(0))
        seen += 1
        if t in dct:
            hits += 1
            return dct[t]
        if t not in cache:
            cache[t] = rule_translit(t)
        return cache[t]

    mapping = {n: INDIC_RUN.sub(word, n) for n in uniq}
    log.info("to_latin: %d unique Indic names; dictionary covers %.3f of Indic word tokens (%d rule-based words)",
             len(uniq), hits / max(seen, 1), len(cache))
    return names.replace_strict(mapping, default=names, return_dtype=pl.String)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True, help="A or A,B")
    a = ap.parse_args()
    load_dict(a.worlds.split(","))
