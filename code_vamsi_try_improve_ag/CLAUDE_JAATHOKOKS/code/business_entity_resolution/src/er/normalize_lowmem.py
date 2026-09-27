"""Low-RAM entry point for normalize.py (which stays verbatim).

split_region() is row-wise given the region values, so running it on row slices and concatenating in order
gives exactly the same result with a fraction of the peak memory (the explode over all address components
of a 6.5M-record world does not fit in 16 GB).

Run:  python -m er.normalize_lowmem --world B --translit-from A --suffix _v2 [--rows 500000]
"""
import argparse
import gc

import polars as pl

from . import normalize

_split_region = normalize.split_region
ROWS = 500_000


def split_region_chunked(df: pl.DataFrame, region_vals: pl.DataFrame) -> pl.DataFrame:
    if len(df) <= ROWS:
        return _split_region(df, region_vals)
    out = []
    for i in range(0, len(df), ROWS):
        out.append(_split_region(df.slice(i, ROWS), region_vals))
        gc.collect()
    return pl.concat(out, how="vertical_relaxed")


normalize.split_region = split_region_chunked

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--translit-from", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rows", type=int, default=ROWS)
    a = ap.parse_args()
    ROWS = a.rows
    normalize.run(a.world, a.force, a.translit_from, a.suffix)
