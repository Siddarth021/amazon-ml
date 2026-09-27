"""Logging, resumability and atomic IO helpers shared by every stage."""
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from . import config


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stderr), logging.FileHandler(config.LOGS / f"{name}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def peak_rss_gb() -> float:
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**30  # peak_wset exists on Windows
    except ImportError:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def stage_done(outs: list[Path], log: logging.Logger, force: bool) -> bool:
    if not force and all(Path(o).exists() for o in outs):
        log.info("skip, outputs exist: %s", ", ".join(str(o) for o in outs))
        return True
    return False


@contextmanager
def timed(log: logging.Logger, msg: str):
    log.info("%s ...", msg)
    t = time.time()
    yield
    log.info("%s done in %.1fs (peak RSS %.1f GB)", msg, time.time() - t, peak_rss_gb())


def write_parquet(df: pl.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
