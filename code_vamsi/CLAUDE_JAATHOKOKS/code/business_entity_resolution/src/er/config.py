"""Paths and global constants. ER_ROOT overrides the project root (the folder that contains dataset/)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("ER_ROOT", Path(__file__).resolve().parents[4]))
DATA = ROOT / "dataset"
WORK = ROOT / "work"
WORLDS = WORK / "worlds"
MODELS = WORK / "models"
PREDS = WORK / "preds"
LOGS = Path(os.environ.get("ER_LOGS", WORK / "logs"))
OUTPUT = Path(os.environ.get("ER_OUTPUT", ROOT / "output"))
SEED = 2026
FEAT_CHUNK = int(os.environ.get("ER_FEAT_CHUNK", 2_000_000))  # pairs per feature part (4M on >= 64 GB RAM)
A_SAMPLE = 0.2  # fraction of world A's records whose candidates get features (stage-1 / CE training)

for _d in (WORK, WORLDS, MODELS, PREDS, LOGS, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)
