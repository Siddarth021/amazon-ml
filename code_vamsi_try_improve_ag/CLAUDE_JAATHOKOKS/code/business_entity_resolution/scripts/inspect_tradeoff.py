import polars as pl
import numpy as np
import sys
sys.path.insert(0, "code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/code/business_entity_resolution/src")
from er.evaluate import score, truth_pairs

s1 = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/s1.parquet", columns=["s1_idx", "country"])
rec = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/rec.parquet", columns=["rec_idx", "s1_idx"])
t_pairs = truth_pairs(rec)

pred = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/preds/ctx_v6_B.parquet")
best = (pred.sort(["rec_idx", "p"], descending=[False, True]).group_by("rec_idx", maintain_order=True).first().select("s1_idx", "rec_idx", "p"))

for th in [0.70, 0.80, 0.85, 0.88, 0.895, 0.91, 0.93, 0.95]:
    p_sub = best.filter(pl.col("p") >= th).select("s1_idx", "rec_idx")
    res = score(s1, t_pairs, p_sub)
    loss_fp = res["loss"]["false_matches"] + res["loss"]["singleton_false_match"]
    loss_fn = res["loss"]["missed_matches"] + res["loss"]["fully_missed"]
    print(f"T={th:.3f} | F0.5={res['f05']:.6f} | P_mac={res['precision_macro']:.4f} | R_mac={res['recall_macro']:.4f} | TP={res['tp']} | FP={res['fp']} | FN={res['fn']} | Loss_FP={loss_fp:.6f} | Loss_FN={loss_fn:.6f}")
