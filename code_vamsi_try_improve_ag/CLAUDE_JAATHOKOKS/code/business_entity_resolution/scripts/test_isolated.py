import polars as pl
import numpy as np
import sys
sys.path.insert(0, "code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/code/business_entity_resolution/src")
from er.evaluate import score, truth_pairs

s1 = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/s1.parquet", columns=["s1_idx", "country"])
rec = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/rec.parquet", columns=["rec_idx", "s1_idx"])
t_pairs = truth_pairs(rec)

pred = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/preds/ctx_v6_B.parquet")

sorted_pred = pred.sort(["rec_idx", "p"], descending=[False, True])
ranked = sorted_pred.with_columns(rnk=pl.int_range(0, pl.len()).over("rec_idx"))
top1 = ranked.filter(pl.col("rnk") == 0).drop("rnk")
top2 = ranked.filter(pl.col("rnk") == 1).select("rec_idx", p2=pl.col("p"))
best = top1.join(top2, on="rec_idx", how="left").with_columns(
    margin=pl.col("p") - pl.col("p2").fill_null(0.0)
).join(s1, on="s1_idx")

# Count how many candidate records belong to each S1 with p >= 0.85
s1_cand_count = (best.filter(pl.col("p") >= 0.85)
                 .group_by("s1_idx").len("s1_n_cand"))

best = best.join(s1_cand_count, on="s1_idx", how="left").with_columns(
    s1_n_cand=pl.col("s1_n_cand").fill_null(0)
)

# Test isolated vs multi-candidate precision:
# If s1_n_cand == 1 (isolated candidate): require higher threshold T_iso
# If s1_n_cand >= 2 (multi candidate): can accept lower threshold T_multi
base_score = score(s1, t_pairs, best.filter(pl.col("p") >= 0.895).select("s1_idx", "rec_idx"))["f05"]
print(f"Base score: {base_score:.6f}")

for t_iso in [0.895, 0.900, 0.905, 0.910, 0.915, 0.920]:
    for t_multi in [0.880, 0.885, 0.890, 0.895]:
        rule = ((pl.col("s1_n_cand") == 1) & (pl.col("p") >= t_iso)) | \
               ((pl.col("s1_n_cand") >= 2) & (pl.col("p") >= t_multi))
        p_sub = best.filter(rule).select("s1_idx", "rec_idx")
        res = score(s1, t_pairs, p_sub)
        print(f"T_iso={t_iso:.3f}, T_multi={t_multi:.3f} -> F0.5 = {res['f05']:.6f} (India: {res['by_country']['India']:.6f}, US: {res['by_country']['US']:.6f})")
