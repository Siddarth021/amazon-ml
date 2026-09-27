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

base_score = score(s1, t_pairs, best.filter(pl.col("p") >= 0.895).select("s1_idx", "rec_idx"))["f05"]
print(f"Base score at T=0.895: {base_score:.6f}")

# Grid search over T_base, T_low, and min_margin
best_overall = base_score
best_params = None

for t_base in [0.890, 0.895, 0.900]:
    for t_low in [0.80, 0.83, 0.85, 0.87]:
        for min_m in [0.60, 0.70, 0.75, 0.80, 0.82]:
            rule = (pl.col("p") >= t_base) | ((pl.col("p") >= t_low) & (pl.col("margin") >= min_m))
            p_sub = best.filter(rule).select("s1_idx", "rec_idx")
            res = score(s1, t_pairs, p_sub)
            sc = res["f05"]
            if sc > best_overall:
                best_overall = sc
                best_params = (t_base, t_low, min_m)
                print(f"NEW BEST: {sc:.6f} with T_base={t_base}, T_low={t_low}, margin={min_m}")

print(f"\nFinal Best Margin Rule: F0.5 = {best_overall:.6f} with params {best_params}")
