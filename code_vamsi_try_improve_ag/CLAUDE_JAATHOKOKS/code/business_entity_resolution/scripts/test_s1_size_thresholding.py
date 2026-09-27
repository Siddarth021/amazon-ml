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

# Candidate set with broad acceptance
cand = best.filter((pl.col("p") >= 0.75) | (pl.col("margin") >= 0.65))
s1_cnt = cand.group_by("s1_idx").len("s1_cluster")
best = best.join(s1_cnt, on="s1_idx", how="left").with_columns(
    s1_cluster=pl.col("s1_cluster").fill_null(0)
)

print("Sweeping cluster-based rules...")
best_overall = 0.990100

for t_single in [0.895, 0.900, 0.905, 0.910, 0.915]:
    for t_multi in [0.850, 0.860, 0.870, 0.880, 0.885, 0.890]:
        for m_single in [0.75, 0.80, 0.85]:
            # India rule
            rule_in = (pl.col("country") == "India") & (
                ((pl.col("s1_cluster") >= 2) & ((pl.col("p") >= t_multi) | ((pl.col("p") >= 0.78) & (pl.col("margin") >= 0.70)))) |
                ((pl.col("s1_cluster") <= 1) & ((pl.col("p") >= t_single) | ((pl.col("p") >= 0.82) & (pl.col("margin") >= m_single))))
            )
            # US rule
            rule_us = (pl.col("country") == "US") & (
                ((pl.col("s1_cluster") >= 2) & ((pl.col("p") >= t_multi + 0.01) | ((pl.col("p") >= 0.82) & (pl.col("margin") >= 0.72)))) |
                ((pl.col("s1_cluster") <= 1) & ((pl.col("p") >= t_single + 0.01) | ((pl.col("p") >= 0.85) & (pl.col("margin") >= m_single))))
            )
            p_sub = best.filter(rule_in | rule_us).select("s1_idx", "rec_idx")
            res = score(s1, t_pairs, p_sub)
            if res["f05"] > best_overall:
                best_overall = res["f05"]
                print(f"NEW BEST F0.5 = {best_overall:.6f} (India: {res['by_country']['India']:.6f}, US: {res['by_country']['US']:.6f}) with t_single={t_single}, t_multi={t_multi}, m_single={m_single}")

print(f"\nFinal Best F0.5: {best_overall:.6f}")
