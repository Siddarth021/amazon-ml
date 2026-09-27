import polars as pl
import numpy as np
import sys
sys.path.insert(0, "code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/code/business_entity_resolution/src")
from er.evaluate import score, truth_pairs, per_s1

s1 = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/s1.parquet", columns=["s1_idx", "country"])
rec = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/rec.parquet", columns=["rec_idx", "s1_idx"])
t_pairs = truth_pairs(rec)

pred = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/preds/ctx_v6_B.parquet")

# Calculate top candidates per record
sorted_pred = pred.sort(["rec_idx", "p"], descending=[False, True])
ranked = sorted_pred.with_columns(rnk=pl.int_range(0, pl.len()).over("rec_idx"))
top1 = ranked.filter(pl.col("rnk") == 0).drop("rnk")
top2 = ranked.filter(pl.col("rnk") == 1).select("rec_idx", p2=pl.col("p"))
best = top1.join(top2, on="rec_idx", how="left").with_columns(
    margin=pl.col("p") - pl.col("p2").fill_null(0.0)
)
print("Best with margin shape:", best.shape)
print(best.head())

# Label each candidate with whether it is a true positive
best_labeled = best.join(
    t_pairs.with_columns(is_true=pl.lit(True)),
    on=["s1_idx", "rec_idx"],
    how="left"
).with_columns(is_true=pl.col("is_true").fill_null(False))

# Look at candidates with p between 0.80 and 0.895
borderline = best_labeled.filter((pl.col("p") >= 0.80) & (pl.col("p") < 0.895))
print(f"Borderline pairs [0.80, 0.895): total={len(borderline)}, true={borderline['is_true'].sum()} ({borderline['is_true'].mean()*100:.2f}%)")

# Compare margin for true vs false in borderline
true_bl = borderline.filter(pl.col("is_true"))
false_bl = borderline.filter(~pl.col("is_true"))
print("True borderline margin quantiles (q25, q50, q75):",
      true_bl["margin"].quantile(0.25), true_bl["margin"].quantile(0.50), true_bl["margin"].quantile(0.75))
print("False borderline margin quantiles (q25, q50, q75):",
      false_bl["margin"].quantile(0.25), false_bl["margin"].quantile(0.50), false_bl["margin"].quantile(0.75))

# Look at S1 cluster size of the candidate:
# Does the S1 already have other high-confidence records (p >= 0.90)?
s1_high_conf_counts = (best_labeled.filter(pl.col("p") >= 0.90)
                       .group_by("s1_idx").len("s1_high_conf_count"))

best_labeled = best_labeled.join(s1_high_conf_counts, on="s1_idx", how="left").with_columns(
    s1_high_conf_count=pl.col("s1_high_conf_count").fill_null(0)
)

borderline = best_labeled.filter((pl.col("p") >= 0.80) & (pl.col("p") < 0.895))
true_bl = borderline.filter(pl.col("is_true"))
false_bl = borderline.filter(~pl.col("is_true"))

print("True borderline s1_high_conf_count == 0:", (true_bl["s1_high_conf_count"] == 0).mean())
print("False borderline s1_high_conf_count == 0:", (false_bl["s1_high_conf_count"] == 0).mean())
