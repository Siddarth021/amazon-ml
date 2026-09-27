import polars as pl
import numpy as np
import sys
sys.path.insert(0, "code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/code/business_entity_resolution/src")
from er.evaluate import truth_pairs

s1 = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/s1.parquet")
rec = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/worlds/B/rec.parquet")
t_pairs = truth_pairs(rec)

pred = pl.read_parquet("code_vamsi_try_improve_ag/CLAUDE_JAATHOKOKS/work/preds/ctx_v6_B.parquet")

sorted_pred = pred.sort(["rec_idx", "p"], descending=[False, True])
ranked = sorted_pred.with_columns(rnk=pl.int_range(0, pl.len()).over("rec_idx"))
top1 = ranked.filter(pl.col("rnk") == 0).drop("rnk")
top2 = ranked.filter(pl.col("rnk") == 1).select("rec_idx", p2=pl.col("p"))
best = top1.join(top2, on="rec_idx", how="left").with_columns(
    margin=pl.col("p") - pl.col("p2").fill_null(0.0)
).join(s1.select("s1_idx", "country"), on="s1_idx")

# Ground truth singleton S1s (n_true == 0)
nt = t_pairs.group_by("s1_idx").len("n_true")
s1_with_nt = s1.join(nt, on="s1_idx", how="left").with_columns(pl.col("n_true").fill_null(0))
true_singleton_s1s = set(s1_with_nt.filter(pl.col("n_true") == 0)["s1_idx"].to_list())

rule_in = (pl.col("country") == "India") & ((pl.col("p") >= 0.90) | ((pl.col("p") >= 0.80) & (pl.col("margin") >= 0.75)))
rule_us = (pl.col("country") == "US") & ((pl.col("p") >= 0.91) | ((pl.col("p") >= 0.85) & (pl.col("margin") >= 0.75)))
accepted = best.filter(rule_in | rule_us)

# S1s that were accepted and are in true_singleton_s1s
fp_on_singleton = accepted.filter(pl.col("s1_idx").is_in(true_singleton_s1s))
print(f"Number of false positive matches hitting true singletons: {len(fp_on_singleton)}")
print(f"Unique true singleton S1s hit: {fp_on_singleton['s1_idx'].n_unique()}")

# What are the properties of these false positive pairs?
print("P quantiles for FP on singletons:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"  q{int(q*100)}: {fp_on_singleton['p'].quantile(q):.4f}")

print("Margin quantiles for FP on singletons:")
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    print(f"  q{int(q*100)}: {fp_on_singleton['margin'].quantile(q):.4f}")

# How many records were accepted for these S1s?
s1_pred_count = accepted.group_by("s1_idx").len("s1_n_pred")
fp_on_singleton = fp_on_singleton.join(s1_pred_count, on="s1_idx", how="left")
print("S1 n_pred distribution for FP on singletons:")
print(fp_on_singleton["s1_n_pred"].value_counts())
