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

s1_in = s1.filter(pl.col("country") == "India")
s1_us = s1.filter(pl.col("country") == "US")
t_in = t_pairs.join(s1_in.select("s1_idx"), on="s1_idx")
t_us = t_pairs.join(s1_us.select("s1_idx"), on="s1_idx")

best_in = best.filter(pl.col("country") == "India")
best_us = best.filter(pl.col("country") == "US")

print("--- FINE TUNING INDIA ---")
best_sc_in = 0.991000
best_p_in = (0.90, 0.80, 0.75)
for t_base in np.round(np.arange(0.890, 0.925, 0.005), 3):
    for t_low in np.round(np.arange(0.76, 0.85, 0.02), 2):
        for min_m in np.round(np.arange(0.70, 0.82, 0.02), 2):
            rule = (pl.col("p") >= t_base) | ((pl.col("p") >= t_low) & (pl.col("margin") >= min_m))
            sc = score(s1_in, t_in, best_in.filter(rule).select("s1_idx", "rec_idx"))["f05"]
            if sc > best_sc_in:
                best_sc_in = sc
                best_p_in = (t_base, t_low, min_m)
                print(f"India BEST: {sc:.6f} with {best_p_in}")

print(f"\n--- FINE TUNING US ---")
best_sc_us = 0.989490
best_p_us = (0.91, 0.85, 0.75)
for t_base in np.round(np.arange(0.900, 0.935, 0.005), 3):
    for t_low in np.round(np.arange(0.80, 0.89, 0.02), 2):
        for min_m in np.round(np.arange(0.70, 0.82, 0.02), 2):
            rule = (pl.col("p") >= t_base) | ((pl.col("p") >= t_low) & (pl.col("margin") >= min_m))
            sc = score(s1_us, t_us, best_us.filter(rule).select("s1_idx", "rec_idx"))["f05"]
            if sc > best_sc_us:
                best_sc_us = sc
                best_p_us = (t_base, t_low, min_m)
                print(f"US BEST: {sc:.6f} with {best_p_us}")

print("\n--- OPTIMAL COMBINED EVALUATION ---")
rule_in = (pl.col("p") >= best_p_in[0]) | ((pl.col("p") >= best_p_in[1]) & (pl.col("margin") >= best_p_in[2]))
rule_us = (pl.col("p") >= best_p_us[0]) | ((pl.col("p") >= best_p_us[1]) & (pl.col("margin") >= best_p_us[2]))

comb = pl.concat([
    best_in.filter(rule_in).select("s1_idx", "rec_idx"),
    best_us.filter(rule_us).select("s1_idx", "rec_idx")
])
final_score = score(s1, t_pairs, comb)
print(f"BASELINE: F0.5 = 0.990061 (India: 0.990967, US: 0.989456)")
print(f"IMPROVED COMBINED: F0.5 = {final_score['f05']:.6f}")
print(f"By Country: {final_score['by_country']}")
print(f"Parameters: India={best_p_in}, US={best_p_us}")
