# Business entity resolution (Amazon ML Challenge 2026)

Matches Source 2 / Source 3 business records to Source 1 entities. No external data: only the provided TSVs and the
`FacebookAI/xlm-roberta-base` weights (MIT) are used.

## Layout

```
<root>/dataset/{train,test}/*.tsv        challenge data
<root>/utils/validate_submission.py      official validator
<root>/code/business_entity_resolution/  this package (src/er/*)
<root>/work/                             worlds, features, models, predictions, logs (created at runtime)
<root>/output/                           matching_results.tsv, candidate_pairs.tsv
```

## Run

```bash
pip install -r code/business_entity_resolution/requirements.txt
bash code/business_entity_resolution/scripts/run_all.sh        # from <root>
```

Every stage is resumable: finished outputs are skipped, long stages write `part_*.parquet` files and continue from
the last finished part. Pass `--force` to a module to recompute it. Logs: `work/logs/<step>.out`.
Single stage: `PYTHONPATH=code/business_entity_resolution/src python -m er.<module> ...`.

## Pipeline

| # | module | output |
|---|---|---|
| 1 | `data` | worlds A / B (training S1 split 50/50, all distractors in both), dev, test |
| 2 | `translit` | Indic→Latin word dictionary learned from aligned true pairs (A; A+B for test) |
| 3 | `normalize` (`normalize_lowmem` = same result, lower RAM) | name/address views, learned regions, rare tokens |
| 4 | `blocking` | ~15 keys + TF-IDF kNN, top-50 per record by key specificity |
| 5 | `features`, `numfeat`, `namefeat` | ~55 + 5 + 2 pair features, aligned parts |
| 6 | `lgbm_pair` | stage-1 LightGBM p; shortlist (p ≥ 0.01, top 3 per record) |
| 7 | `cross_encoder` | XLM-R cross-encoder ce_v3 (A), ce_v4 (test pseudo-labels + A replay) |
| 8 | `context` | stacker LightGBM on B (2-fold cross-fit) with competition features |
| 9 | `decide` | argmax S1 per record, accept if p ≥ T (tuned on B out-of-fold) |
| 10 | `write_output` | submission TSVs + official validator |

## Low-memory settings (16 GB RAM)

`run_all.sh` defaults: `SHARD=120000`, `TF_CHUNK=2000`, `POLARS_MAX_THREADS=4`, `ER_FEAT_CHUNK=2000000`. None of them
changes results (shards, TF-IDF batches and feature parts are independent). The stage-1 and stacker training sets
are capped by `--max-train-rows` (negatives subsampled, weight 1/fraction).
On ≥ 64 GB: `SHARD=1500000 TF_CHUNK=20000 ER_FEAT_CHUNK=4000000`.

## Tests

`PYTHONPATH=src python -m pytest tests` (scorer vs the README example: 0.7142857).
