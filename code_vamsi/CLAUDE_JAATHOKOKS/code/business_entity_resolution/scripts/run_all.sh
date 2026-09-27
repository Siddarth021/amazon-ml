#!/usr/bin/env bash
# Full pipeline, in order. Every step is resumable (skips finished outputs), so rerunning continues where it stopped.
# Run from the project root (the folder containing dataset/):  bash code/business_entity_resolution/scripts/run_all.sh
# Low-RAM (16 GB) settings are the defaults below; on >= 64 GB use SHARD=1500000 TF_CHUNK=20000 ER_FEAT_CHUNK=4000000.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
export ER_ROOT="$ROOT"
export PYTHONPATH="$ROOT/code/business_entity_resolution/src"
export PYTHONIOENCODING=utf-8
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-4}"
SHARD="${SHARD:-120000}"
TF_CHUNK="${TF_CHUNK:-2000}"
MAX_CAND="${MAX_CAND:-50}"
LOGS="${ER_LOGS:-$ROOT/work/logs}"
mkdir -p "$LOGS"

step() {  # step <log name> <module> [args...]  -- retries a crashed stage up to 2 times
  local name="$1"; shift
  for attempt in 1 2 3; do
    echo "[$(date +%T)] $name (attempt $attempt)"
    if python -m "$@" >> "$LOGS/$name.out" 2>&1; then return 0; fi
  done
  echo "FAILED: $name (see $LOGS/$name.out)"; exit 1
}

step data            er.data
step translit_A      er.translit --worlds A
step translit_A_B    er.translit --worlds A,B
for w in A B; do
  step normalize_$w  er.normalize_lowmem --world $w --translit-from A --suffix _v2
done
step normalize_test  er.normalize_lowmem --world test --translit-from A,B --suffix _v2
for w in B A test; do   # check B's recall / ceiling before going further
  step blocking_$w   er.blocking_lowmem --world $w --norm _v2 --out cand_v1 --shard $SHARD --tf-chunk $TF_CHUNK --max-cand $MAX_CAND
done
for w in A B test; do
  step features_$w   er.features --world $w
  step numfeat_$w    er.numfeat  --world $w
  step namefeat_$w   er.namefeat --world $w
done
step lgbm_train      er.lgbm_pair train
for w in A B test; do
  step lgbm_pred_$w  er.lgbm_pair predict   --world $w
  step shortlist_$w  er.lgbm_pair shortlist --world $w
done
step stage1_check    er.lgbm_pair check --world B

step ce_v3_pairs     er.cross_encoder pairs --name ce_v3
step ce_v3_train     er.cross_encoder train --name ce_v3
for w in B test; do step ce_v3_pred_$w er.cross_encoder predict --name ce_v3 --world $w; done
step ctx_v3_fit      er.context fit     --world B    --ce ce_v3 --name ctx_v3
step ctx_v3_pred     er.context predict --world test --ce ce_v3 --name ctx_v3

step ce_v4_pairs     er.cross_encoder pseudo --name ce_v4 --stacker ctx_v3 --replay ce_v3
step ce_v4_train     er.cross_encoder train  --name ce_v4 --init ce_v3 --lr 1e-5
for w in B test; do step ce_v4_pred_$w er.cross_encoder predict --name ce_v4 --world $w; done

step ctx_v6_fit      er.context fit     --world B    --ce ce_v3,ce_v4 --numfeat numfeat,namefeat --name ctx_v6
step decide_tune     er.decide tune     --world B    --name ctx_v6
step ctx_v6_pred     er.context predict --world test --ce ce_v3,ce_v4 --numfeat numfeat,namefeat --name ctx_v6
step decide_write    er.decide write    --world test --name ctx_v6
step write_output    er.write_output --cand cand_v1
echo "done: $ROOT/output/matching_results.tsv, $ROOT/output/candidate_pairs.tsv"
