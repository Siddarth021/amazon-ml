"""Inference: load trained models, predict on test features, post-process, and export."""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import time
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_inference():
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info("=== STARTING INFERENCE V9 ===")
    
    # Load test features
    logging.info("Loading test features...")
    df = pd.read_parquet(f"{processed_dir}/test_features.parquet")
    
    import glob
    pipeline_dirs = sorted(glob.glob("runs/pipeline_*"))
    run_dir = pipeline_dirs[-1] if pipeline_dirs else "output"
    
    # Load meta
    with open(f"{run_dir}/meta.json", "r") as f:
        meta = json.load(f)
    features = meta['features']
    optimal_threshold = meta.get('optimal_threshold', 0.85)
    
    logging.info(f"Features: {len(features)} | Threshold: {optimal_threshold}")
    
    # Load models
    logging.info("Loading XGBoost model...")
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(f"{run_dir}/xgb_model.json")
    
    logging.info("Loading LightGBM model...")
    model_lgb = lgb.Booster(model_file=f"{run_dir}/lgb_model.txt")
    
    # Predict in chunks
    logging.info("Predicting probabilities (ensemble)...")
    chunk_size = 500000
    probs = np.zeros(len(df), dtype=np.float32)
    
    for i in range(0, len(df), chunk_size):
        end = min(i + chunk_size, len(df))
        chunk_X = df.iloc[i:end][features].values.astype(np.float32)
        
        xgb_preds = model_xgb.predict_proba(chunk_X)[:, 1]
        lgb_preds = model_lgb.predict(chunk_X)
        
        probs[i:end] = (xgb_preds + lgb_preds) / 2.0
        
        if (i // chunk_size + 1) % 10 == 0:
            logging.info(f"  Predicted {end:,} / {len(df):,}")
    
    df['prob'] = probs
    logging.info(f"Prediction complete. Mean prob: {probs.mean():.4f}")
    
    # Apply threshold
    predictions = df[df['prob'] >= optimal_threshold].copy()
    logging.info(f"Candidates above threshold {optimal_threshold}: {len(predictions):,}")
    
    # Post-processing: 1-to-1 S2/S3 constraint
    logging.info("Applying 1-to-1 S2/S3 constraint...")
    predictions = predictions.sort_values('prob', ascending=False)
    
    # 1. An S2/S3 entity can only be matched to one S1
    predictions = predictions.drop_duplicates(subset=['candidate_entity_id'])
    
    # 2. An S1 entity can only have at most one S2 and at most one S3
    predictions['cand_src'] = predictions['candidate_entity_id'].str[:2]
    predictions = predictions.drop_duplicates(subset=['source1_entity_id', 'cand_src'])
    
    logging.info(f"After 1-to-1 constraint: {len(predictions):,}")
    
    # Generate matching_results.tsv
    logging.info("Formatting matching_results.tsv...")
    grouped = predictions.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    s1_test = pd.read_parquet(f"{processed_dir}/test_s1.parquet", columns=['entity_id'])
    s1_all = pd.DataFrame({'source1_entity_id': s1_test['entity_id']})
    
    final_output = pd.merge(s1_all, grouped, on='source1_entity_id', how='left')
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna('')
    
    os.makedirs("output", exist_ok=True)
    final_output.to_csv(f"{run_dir}/matching_results.tsv", sep='\t', index=False)
    logging.info(f"Saved matching_results.tsv ({len(final_output):,} rows) to {run_dir}")
    
    # Generate candidate_pairs.tsv
    logging.info("Formatting candidate_pairs.tsv...")
    cand_grouped = df.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    cand_grouped.columns = ['source1_entity_id', 'candidate_entity_ids']
    
    cand_output = pd.merge(s1_all, cand_grouped, on='source1_entity_id', how='left')
    cand_output['candidate_entity_ids'] = cand_output['candidate_entity_ids'].fillna('')
    
    cand_output.to_csv(f"{run_dir}/candidate_pairs.tsv", sep='\t', index=False)
    logging.info(f"Saved candidate_pairs.tsv ({len(cand_output):,} rows) to {run_dir}")
    
    # Stats
    matched = final_output[final_output['matched_entity_ids'] != '']
    singletons = final_output[final_output['matched_entity_ids'] == '']
    logging.info(f"Matched entities: {len(matched):,} | Singletons: {len(singletons):,}")
    
    logging.info(f"Inference completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    run_inference()
