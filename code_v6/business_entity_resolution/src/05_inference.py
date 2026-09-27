import pandas as pd
import numpy as np
import xgboost as xgb
import time
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_inference():
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info("=== STARTING INFERENCE ===")
    
    # Load test features
    logging.info("Loading test features...")
    df = pd.read_parquet(f"{processed_dir}/test_features.parquet")
    
    # Load meta
    with open("output/meta.json", "r") as f:
        meta = json.load(f)
    features = meta['features']
    optimal_threshold = meta.get('optimal_threshold', 0.85)
    
    logging.info(f"Loaded {len(features)} features. Optimal threshold: {optimal_threshold}")
    
    # Load model
    logging.info("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model("output/xgboost_model.json")
    
    # Predict
    logging.info("Predicting probabilities...")
    # Predict in chunks if large
    chunk_size = 500000
    probs = np.zeros(len(df))
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size][features]
        probs[i:i+len(chunk)] = model.predict_proba(chunk)[:, 1]
        
    df['prob'] = probs
    
    # Apply Threshold
    predictions = df[df['prob'] >= optimal_threshold].copy()
    
    logging.info(f"Candidates above threshold {optimal_threshold}: {len(predictions)}")
    
    # Post-processing: 1-to-1 S2/S3 constraint
    logging.info("Applying 1-to-1 constraint...")
    predictions = predictions.sort_values('prob', ascending=False)
    predictions = predictions.drop_duplicates(subset=['candidate_entity_id'])
    
    logging.info(f"Candidates after 1-to-1 constraint: {len(predictions)}")
    
    # Generate Output
    logging.info("Formatting matching_results.tsv...")
    grouped = predictions.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    grouped.columns = ['source1_entity_id', 'matched_entity_ids']
    
    s1_test = pd.read_parquet(f"{processed_dir}/test_s1.parquet", columns=['entity_id'])
    s1_all = pd.DataFrame({'source1_entity_id': s1_test['entity_id']})
    
    final_output = pd.merge(s1_all, grouped, on='source1_entity_id', how='left')
    final_output['matched_entity_ids'] = final_output['matched_entity_ids'].fillna('')
    
    os.makedirs("output", exist_ok=True)
    final_output.to_csv("output/matching_results.tsv", sep='\t', index=False)
    logging.info("Saved final predictions to output/matching_results.tsv")
    
    logging.info(f"Inference completed in {time.time() - t0:.2f} s")

if __name__ == "__main__":
    run_inference()
