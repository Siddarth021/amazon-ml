import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from rapidfuzz import distance, fuzz

# Add src dir to path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(src_dir)

import importlib
preprocess_module = importlib.import_module("01_preprocess")
blocking_module = importlib.import_module("02_blocking")

process_file_parallel = preprocess_module.process_file_parallel
get_candidates_for_country = blocking_module.get_candidates_for_country

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, log_file=None):
    formatted_msg = f"[{get_timestamp()}] {msg}"
    print(formatted_msg, flush=True)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(formatted_msg + "\n")
        except Exception:
            pass

def compute_features_chunk(chunk_df, dict_s1_name, dict_s1_addr, dict_s23_name, dict_s23_addr):
    s1_ids = chunk_df['source1_entity_id'].values
    c23_ids = chunk_df['candidate_entity_id'].values
    
    n = len(chunk_df)
    f_name_jaro = np.zeros(n, dtype=np.float32)
    f_name_sort = np.zeros(n, dtype=np.float32)
    f_name_part = np.zeros(n, dtype=np.float32)
    f_addr_jaro = np.zeros(n, dtype=np.float32)
    f_addr_set = np.zeros(n, dtype=np.float32)
    f_exact_name = np.zeros(n, dtype=np.float32)
    
    for i in range(n):
        s1_id = s1_ids[i]
        c23_id = c23_ids[i]
        
        n1 = dict_s1_name.get(s1_id, '')
        a1 = dict_s1_addr.get(s1_id, '')
        n2 = dict_s23_name.get(c23_id, '')
        a2 = dict_s23_addr.get(c23_id, '')
        
        f_name_jaro[i] = distance.JaroWinkler.similarity(n1, n2) if (n1 and n2) else 0.0
        f_name_sort[i] = fuzz.token_sort_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
        f_name_part[i] = fuzz.partial_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
        
        f_addr_jaro[i] = distance.JaroWinkler.similarity(a1, a2) if (a1 and a2) else 0.0
        f_addr_set[i] = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
        f_exact_name[i] = 1.0 if (n1 and n2 and n1 == n2) else 0.0
        
    return np.column_stack([
        f_name_jaro, f_name_sort, f_name_part,
        f_addr_jaro, f_addr_set, f_exact_name
    ])

def main():
    base_dir = os.path.abspath(os.path.join(src_dir, ".."))
    root_dir = os.path.abspath(os.path.join(base_dir, "..", ".."))
    
    dataset_test_dir = os.path.join(root_dir, "student_resource", "dataset", "test")
    processed_dir = os.path.join(base_dir, "processed")
    output_dir = os.path.join(base_dir, "output")
    model_path = os.path.join(output_dir, "xgboost_model.json")
    log_file = os.path.join(base_dir, "train.log")
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    log("=" * 60, log_file)
    log("TEST INFERENCE PIPELINE STARTED", log_file)
    log("=" * 60, log_file)
    
    # -------------------------------------------------------------
    # STAGE 1: PREPROCESSING TEST DATASET
    # -------------------------------------------------------------
    clean_test_s1_path = os.path.join(processed_dir, "clean_test_s1.tsv")
    clean_test_s2_path = os.path.join(processed_dir, "clean_test_s2.tsv")
    clean_test_s3_path = os.path.join(processed_dir, "clean_test_s3.tsv")
    
    if not (os.path.exists(clean_test_s1_path) and os.path.exists(clean_test_s2_path) and os.path.exists(clean_test_s3_path)):
        log("STEP 1: Preprocessing raw test set files...", log_file)
        process_file_parallel(os.path.join(dataset_test_dir, "test_source1.tsv"), clean_test_s1_path, log_file)
        process_file_parallel(os.path.join(dataset_test_dir, "test_source2.tsv"), clean_test_s2_path, log_file)
        process_file_parallel(os.path.join(dataset_test_dir, "test_source3.tsv"), clean_test_s3_path, log_file)
    else:
        log("Preprocessed clean test files found in processed/. Skipping Step 1.", log_file)
        
    # -------------------------------------------------------------
    # STAGE 2: CANDIDATE BLOCKING ON TEST DATASET
    # -------------------------------------------------------------
    test_cand_path = os.path.join(processed_dir, "test_candidate_pairs.parquet")
    if not os.path.exists(test_cand_path):
        log("STEP 2: Inverted Index candidate blocking on test dataset...", log_file)
        df_s1 = pd.read_csv(clean_test_s1_path, sep='\t', low_memory=False)
        df_s2 = pd.read_csv(clean_test_s2_path, sep='\t', low_memory=False)
        df_s3 = pd.read_csv(clean_test_s3_path, sep='\t', low_memory=False)
        df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
        
        countries = df_s1['country'].fillna('UNKNOWN').unique()
        log(f"Blocking test set across {len(countries)} countries: {list(countries)}", log_file)
        
        all_candidates = []
        for country in countries:
            log(f"Processing test country block: '{country}'...", log_file)
            c_s1 = df_s1[df_s1['country'].fillna('UNKNOWN') == country]
            c_s23 = df_s23[df_s23['country'].fillna('UNKNOWN') == country]
            log(f"  - S1 rows: {len(c_s1):,}, S23 rows: {len(c_s23):,}", log_file)
            if len(c_s1) > 0 and len(c_s23) > 0:
                cands = get_candidates_for_country(c_s1, c_s23, top_k=15, log_file=log_file)
                all_candidates.append(cands)
                
        final_candidates = pd.concat(all_candidates, ignore_index=True)
        log(f"Total test candidate pairs generated: {len(final_candidates):,}", log_file)
        final_candidates.to_parquet(test_cand_path, index=False)
    else:
        log("Test candidate pairs found in processed/test_candidate_pairs.parquet. Skipping Step 2.", log_file)
        
    # -------------------------------------------------------------
    # STAGE 3: FEATURE EXTRACTION & XGBOOST INFERENCE
    # -------------------------------------------------------------
    log("STEP 3: Loading preprocessed test data into lookup maps...", log_file)
    df_s1 = pd.read_csv(clean_test_s1_path, sep='\t', usecols=['entity_id', 'clean_name', 'clean_address'], low_memory=False)
    df_s2 = pd.read_csv(clean_test_s2_path, sep='\t', usecols=['entity_id', 'clean_name', 'clean_address'], low_memory=False)
    df_s3 = pd.read_csv(clean_test_s3_path, sep='\t', usecols=['entity_id', 'clean_name', 'clean_address'], low_memory=False)
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    dict_s1_name = dict(zip(df_s1['entity_id'].values, df_s1['clean_name'].fillna('').values))
    dict_s1_addr = dict(zip(df_s1['entity_id'].values, df_s1['clean_address'].fillna('').values))
    
    dict_s23_name = dict(zip(df_s23['entity_id'].values, df_s23['clean_name'].fillna('').values))
    dict_s23_addr = dict(zip(df_s23['entity_id'].values, df_s23['clean_address'].fillna('').values))
    
    s1_all_ids = df_s1['entity_id'].values
    del df_s1, df_s2, df_s3, df_s23
    
    log("Loading test candidate pairs...", log_file)
    cands_df = pd.read_parquet(test_cand_path)
    total_pairs = len(cands_df)
    log(f"Test candidate pairs count: {total_pairs:,}", log_file)
    
    log(f"Loading trained XGBoost model from {model_path}...", log_file)
    bst = xgb.Booster()
    bst.load_model(model_path)
    
    log("Extracting features and predicting probabilities in chunks...", log_file)
    chunk_size = 500000
    all_probs = []
    
    t_feat_start = time.time()
    for start_idx in range(0, total_pairs, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pairs)
        chunk_cands = cands_df.iloc[start_idx:end_idx]
        
        feats = compute_features_chunk(chunk_cands, dict_s1_name, dict_s1_addr, dict_s23_name, dict_s23_addr)
        dmatrix = xgb.DMatrix(feats)
        probs = bst.predict(dmatrix)
        all_probs.append(probs)
        
        if (end_idx % 2000000 == 0) or (end_idx == total_pairs):
            log(f"  * Inference progress: {end_idx:,} / {total_pairs:,} ({end_idx/total_pairs*100:.1f}%)...", log_file)
            
    all_probs = np.concatenate(all_probs)
    log(f"Inference completed in {time.time() - t_feat_start:.2f} seconds.", log_file)
    
    # Apply optimal threshold 0.90
    threshold = 0.90
    log(f"Applying optimal classification threshold: {threshold}...", log_file)
    cands_df['match_pred'] = (all_probs >= threshold)
    
    # -------------------------------------------------------------
    # STAGE 4: GENERATE OUTPUT TSV FILES
    # -------------------------------------------------------------
    # 1. candidate_pairs.tsv
    out_cand_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    log(f"Writing {out_cand_tsv} line by line...", log_file)
    
    cand_dict = {}
    for s1_id, match_id in zip(cands_df['source1_entity_id'].values, cands_df['candidate_entity_id'].values):
        if s1_id not in cand_dict:
            cand_dict[s1_id] = [match_id]
        else:
            cand_dict[s1_id].append(match_id)
            
    with open(out_cand_tsv, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_all_ids:
            if s1_id in cand_dict:
                f.write(f"{s1_id}\t{','.join(cand_dict[s1_id])}\n")
            else:
                f.write(f"{s1_id}\t\n")
    log(f"Saved {out_cand_tsv} successfully.", log_file)
    
    # 2. matching_results.tsv
    out_match_tsv = os.path.join(output_dir, "matching_results.tsv")
    log(f"Writing {out_match_tsv} line by line...", log_file)
    
    matched_df = cands_df[cands_df['match_pred'] == True]
    match_dict = {}
    for s1_id, match_id in zip(matched_df['source1_entity_id'].values, matched_df['candidate_entity_id'].values):
        if s1_id not in match_dict:
            match_dict[s1_id] = [match_id]
        else:
            match_dict[s1_dict].append(match_id) if 's1_dict' in locals() else match_dict[s1_id].append(match_id)
            
    with open(out_match_tsv, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_ids:
            if s1_id in match_dict:
                f.write(f"{s1_id}\t{','.join(match_dict[s1_id])}\n")
            else:
                f.write(f"{s1_id}\t\n")
    log(f"Saved {out_match_tsv} successfully.", log_file)
    log("=" * 60, log_file)
    log("TEST INFERENCE PIPELINE COMPLETED SUCCESSFULLY!", log_file)
    log("=" * 60, log_file)

if __name__ == "__main__":
    main()
