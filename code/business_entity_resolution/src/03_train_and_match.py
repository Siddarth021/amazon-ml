import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import xgboost as xgb
from sklearn.metrics import fbeta_score, precision_score, recall_score

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

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, "..", "..", ".."))
    base_dir = os.path.abspath(os.path.join(src_dir, ".."))
    processed_dir = os.path.join(base_dir, "processed")
    dataset_dir = os.path.join(root_dir, "student_resource", "dataset")
    output_dir = os.path.join(base_dir, "output")
    log_file = os.path.join(base_dir, "train.log")
    os.makedirs(output_dir, exist_ok=True)
    
    log("=" * 60, log_file)
    log("STEP 3: FEATURE EXTRACTION, XGBOOST TRAINING & MATCHING STARTED", log_file)
    log("=" * 60, log_file)
    
    step_start = time.time()
    log(f"Loading preprocessed clean dataset tables from {processed_dir}...", log_file)
    df_s1 = pd.read_csv(os.path.join(processed_dir, 'clean_train_s1.tsv'), sep='\t', low_memory=False)
    df_s2 = pd.read_csv(os.path.join(processed_dir, 'clean_train_s2.tsv'), sep='\t', low_memory=False)
    df_s3 = pd.read_csv(os.path.join(processed_dir, 'clean_train_s3.tsv'), sep='\t', low_memory=False)
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    log("Building fast dictionary lookups for string comparisons...", log_file)
    s1_names = df_s1.set_index('entity_id')['clean_name'].fillna('').astype(str).to_dict()
    s1_addrs = df_s1.set_index('entity_id')['clean_address'].fillna('').astype(str).to_dict()
    s23_names = df_s23.set_index('entity_id')['clean_name'].fillna('').astype(str).to_dict()
    s23_addrs = df_s23.set_index('entity_id')['clean_address'].fillna('').astype(str).to_dict()
    
    del df_s2, df_s3, df_s23
    
    log("Loading candidate pairs from candidate_pairs.parquet...", log_file)
    cands_path = os.path.join(processed_dir, 'candidate_pairs.parquet')
    cands = pd.read_parquet(cands_path)
    n_pairs = len(cands)
    log(f"Loaded {n_pairs:,} candidate pairs.", log_file)
    
    gt_path = os.path.join(dataset_dir, 'train', 'train_ground_truth.tsv')
    log(f"Loading ground truth labels from {gt_path} using fast streaming set reader...", log_file)
    gt_start = time.time()
    gt_set = set()
    with open(gt_path, 'r', encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                for c_id in parts[1].split(','):
                    if c_id:
                        gt_set.add((s1_id, c_id))
                        
    log(f"Ground truth set created: {len(gt_set):,} positive pairs in {time.time() - gt_start:.2f}s.", log_file)
    
    log("Labeling candidate pairs using chunked set membership...", log_file)
    match_start = time.time()
    s1_arr = cands['source1_entity_id'].values
    c23_arr = cands['candidate_entity_id'].values
    tfidf_sim_arr = cands['tfidf_sim'].values.astype(np.float32)
    
    del cands
    
    labels = np.zeros(n_pairs, dtype=np.int8)
    chunk_size = 1000000
    for i in range(0, n_pairs, chunk_size):
        end = min(i + chunk_size, n_pairs)
        sub_s1 = s1_arr[i:end]
        sub_c23 = c23_arr[i:end]
        labels[i:end] = [1 if (s, c) in gt_set else 0 for s, c in zip(sub_s1, sub_c23)]
        
    log(f"Ground truth labeling completed in {time.time() - match_start:.2f}s.", log_file)
    
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()
    log(f"Dataset statistics: {n_pairs:,} total candidate pairs | {pos_count:,} positive matches | {neg_count:,} negatives.", log_file)
    
    log(f"Extracting 6 RapidFuzz fuzzy string comparison features for {n_pairs:,} pairs...", log_file)
    feat_start = time.time()
    
    name_jaro = np.empty(n_pairs, dtype=np.float32)
    name_token_sort = np.empty(n_pairs, dtype=np.float32)
    name_partial = np.empty(n_pairs, dtype=np.float32)
    address_jaro = np.empty(n_pairs, dtype=np.float32)
    address_token_set = np.empty(n_pairs, dtype=np.float32)
    exact_name = np.empty(n_pairs, dtype=np.float32)
    
    chunk_size = 500000
    for i in range(0, n_pairs, chunk_size):
        end = min(i + chunk_size, n_pairs)
        b_s1 = s1_arr[i:end]
        b_c23 = c23_arr[i:end]
        
        b_n1 = [s1_names.get(s, '') for s in b_s1]
        b_n2 = [s23_names.get(c, '') for c in b_c23]
        b_a1 = [s1_addrs.get(s, '') for s in b_s1]
        b_a2 = [s23_addrs.get(c, '') for c in b_c23]
        
        name_jaro[i:end] = [JaroWinkler.normalized_similarity(n1, n2) for n1, n2 in zip(b_n1, b_n2)]
        name_token_sort[i:end] = [fuzz.token_sort_ratio(n1, n2) for n1, n2 in zip(b_n1, b_n2)]
        name_partial[i:end] = [fuzz.partial_ratio(n1, n2) for n1, n2 in zip(b_n1, b_n2)]
        address_jaro[i:end] = [JaroWinkler.normalized_similarity(a1, a2) for a1, a2 in zip(b_a1, b_a2)]
        address_token_set[i:end] = [fuzz.token_set_ratio(a1, a2) for a1, a2 in zip(b_a1, b_a2)]
        exact_name[i:end] = [1.0 if n1 and n1 == n2 else 0.0 for n1, n2 in zip(b_n1, b_n2)]
        
        if (i + chunk_size) % 5000000 < chunk_size or end == n_pairs:
            log(f"  * Feature computation progress: {end:,} / {n_pairs:,} pairs ({end/n_pairs*100:.1f}%)...", log_file)
            
    feat_elapsed = time.time() - feat_start
    log(f"Fuzzy feature computation finished in {feat_elapsed:.2f} seconds ({n_pairs/feat_elapsed:.0f} pairs/sec).", log_file)
    
    X = np.column_stack([tfidf_sim_arr, name_jaro, name_token_sort, name_partial, address_jaro, address_token_set, exact_name])
    y = labels
    
    del name_jaro, name_token_sort, name_partial, address_jaro, address_token_set, exact_name, tfidf_sim_arr
    
    log("Training XGBoost Classifier (300 trees, max depth 6, learning rate 0.08)...", log_file)
    train_start = time.time()
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(scale_pos_weight, 5.0),
        tree_method='hist',
        n_jobs=16,
        random_state=42
    )
    
    model.fit(X, y)
    log(f"XGBoost model training finished in {time.time() - train_start:.2f} seconds.", log_file)
    
    model_path = os.path.join(output_dir, 'xgboost_model.json')
    model.save_model(model_path)
    log(f"Saved trained XGBoost model to {model_path}", log_file)
    
    log("Evaluating probability predictions & finding optimal threshold for F_0.5 score...", log_file)
    probs = model.predict_proba(X)[:, 1]
    
    best_thresh = 0.85
    best_f05 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    
    for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds = (probs >= thresh).astype(int)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f05 = fbeta_score(y, preds, beta=0.5, zero_division=0)
        log(f"  Threshold {thresh:.2f} -> Precision: {prec:.4f} | Recall: {rec:.4f} | F_0.5 Score: {f05:.4f}", log_file)
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_prec = prec
            best_rec = rec
            
    log(f"Optimal Threshold: {best_thresh:.2f} | Best F_0.5 Score: {best_f05:.4f} | Precision: {best_prec:.4f} | Recall: {best_rec:.4f}", log_file)
    
    log("Formatting and generating output TSV files (candidate_pairs.tsv & matching_results.tsv)...", log_file)
    fmt_start = time.time()
    
    # 1. Output candidate_pairs.tsv using streaming dictionary writer
    cands_dict = {}
    for s1_id, match_id in zip(s1_arr, c23_arr):
        if s1_id not in cands_dict:
            cands_dict[s1_id] = [match_id]
        else:
            cands_dict[s1_id].append(match_id)
            
    cands_res_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    with open(cands_res_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in df_s1['entity_id'].values:
            if s1_id in cands_dict:
                m_str = ','.join(cands_dict[s1_id])
                f.write(f"{s1_id}\t{m_str}\n")
            else:
                f.write(f"{s1_id}\t\n")
                
    log(f"Saved candidate pairs output to {cands_res_path}", log_file)
    del cands_dict
    
    # 2. Output matching_results.tsv using streaming dictionary writer
    match_mask = probs >= best_thresh
    matched_s1 = s1_arr[match_mask]
    matched_c23 = c23_arr[match_mask]
    
    matches_dict = {}
    for s1_id, match_id in zip(matched_s1, matched_c23):
        if s1_id not in matches_dict:
            matches_dict[s1_id] = [match_id]
        else:
            matches_dict[s1_id].append(match_id)
            
    result_path = os.path.join(output_dir, 'matching_results.tsv')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in df_s1['entity_id'].values:
            if s1_id in matches_dict:
                m_str = ','.join(matches_dict[s1_id])
                f.write(f"{s1_id}\t{m_str}\n")
            else:
                f.write(f"{s1_id}\t\n")
                
    log(f"Matching results successfully saved to {result_path} in {time.time() - fmt_start:.2f}s.", log_file)
    log(f"Processed {len(df_s1):,} Source 1 entities.", log_file)
    log(f"STEP 3 COMPLETED IN {time.time() - step_start:.2f} SECONDS.\n", log_file)

if __name__ == "__main__":
    main()
