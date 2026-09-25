import os
import sys
import time
import datetime
import json
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import xgboost as xgb
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import KFold

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def log(msg, log_files=None):
    formatted_msg = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(formatted_msg, flush=True)
    if log_files:
        if not isinstance(log_files, list):
            log_files = [log_files]
        for lf in log_files:
            if lf:
                try:
                    with open(lf, "a", encoding="utf-8") as f:
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
    runs_dir = os.path.join(base_dir, "runs")
    
    # Create timestamped run folder
    run_timestamp = get_timestamp()
    run_folder = os.path.join(runs_dir, f"run_{run_timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    run_log_file = os.path.join(run_folder, "train.log")
    main_log_file = os.path.join(base_dir, "train.log")
    active_logs = [run_log_file, main_log_file]
    
    log("=" * 70, active_logs)
    log(f"10-FOLD CROSS VALIDATION & XGBOOST MODEL TRAINING INITIALIZED", active_logs)
    log(f"Run Folder Created: {run_folder}", active_logs)
    log("=" * 70, active_logs)
    
    step_start = time.time()
    log(f"Loading preprocessed clean dataset tables from {processed_dir}...", active_logs)
    df_s1 = pd.read_csv(os.path.join(processed_dir, 'clean_train_s1.tsv'), sep='\t', low_memory=False)
    df_s2 = pd.read_csv(os.path.join(processed_dir, 'clean_train_s2.tsv'), sep='\t', low_memory=False)
    df_s3 = pd.read_csv(os.path.join(processed_dir, 'clean_train_s3.tsv'), sep='\t', low_memory=False)
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    log("Building string lookup dictionaries...", active_logs)
    s1_names = dict(zip(df_s1['entity_id'].values, df_s1['clean_name'].fillna('').astype(str).values))
    s1_addrs = dict(zip(df_s1['entity_id'].values, df_s1['clean_address'].fillna('').astype(str).values))
    s23_names = dict(zip(df_s23['entity_id'].values, df_s23['clean_name'].fillna('').astype(str).values))
    s23_addrs = dict(zip(df_s23['entity_id'].values, df_s23['clean_address'].fillna('').astype(str).values))
    
    s1_all_entities = df_s1['entity_id'].values
    del df_s2, df_s3, df_s23
    
    log("Loading candidate pairs from candidate_pairs.parquet...", active_logs)
    cands_path = os.path.join(processed_dir, 'candidate_pairs.parquet')
    cands = pd.read_parquet(cands_path)
    n_pairs = len(cands)
    log(f"Loaded {n_pairs:,} candidate pairs.", active_logs)
    
    gt_path = os.path.join(dataset_dir, 'train', 'train_ground_truth.tsv')
    log(f"Loading ground truth labels from {gt_path}...", active_logs)
    gt_start = time.time()
    gt_set = set()
    with open(gt_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                for c_id in parts[1].split(','):
                    if c_id:
                        gt_set.add((s1_id, c_id))
                        
    log(f"Ground truth set loaded: {len(gt_set):,} positive pairs in {time.time() - gt_start:.2f}s.", active_logs)
    
    s1_arr = cands['source1_entity_id'].values
    c23_arr = cands['candidate_entity_id'].values
    tfidf_sim_arr = cands['tfidf_sim'].values.astype(np.float32)
    del cands
    
    log("Labeling ground truth matches...", active_logs)
    labels = np.array([1 if (s, c) in gt_set else 0 for s, c in zip(s1_arr, c23_arr)], dtype=np.int8)
    pos_count = (labels == 1).sum()
    neg_count = (labels == 0).sum()
    log(f"Dataset statistics: {n_pairs:,} candidate pairs | {pos_count:,} positives | {neg_count:,} negatives.", active_logs)
    
    # -------------------------------------------------------------
    # FEATURE EXTRACTION (6 Similarity Features)
    # -------------------------------------------------------------
    log(f"Extracting 6 RapidFuzz fuzzy features for {n_pairs:,} candidate pairs...", active_logs)
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
        name_token_sort[i:end] = [fuzz.token_sort_ratio(n1, n2) / 100.0 for n1, n2 in zip(b_n1, b_n2)]
        name_partial[i:end] = [fuzz.partial_ratio(n1, n2) / 100.0 for n1, n2 in zip(b_n1, b_n2)]
        address_jaro[i:end] = [JaroWinkler.normalized_similarity(a1, a2) for a1, a2 in zip(b_a1, b_a2)]
        address_token_set[i:end] = [fuzz.token_set_ratio(a1, a2) / 100.0 for a1, a2 in zip(b_a1, b_a2)]
        exact_name[i:end] = [1.0 if (n1 and n1 == n2) else 0.0 for n1, n2 in zip(b_n1, b_n2)]
        
        if (i + chunk_size) % 5000000 < chunk_size or end == n_pairs:
            log(f"  * Feature computation progress: {end:,} / {n_pairs:,} pairs ({end/n_pairs*100:.1f}%)...", active_logs)
            
    log(f"Fuzzy feature computation finished in {time.time() - feat_start:.2f} seconds.", active_logs)
    
    X = np.column_stack([tfidf_sim_arr, name_jaro, name_token_sort, name_partial, address_jaro, address_token_set, exact_name])
    y = labels
    del tfidf_sim_arr, name_jaro, name_token_sort, name_partial, address_jaro, address_token_set, exact_name
    
    # -------------------------------------------------------------
    # 10-FOLD GROUP-BASED CROSS VALIDATION
    # -------------------------------------------------------------
    n_splits = 10
    log("=" * 60, active_logs)
    log(f"STARTING {n_splits}-FOLD GROUP-BASED CROSS VALIDATION (STRICT ENTITY SPLIT)", active_logs)
    log("=" * 60, active_logs)
    
    unique_s1_entities = np.unique(s1_arr)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(n_pairs, dtype=np.float32)
    fold_metrics = []
    
    scale_pos_weight = min(neg_count / max(pos_count, 1), 5.0)
    
    cv_start = time.time()
    for fold, (train_ent_idx, val_ent_idx) in enumerate(kf.split(unique_s1_entities), 1):
        log(f"\n--- FOLD {fold}/{n_splits} ---", active_logs)
        val_entities_set = set(unique_s1_entities[val_ent_idx])
        
        val_mask = np.array([s in val_entities_set for s in s1_arr], dtype=bool)
        train_mask = ~val_mask
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        
        log(f"  Train pairs: {len(X_tr):,} | Val pairs: {len(X_va):,}", active_logs)
        
        fold_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            n_jobs=16,
            random_state=42 + fold
        )
        
        t_fold = time.time()
        fold_model.fit(X_tr, y_tr)
        val_preds_prob = fold_model.predict_proba(X_va)[:, 1]
        oof_predictions[val_mask] = val_preds_prob
        
        # Calculate fold metrics at 0.90 threshold
        val_preds_bin = (val_preds_prob >= 0.90).astype(int)
        p_f = precision_score(y_va, val_preds_bin, zero_division=0)
        r_f = recall_score(y_va, val_preds_bin, zero_division=0)
        f05_f = fbeta_score(y_va, val_preds_bin, beta=0.5, zero_division=0)
        
        log(f"  Fold {fold} finished in {time.time() - t_fold:.2f}s | Prec: {p_f:.4f} | Rec: {r_f:.4f} | F_0.5: {f05_f:.4f}", active_logs)
        fold_metrics.append({
            "fold": fold,
            "train_pairs": int(len(X_tr)),
            "val_pairs": int(len(X_va)),
            "precision": float(p_f),
            "recall": float(r_f),
            "f05_score": float(f05_f)
        })
        
    log(f"\n{n_splits}-Fold Cross Validation completed in {time.time() - cv_start:.2f} seconds.", active_logs)
    
    # -------------------------------------------------------------
    # OVERALL OUT-OF-FOLD THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------
    log("=" * 60, active_logs)
    log("OVERALL OUT-OF-FOLD (OOF) THRESHOLD SWEEP", active_logs)
    log("=" * 60, active_logs)
    
    threshold_results = []
    best_thresh = 0.90
    best_oof_f05 = -1.0
    best_prec = 0.0
    best_rec = 0.0
    
    for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds_bin = (oof_predictions >= thresh).astype(int)
        prec = precision_score(y, preds_bin, zero_division=0)
        rec = recall_score(y, preds_bin, zero_division=0)
        f05 = fbeta_score(y, preds_bin, beta=0.5, zero_division=0)
        
        log(f"  Threshold {thresh:.2f} -> Precision: {prec:.4f} | Recall: {rec:.4f} | OOF F_0.5: {f05:.4f}", active_logs)
        threshold_results.append({
            "threshold": float(thresh),
            "precision": float(prec),
            "recall": float(rec),
            "f05_score": float(f05)
        })
        if f05 > best_oof_f05:
            best_oof_f05 = f05
            best_thresh = thresh
            best_prec = prec
            best_rec = rec
            
    log(f"\nOPTIMAL OOF THRESHOLD: {best_thresh:.2f} | BEST OOF F_0.5 SCORE: {best_oof_f05:.4f} ({best_oof_f05*100:.2f}%)", active_logs)
    
    # -------------------------------------------------------------
    # FINAL MODEL FIT ON FULL DATASET & SAVE RUN ARTIFACTS
    # -------------------------------------------------------------
    log("\nTraining Final Production Model on 100% Full Training Dataset...", active_logs)
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=16,
        random_state=42
    )
    t_fin = time.time()
    final_model.fit(X, y)
    log(f"Final Model trained in {time.time() - t_fin:.2f}s.", active_logs)
    
    # Save model checkpoint to run_folder & output_dir
    run_model_path = os.path.join(run_folder, "xgboost_model.json")
    output_model_path = os.path.join(output_dir, "xgboost_model.json")
    final_model.save_model(run_model_path)
    final_model.save_model(output_model_path)
    log(f"Saved XGBoost model to {run_model_path} and {output_model_path}", active_logs)
    
    # Save validation metrics JSON inside runs/<timestamp>/
    validation_summary = {
        "run_timestamp": run_timestamp,
        "n_splits": n_splits,
        "total_candidate_pairs": int(n_pairs),
        "positives": int(pos_count),
        "negatives": int(neg_count),
        "optimal_threshold": float(best_thresh),
        "best_oof_f05_score": float(best_oof_f05),
        "best_oof_precision": float(best_prec),
        "best_oof_recall": float(best_rec),
        "fold_metrics": fold_metrics,
        "threshold_sweep": threshold_results
    }
    
    metrics_json_path = os.path.join(run_folder, "validation_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)
    log(f"Saved detailed validation metrics JSON to {metrics_json_path}", active_logs)
    
    # Generate output matching_results.tsv for training set
    log("Formatting and writing training matching_results.tsv...", active_logs)
    match_mask = oof_predictions >= best_thresh
    matched_s1 = s1_arr[match_mask]
    matched_c23 = c23_arr[match_mask]
    
    matches_dict = {}
    for s1_id, match_id in zip(matched_s1, matched_c23):
        if s1_id not in matches_dict:
            matches_dict[s1_id] = [match_id]
        else:
            matches_dict[s1_id].append(match_id)
            
    out_match_path = os.path.join(output_dir, "matching_results.tsv")
    with open(out_match_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_all_entities:
            if s1_id in matches_dict:
                f.write(f"{s1_id}\t{','.join(matches_dict[s1_id])}\n")
            else:
                f.write(f"{s1_id}\t\n")
                
    log(f"Saved matching results to {out_match_path}", active_logs)
    log("=" * 70, active_logs)
    log(f"RUN {run_timestamp} COMPLETED SUCCESSFULLY IN {(time.time() - step_start)/60:.2f} MINUTES!", active_logs)
    log("=" * 70, active_logs)

if __name__ == "__main__":
    main()
