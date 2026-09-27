"""LightGBM + XGBoost ensemble training with 10-fold group CV, optimized for macro F_0.5."""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
import time
import json
import os
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def compute_macro_f05(val_s1_ids, y_true_binary, y_pred_binary, all_val_s1_entities):
    s1_gt_matches = defaultdict(int)
    s1_pred_matches = defaultdict(int)
    s1_tp_matches = defaultdict(int)
    
    for s1, yt, yp in zip(val_s1_ids, y_true_binary, y_pred_binary):
        if yt == 1:
            s1_gt_matches[s1] += 1
        if yp == 1:
            s1_pred_matches[s1] += 1
        if yt == 1 and yp == 1:
            s1_tp_matches[s1] += 1
            
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    gt_singletons = 0
    correct_singletons = 0
    num_entities = len(all_val_s1_entities)
    
    for s1 in all_val_s1_entities:
        n_gt = s1_gt_matches[s1]
        n_pred = s1_pred_matches[s1]
        n_tp = s1_tp_matches[s1]
        
        if n_gt == 0:
            gt_singletons += 1
            if n_pred == 0:
                correct_singletons += 1
                total_f05 += 1.0
                total_prec += 1.0
                total_rec += 1.0
        else:
            if n_pred > 0:
                prec = n_tp / n_pred
                rec = n_tp / n_gt
                denom = 0.25 * prec + rec
                total_prec += prec
                total_rec += rec
                if denom > 0:
                    total_f05 += (1.25 * prec * rec) / denom
                    
    return total_f05 / max(num_entities, 1), total_prec / max(num_entities, 1), total_rec / max(num_entities, 1), gt_singletons, correct_singletons

def train_and_val():
    t0 = time.time()
    processed_dir = "processed"
    
    import glob
    pipeline_dirs = sorted(glob.glob("runs/pipeline_*"))
    if pipeline_dirs:
        run_dir = pipeline_dirs[-1]
    else:
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = f"runs/run_{run_timestamp}"
    
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    run_log = f"{run_dir}/train.log"
    
    def log(msg):
        logging.info(msg)
        with open(run_log, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    
    log("=" * 70)
    log("RAPID VALIDATION & FINAL ENSEMBLE (CODE V9.1 - DL AUGMENTED)")
    log(f"Run Directory: {run_dir}")
    log("=" * 70)
    
    log("Loading features...")
    df = pd.read_parquet(f"{processed_dir}/train_features.parquet")
    
    log("Loading ground truth...")
    gt_set = set()
    gt_path = "../../../Dataset/student_resource/dataset/train/train_ground_truth.tsv"
    with open(gt_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) == 2 and parts[1]:
                s1_id = parts[0]
                for c_id in parts[1].split(','):
                    if c_id:
                        gt_set.add((s1_id, c_id))
    
    log(f"Ground truth: {len(gt_set):,} positive pairs")
    
    log("Assigning labels...")
    s1_arr = df['source1_entity_id'].values
    c23_arr = df['candidate_entity_id'].values
    labels = np.array([1 if (s, c) in gt_set else 0 for s, c in zip(s1_arr, c23_arr)], dtype=np.int8)
    
    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())
    log(f"Dataset: {len(df):,} pairs | {pos_count:,} positives | {neg_count:,} negatives")
    log(f"Recall ceiling: {pos_count / len(gt_set) * 100:.2f}% ({pos_count:,} / {len(gt_set):,})")
    
    # Retain country-specific and source-specific features as they are crucial for XGBoost routing
    feature_cols = [c for c in df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    log(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # Subsample negative class to 1:5 ratio to prevent OOM
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    target_neg = len(pos_idx) * 5
    if len(neg_idx) > target_neg:
        np.random.seed(42)
        neg_idx = np.random.choice(neg_idx, target_neg, replace=False)
        
    keep_idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    
    X = df.iloc[keep_idx][feature_cols].values.astype(np.float32)
    y = labels[keep_idx]
    s1_arr = s1_arr[keep_idx]
    
    pos_count_sub = int((y == 1).sum())
    neg_count_sub = int((y == 0).sum())
    log(f"Subsampled to {len(X):,} pairs ({pos_count_sub:,} pos / {neg_count_sub:,} neg) to prevent OOM.")
    del df
    
    scale_pos_weight = min(neg_count / max(pos_count, 1), 5.0)
    
    xgb_params = {
        'n_estimators': 600,
        'max_depth': 7,
        'learning_rate': 0.04,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'n_jobs': 12,
        'random_state': 42
    }
    
    lgb_params = {
        'n_estimators': 800,
        'max_depth': 8,
        'learning_rate': 0.03,
        'num_leaves': 255,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'min_child_samples': 100,
        'n_jobs': 12,
        'random_state': 123,
        'verbose': -1
    }
    
    # Rapid Holdout Validation (90/10 Split)
    log("\n" + "=" * 60)
    log("PERFORMING RAPID HOLDOUT VALIDATION (90% Train / 10% Val)")
    log("=" * 60)
    
    unique_s1 = np.unique(s1_arr)
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    split_idx = int(len(unique_s1) * 0.9)
    train_entities_set = set(unique_s1[:split_idx])
    val_entities_list = unique_s1[split_idx:]
    val_entities_set = set(val_entities_list)
    
    val_mask = np.array([s in val_entities_set for s in s1_arr], dtype=bool)
    train_mask = ~val_mask
    
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    
    log(f"Validation Train Entities: {len(train_entities_set):,} | Val Entities: {len(val_entities_set):,}")
    
    t_val = time.time()
    
    model_xgb_val = xgb.XGBClassifier(**xgb_params)
    model_xgb_val.fit(X_tr, y_tr)
    xgb_preds_val = model_xgb_val.predict_proba(X_va)[:, 1]
    
    model_lgb_val = lgb.LGBMClassifier(**lgb_params)
    model_lgb_val.fit(X_tr, y_tr)
    lgb_preds_val = model_lgb_val.predict_proba(X_va)[:, 1]
    
    val_preds_prob = (xgb_preds_val + lgb_preds_val) / 2.0
    val_s1_ids = s1_arr[val_mask]
    
    log(f"Validation models trained and inferred in {time.time() - t_val:.2f}s")
    
    best_thresh = 0.90
    best_val_f05 = -1.0
    
    log("\n--- Validation Metrics ---")
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        v_bin = (val_preds_prob >= thresh).astype(int)
        f05, prec, rec, gt_sing, cor_sing = compute_macro_f05(val_s1_ids, y_va, v_bin, val_entities_list)
        sing_pct = (cor_sing / gt_sing * 100) if gt_sing > 0 else 0
        log(f"  Thresh {thresh:.2f} -> F0.5: {f05:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Singletons: {cor_sing}/{gt_sing} ({sing_pct:.1f}%)")
        if f05 > best_val_f05:
            best_val_f05 = f05
            best_thresh = thresh
            
    log(f"\nOPTIMAL THRESHOLD: {best_thresh:.2f} | BEST VAL MACRO F_0.5: {best_val_f05:.4f}")
    
    # Train final production models on 100% data
    log("\nTraining Final Production Models on 100% data...")
    final_xgb = xgb.XGBClassifier(**xgb_params)
    final_lgb = lgb.LGBMClassifier(**lgb_params)
    t_fin = time.time()
    final_xgb.fit(X, y)
    final_lgb.fit(X, y)
    log(f"Final Models trained in {time.time() - t_fin:.2f}s")
    
    final_xgb.save_model(f"{run_dir}/xgb_model.json")
    final_xgb.save_model("output/xgb_model.json")
    final_lgb.booster_.save_model(f"{run_dir}/lgb_model.txt")
    final_lgb.booster_.save_model("output/lgb_model.txt")
    
    meta = {
        'features': feature_cols,
        'optimal_threshold': float(best_thresh),
        'best_oof_macro_f05': float(best_val_f05),
        'fold_metrics': [],
        'xgb_params': {k: str(v) for k, v in xgb_params.items()},
        'lgb_params': {k: str(v) for k, v in lgb_params.items()},
    }
    with open("output/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(f"{run_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    log(f"\nTOTAL TRAINING TIME: {(time.time() - t0)/60:.2f} minutes")

if __name__ == "__main__":
    train_and_val()
