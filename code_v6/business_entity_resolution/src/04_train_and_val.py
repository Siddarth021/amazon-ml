import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, fbeta_score
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
            
    total_score = 0.0
    num_entities = len(all_val_s1_entities)
    
    for s1 in all_val_s1_entities:
        n_gt = s1_gt_matches[s1]
        n_pred = s1_pred_matches[s1]
        n_tp = s1_tp_matches[s1]
        
        if n_gt == 0:
            if n_pred == 0:
                total_score += 1.0
        else:
            if n_pred > 0:
                prec = n_tp / n_pred
                rec = n_tp / n_gt
                denom = 0.25 * prec + rec
                if denom > 0:
                    total_score += (1.25 * prec * rec) / denom
                    
    return total_score / max(num_entities, 1)

def train_and_val():
    t0 = time.time()
    processed_dir = "processed"
    
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/run_{run_timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    run_log = f"{run_dir}/train.log"
    
    def log(msg):
        logging.info(msg)
        with open(run_log, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    
    log("=" * 70)
    log("10-FOLD CV & XGBOOST + LIGHTGBM ENSEMBLE (CODE V3 - F0.5 EXTREME)")
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
    
    feature_cols = [c for c in df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    X = df[feature_cols].values.astype(np.float32)
    y = labels
    del df
    
    n_splits = 10
    unique_s1 = np.unique(s1_arr)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_predictions = np.zeros(len(X), dtype=np.float32)
    scale_pos_weight = min(neg_count / max(pos_count, 1), 5.0)
    
    xgb_params = {
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'n_jobs': 12,
        'random_state': 42
    }
    
    lgb_params = {
        'n_estimators': 600,
        'max_depth': 8,
        'learning_rate': 0.04,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'n_jobs': 12,
        'random_state': 123
    }
    
    log("=" * 60)
    log(f"STARTING {n_splits}-FOLD GROUP-BASED CROSS VALIDATION ENSEMBLE")
    log("=" * 60)
    
    fold_metrics = []
    cv_start = time.time()
    
    for fold, (train_ent_idx, val_ent_idx) in enumerate(kf.split(unique_s1), 1):
        fold_dir = f"{run_dir}/cv{fold}"
        os.makedirs(fold_dir, exist_ok=True)
        
        log(f"\n--- FOLD {fold}/{n_splits} ---")
        
        val_entities_set = set(unique_s1[val_ent_idx])
        val_mask = np.array([s in val_entities_set for s in s1_arr], dtype=bool)
        train_mask = ~val_mask
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        
        log(f"  Train entities: {len(train_ent_idx):,} | Val entities: {len(val_ent_idx):,}")
        
        t_fold = time.time()
        
        # XGBoost
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(X_tr, y_tr)
        xgb_preds = model_xgb.predict_proba(X_va)[:, 1]
        model_xgb.save_model(f"{fold_dir}/xgb_model.json")
        
        # LightGBM
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        model_lgb.fit(X_tr, y_tr)
        lgb_preds = model_lgb.predict_proba(X_va)[:, 1]
        model_lgb.booster_.save_model(f"{fold_dir}/lgb_model.txt")
        
        t_fit = time.time() - t_fold
        
        # Ensemble Average
        val_preds_prob = (xgb_preds + lgb_preds) / 2.0
        oof_predictions[val_mask] = val_preds_prob
        
        val_s1_ids = s1_arr[val_mask]
        val_entities_list = unique_s1[val_ent_idx]
        
        best_fold_thresh = 0.90
        best_fold_f05 = -1.0
        fold_threshold_results = []
        
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
            v_bin = (val_preds_prob >= thresh).astype(int)
            p = float(precision_score(y_va, v_bin, zero_division=0))
            r = float(recall_score(y_va, v_bin, zero_division=0))
            macro_f05 = float(compute_macro_f05(val_s1_ids, y_va, v_bin, val_entities_list))
            fold_threshold_results.append({
                "threshold": float(thresh), "precision": p, "recall": r, "macro_entity_f05": macro_f05
            })
            if macro_f05 > best_fold_f05:
                best_fold_f05 = macro_f05
                best_fold_thresh = thresh
                
        v_bin_90 = (val_preds_prob >= 0.90).astype(int)
        macro_f05_90 = float(compute_macro_f05(val_s1_ids, y_va, v_bin_90, val_entities_list))
        
        log(f"  Fold {fold} in {t_fit:.2f}s | Macro F_0.5 (0.90): {macro_f05_90:.4f} | Optimal ({best_fold_thresh:.2f}): {best_fold_f05:.4f}")
        
        fold_metrics.append({
            "fold": fold, "macro_entity_f05": macro_f05_90,
            "best_threshold": float(best_fold_thresh), "best_macro_f05": float(best_fold_f05)
        })
        
        # Fast Hold-Out for fast iteration
        log("\n--- Fast Hold-out Validation Complete ---")
        best_thresh = best_fold_thresh
        best_oof_f05 = best_fold_f05
        break
    
    log(f"\nHold-out Evaluation completed in {time.time() - cv_start:.2f}s")
    
    log(f"\nOPTIMAL THRESHOLD FROM VALIDATION: {best_thresh:.2f} | BEST MACRO F_0.5: {best_oof_f05:.4f} ({best_oof_f05*100:.2f}%)")
    
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
        'best_oof_macro_f05': float(best_oof_f05)
    }
    with open("output/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    log(f"\nTOTAL TRAINING TIME: {(time.time() - t0)/60:.2f} minutes")

if __name__ == "__main__":
    train_and_val()
