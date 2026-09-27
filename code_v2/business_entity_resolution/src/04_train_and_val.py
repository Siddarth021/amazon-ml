import pandas as pd
import numpy as np
import xgboost as xgb
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
    """
    Computes official competition Macro-average F_0.5 score across all Source 1 entities.
    Includes singletons (entities with 0 true matches).
    """
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
    
    # Create run folder
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
    log("10-FOLD CV & XGBOOST TRAINING (CODE V2 - F0.5 OPTIMIZED)")
    log(f"Run Directory: {run_dir}")
    log("=" * 70)
    
    # Load features
    log("Loading features...")
    df = pd.read_parquet(f"{processed_dir}/train_features.parquet")
    
    # Load Ground Truth
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
    
    # Assign labels
    log("Assigning labels...")
    s1_arr = df['source1_entity_id'].values
    c23_arr = df['candidate_entity_id'].values
    labels = np.array([1 if (s, c) in gt_set else 0 for s, c in zip(s1_arr, c23_arr)], dtype=np.int8)
    
    pos_count = int((labels == 1).sum())
    neg_count = int((labels == 0).sum())
    log(f"Dataset: {len(df):,} pairs | {pos_count:,} positives | {neg_count:,} negatives")
    
    # Feature columns
    feature_cols = [c for c in df.columns if c not in ['source1_entity_id', 'candidate_entity_id']]
    X = df[feature_cols].values.astype(np.float32)
    y = labels
    
    del df
    
    # 10-Fold Group CV (split by S1 entity)
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
        'n_jobs': 16,
        'random_state': 42,
        'eval_metric': 'aucpr'
    }
    
    log("=" * 60)
    log(f"STARTING {n_splits}-FOLD GROUP-BASED CROSS VALIDATION")
    log(f"XGBoost params: n_estimators={xgb_params['n_estimators']}, max_depth={xgb_params['max_depth']}, lr={xgb_params['learning_rate']}")
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
        log(f"  Train pairs: {len(X_tr):,} (pos: {(y_tr==1).sum():,}, neg: {(y_tr==0).sum():,})")
        log(f"  Val pairs: {len(X_va):,} (pos: {(y_va==1).sum():,}, neg: {(y_va==0).sum():,})")
        
        model = xgb.XGBClassifier(**xgb_params)
        t_fold = time.time()
        model.fit(X_tr, y_tr)
        t_fit = time.time() - t_fold
        
        val_preds_prob = model.predict_proba(X_va)[:, 1]
        oof_predictions[val_mask] = val_preds_prob
        
        model.save_model(f"{fold_dir}/model.json")
        
        # Threshold sweep
        val_s1_ids = s1_arr[val_mask]
        val_entities_list = unique_s1[val_ent_idx]
        
        best_fold_thresh = 0.90
        best_fold_f05 = -1.0
        fold_threshold_results = []
        
        for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            v_bin = (val_preds_prob >= thresh).astype(int)
            p = float(precision_score(y_va, v_bin, zero_division=0))
            r = float(recall_score(y_va, v_bin, zero_division=0))
            pair_f05 = float(fbeta_score(y_va, v_bin, beta=0.5, zero_division=0))
            macro_f05 = float(compute_macro_f05(val_s1_ids, y_va, v_bin, val_entities_list))
            fold_threshold_results.append({
                "threshold": float(thresh), "precision": p, "recall": r,
                "pair_f05_score": pair_f05, "macro_entity_f05": macro_f05
            })
            if macro_f05 > best_fold_f05:
                best_fold_f05 = macro_f05
                best_fold_thresh = thresh
        
        # Metrics at default 0.90
        v_bin_90 = (val_preds_prob >= 0.90).astype(int)
        p_90 = float(precision_score(y_va, v_bin_90, zero_division=0))
        r_90 = float(recall_score(y_va, v_bin_90, zero_division=0))
        macro_f05_90 = float(compute_macro_f05(val_s1_ids, y_va, v_bin_90, val_entities_list))
        
        fold_val_summary = {
            "cv_fold": fold, "fold_dir": fold_dir,
            "train_entities": int(len(train_ent_idx)), "val_entities": int(len(val_ent_idx)),
            "train_pairs": int(len(X_tr)), "val_pairs": int(len(X_va)),
            "val_positives": int((y_va == 1).sum()), "val_negatives": int((y_va == 0).sum()),
            "training_time_seconds": round(t_fit, 2),
            "metrics_at_default_0.90": {
                "precision": p_90, "recall": r_90, "macro_entity_f05": macro_f05_90
            },
            "best_threshold": {
                "threshold": float(best_fold_thresh), "macro_entity_f05": float(best_fold_f05)
            },
            "threshold_sweep": fold_threshold_results
        }
        
        with open(f"{fold_dir}/val_results.json", "w") as f:
            json.dump(fold_val_summary, f, indent=2)
        
        log(f"  Fold {fold} in {t_fit:.2f}s | Macro F_0.5 (0.90): {macro_f05_90:.4f} | Optimal ({best_fold_thresh:.2f}): {best_fold_f05:.4f}")
        
        fold_metrics.append({
            "fold": fold, "macro_entity_f05": macro_f05_90,
            "best_threshold": float(best_fold_thresh), "best_macro_f05": float(best_fold_f05)
        })
    
    log(f"\n{n_splits}-Fold CV completed in {time.time() - cv_start:.2f}s")
    
    # OOF threshold sweep
    log("=" * 60)
    log("OVERALL OUT-OF-FOLD THRESHOLD SWEEP")
    log("=" * 60)
    
    best_thresh = 0.90
    best_oof_f05 = -1.0
    threshold_results = []
    
    for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        preds_bin = (oof_predictions >= thresh).astype(int)
        prec = float(precision_score(y, preds_bin, zero_division=0))
        rec = float(recall_score(y, preds_bin, zero_division=0))
        pair_f05 = float(fbeta_score(y, preds_bin, beta=0.5, zero_division=0))
        macro_f05 = float(compute_macro_f05(s1_arr, y, preds_bin, unique_s1))
        
        log(f"  Threshold {thresh:.2f} -> Prec: {prec:.4f} | Rec: {rec:.4f} | Macro F_0.5: {macro_f05:.4f}")
        threshold_results.append({
            "threshold": float(thresh), "precision": prec, "recall": rec,
            "pair_f05_score": pair_f05, "macro_entity_f05": macro_f05
        })
        if macro_f05 > best_oof_f05:
            best_oof_f05 = macro_f05
            best_thresh = thresh
    
    log(f"\nOPTIMAL THRESHOLD: {best_thresh:.2f} | BEST OOF MACRO F_0.5: {best_oof_f05:.4f} ({best_oof_f05*100:.2f}%)")
    
    # Train final model on all data
    log("\nTraining Final Production Model on 100% data...")
    final_model = xgb.XGBClassifier(**xgb_params)
    t_fin = time.time()
    final_model.fit(X, y)
    log(f"Final Model trained in {time.time() - t_fin:.2f}s")
    
    final_model.save_model(f"{run_dir}/xgboost_model.json")
    final_model.save_model("output/xgboost_model.json")
    
    # Save meta
    meta = {
        'features': feature_cols,
        'optimal_threshold': float(best_thresh),
        'best_oof_macro_f05': float(best_oof_f05)
    }
    with open(f"{run_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open("output/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    # Save validation summary
    cv_mean = float(np.mean([m['macro_entity_f05'] for m in fold_metrics]))
    cv_std = float(np.std([m['macro_entity_f05'] for m in fold_metrics]))
    
    log(f"\n10-FOLD CV SUMMARY: Mean Macro F_0.5: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    validation_summary = {
        "run_timestamp": run_timestamp,
        "cv_mean_macro_f05": cv_mean, "cv_std_macro_f05": cv_std,
        "optimal_threshold": float(best_thresh),
        "best_oof_macro_f05": float(best_oof_f05),
        "fold_metrics": fold_metrics,
        "threshold_sweep": threshold_results
    }
    with open(f"{run_dir}/validation_metrics.json", "w") as f:
        json.dump(validation_summary, f, indent=2)
    
    log(f"\nTOTAL TRAINING TIME: {(time.time() - t0)/60:.2f} minutes")

if __name__ == "__main__":
    train_and_val()
