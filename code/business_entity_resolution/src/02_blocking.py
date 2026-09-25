import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
from collections import defaultdict

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

# Generic entity & address stop-words
STOP_WORDS = set([
    'street', 'road', 'avenue', 'drive', 'lane', 'suite', 'floor', 'apartment',
    'incorporated', 'corporation', 'company', 'limited', 'private', 'group', 'number',
    'boulevard', 'court', 'place', 'square', 'highway', 'parkway', 'inc', 'corp', 'co', 'ltd', 'pvt'
])

def get_candidates_for_country(df1, df23, top_k=15, log_file=None):
    s1_text = (df1['clean_name'].fillna('') + ' ' + df1['clean_address'].fillna('')).values
    s23_text = (df23['clean_name'].fillna('') + ' ' + df23['clean_address'].fillna('')).values
    
    s1_ids = df1['entity_id'].values
    s23_ids = df23['entity_id'].values
    
    log(f"  - Building Inverted Token Index over {len(s23_text):,} Source 2&3 rows...", log_file)
    idx_start = time.time()
    
    index = defaultdict(list)
    for idx, text in enumerate(s23_text):
        tokens = [w for w in str(text).split() if len(w) >= 3 and w not in STOP_WORDS]
        for tok in set(tokens):
            index[tok].append(idx)
            
    # Prune ultra-frequent tokens (appearing in > 0.5% of rows) to maximize query precision and speed
    max_docs = max(int(len(s23_text) * 0.005), 50)
    pruned_index = {k: np.array(v, dtype=np.int32) for k, v in index.items() if 1 <= len(v) <= max_docs}
    del index # Free initial dict memory immediately
    log(f"  - Inverted Index built: {len(pruned_index):,} distinct key-tokens in {time.time() - idx_start:.2f}s.", log_file)
    
    log(f"  - Querying candidate matches for {len(s1_text):,} Source 1 entities...", log_file)
    search_start = time.time()
    
    s1_res = []
    c23_res = []
    sim_res = []
    
    n_s1 = len(s1_text)
    for idx, text in enumerate(s1_text):
        tokens = [w for w in str(text).split() if len(w) >= 3 and w in pruned_index]
        if not tokens:
            continue
            
        matched_s23 = np.concatenate([pruned_index[tok] for tok in tokens])
        unq, cnts = np.unique(matched_s23, return_counts=True)
        
        if len(unq) > top_k:
            top_idx = np.argpartition(cnts, -top_k)[-top_k:]
            top_cands = unq[top_idx]
            top_cnts = cnts[top_idx]
        else:
            top_cands = unq
            top_cnts = cnts
            
        s1_id = s1_ids[idx]
        for c_idx, count in zip(top_cands, top_cnts):
            s1_res.append(s1_id)
            c23_res.append(s23_ids[c_idx])
            sim_res.append(float(count))
            
        if (idx + 1) % 100000 == 0 or (idx + 1) == n_s1:
            log(f"    * Inverted Index search progress: {idx + 1:,} / {n_s1:,} S1 queries ({ (idx + 1)/n_s1*100:.1f}%)...", log_file)
            
    cands_df = pd.DataFrame({
        'source1_entity_id': s1_res,
        'candidate_entity_id': c23_res,
        'tfidf_sim': sim_res
    })
    log(f"  - Inverted Index candidate search finished in {time.time() - search_start:.2f}s! Generated {len(cands_df):,} candidate pairs.", log_file)
    return cands_df

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(src_dir, ".."))
    processed_dir = os.path.join(base_dir, "processed")
    log_file = os.path.join(base_dir, "train.log")
    
    log("=" * 60, log_file)
    log("STEP 2: INVERTED INDEX BLOCKING STARTED", log_file)
    log("=" * 60, log_file)
    
    step_start = time.time()
    path_s1 = os.path.join(processed_dir, 'clean_train_s1.tsv')
    path_s2 = os.path.join(processed_dir, 'clean_train_s2.tsv')
    path_s3 = os.path.join(processed_dir, 'clean_train_s3.tsv')
    
    log("Loading preprocessed clean data tables...", log_file)
    df_s1 = pd.read_csv(path_s1, sep='\t', low_memory=False)
    df_s2 = pd.read_csv(path_s2, sep='\t', low_memory=False)
    df_s3 = pd.read_csv(path_s3, sep='\t', low_memory=False)
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    log(f"Data loaded: Source1={len(df_s1):,} rows | Source2&3={len(df_s23):,} rows.", log_file)
    
    countries = df_s1['country'].fillna('UNKNOWN').unique()
    log(f"Blocking across {len(countries)} countries: {list(countries)}", log_file)
    
    all_candidates = []
    for country in countries:
        log(f"Processing country block: '{country}'...", log_file)
        c_s1 = df_s1[df_s1['country'].fillna('UNKNOWN') == country]
        c_s23 = df_s23[df_s23['country'].fillna('UNKNOWN') == country]
        
        log(f"  - Source 1 rows: {len(c_s1):,}, Source 2&3 rows: {len(c_s23):,}", log_file)
        if len(c_s1) > 0 and len(c_s23) > 0:
            cands = get_candidates_for_country(c_s1, c_s23, top_k=15, log_file=log_file)
            all_candidates.append(cands)
            log(f"  - Generated {len(cands):,} candidate pairs for '{country}'.", log_file)
            
    final_candidates = pd.concat(all_candidates, ignore_index=True)
    log(f"Total candidate pairs generated: {len(final_candidates):,}", log_file)
    
    save_start = time.time()
    output_path = os.path.join(processed_dir, 'candidate_pairs.parquet')
    final_candidates.to_parquet(output_path, index=False)
    log(f"Saved candidate pairs to {output_path} in {time.time() - save_start:.2f}s.", log_file)
    log(f"STEP 2: BLOCKING COMPLETED IN {time.time() - step_start:.2f} SECONDS.\n", log_file)
