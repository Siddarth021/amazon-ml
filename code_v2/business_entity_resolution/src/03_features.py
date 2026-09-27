import pandas as pd
import numpy as np
import time
import logging
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import multiprocessing
import os
import argparse
import gc

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def safe_jaccard(set1, set2):
    if not set1 and not set2:
        return 0.0
    u = len(set1 | set2)
    return len(set1 & set2) / u if u > 0 else 0.0

def _compute_chunk_features(df_chunk):
    results = []
    for row in df_chunk.itertuples():
        # row: Index, s1_id, c_id, sim, n1, n2, a1, a2, nums1, nums2, pins1, pins2, fw1, fw2, nt1, nt2, at1, at2
        n1, n2 = str(row.n1), str(row.n2)
        a1, a2 = str(row.a1), str(row.a2)
        
        nums1 = set(str(row.nums1).split())
        nums2 = set(str(row.nums2).split())
        pins1 = set(str(row.pins1).split())
        pins2 = set(str(row.pins2).split())
        fw1, fw2 = str(row.fw1), str(row.fw2)
        nt1 = set(str(row.nt1).split())
        nt2 = set(str(row.nt2).split())
        at1 = set(str(row.at1).split())
        at2 = set(str(row.at2).split())
        
        # === NAME FEATURES (8) ===
        name_jaro = JaroWinkler.normalized_similarity(n1, n2)
        name_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_tset = fuzz.token_set_ratio(n1, n2) / 100.0
        name_partial = fuzz.partial_ratio(n1, n2) / 100.0
        name_qratio = fuzz.QRatio(n1, n2) / 100.0
        name_exact = 1.0 if (n1 and n1 == n2) else 0.0
        fw_exact = 1.0 if (fw1 and fw1 == fw2 and len(fw1) >= 2) else 0.0
        name_tok_jaccard = safe_jaccard(nt1, nt2)
        
        # === ADDRESS FEATURES (4) ===
        addr_jaro = JaroWinkler.normalized_similarity(a1, a2) if (a1 and a2) else 0.0
        addr_tset = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
        addr_tok_jaccard = safe_jaccard(at1, at2)
        has_addr = 1.0 if a2 else 0.0
        
        # === STRUCTURAL FEATURES (4) ===
        num_jaccard = safe_jaccard(nums1, nums2)
        num_intersect = 1.0 if (nums1 & nums2) else 0.0
        pin_jaccard = safe_jaccard(pins1, pins2)
        pin_intersect = 1.0 if (pins1 & pins2) else 0.0
        
        # === TOKEN CONTAINMENT (2) ===
        tok_contain = len(nt1 & nt2) / max(1, min(len(nt1), len(nt2))) if nt1 and nt2 else 0.0
        addr_contain = len(at1 & at2) / max(1, min(len(at1), len(at2))) if at1 and at2 else 0.0
        
        # === META (2) ===
        is_s3 = 1.0 if str(row.c_id).startswith('S3') else 0.0
        
        feats = [
            name_jaro, name_tsort, name_tset, name_partial, name_qratio, name_exact, fw_exact, name_tok_jaccard,
            addr_jaro, addr_tset, addr_tok_jaccard, has_addr,
            num_jaccard, num_intersect, pin_jaccard, pin_intersect,
            tok_contain, addr_contain,
            is_s3
        ]
        results.append(feats)
    return results

def process_chunk_file(args):
    chunk_file, s1_df, s23_df = args
    df_pairs = pd.read_parquet(chunk_file)
    
    # Fast Pandas Merge (Join) instead of memory-heavy python dicts
    # Rename columns so they don't overlap
    s1_cols = {'entity_id': 's1_id', 'clean_name': 'n1', 'clean_address': 'a1', 'numbers': 'nums1', 'pincodes': 'pins1', 'first_word': 'fw1', 'name_tokens': 'nt1', 'address_tokens': 'at1'}
    s23_cols = {'entity_id': 'c_id', 'clean_name': 'n2', 'clean_address': 'a2', 'numbers': 'nums2', 'pincodes': 'pins2', 'first_word': 'fw2', 'name_tokens': 'nt2', 'address_tokens': 'at2'}
    
    merged = pd.merge(df_pairs, s1_df.rename(columns=s1_cols), left_on='source1_entity_id', right_on='s1_id', how='inner')
    merged = pd.merge(merged, s23_df.rename(columns=s23_cols), left_on='candidate_entity_id', right_on='c_id', how='inner')
    
    # We only need specific columns for iteration
    iter_cols = ['s1_id', 'c_id', 'tfidf_sim', 'n1', 'n2', 'a1', 'a2', 'nums1', 'nums2', 'pins1', 'pins2', 'fw1', 'fw2', 'nt1', 'nt2', 'at1', 'at2']
    merged = merged[iter_cols]
    
    # Compute features sequentially in this process
    feats = _compute_chunk_features(merged)
    
    # Build feature dataframe
    feat_cols = [
        'name_jaro', 'name_tsort', 'name_tset', 'name_partial', 'name_qratio', 'name_exact', 'fw_exact', 'name_tok_jaccard',
        'addr_jaro', 'addr_tset', 'addr_tok_jaccard', 'has_addr',
        'num_jaccard', 'num_intersect', 'pin_jaccard', 'pin_intersect',
        'tok_contain', 'addr_contain',
        'is_s3'
    ]
    df_feats = pd.DataFrame(feats, columns=feat_cols)
    
    # Concatenate IDs back
    final_chunk = pd.concat([merged[['s1_id', 'c_id', 'tfidf_sim']].rename(columns={'s1_id': 'source1_entity_id', 'c_id': 'candidate_entity_id'}), df_feats], axis=1)
    
    # Overwrite the chunk file with features
    final_chunk.to_parquet(chunk_file, index=False)
    
    # Cleanup memory
    del df_pairs, merged, df_feats, final_chunk
    gc.collect()
    
    return chunk_file

def run_feature_extraction(mode='train'):
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== EXTRACTING FEATURES MEMORY-EFFICIENTLY ({mode}) ===")
    
    # 1. Load entities
    logging.info("Loading entity dataframes...")
    s1_df = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet")
    s2_df = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet")
    s3_df = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet")
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    # 2. Chunk Candidate Pairs to disk to prevent OOM
    logging.info("Splitting candidate pairs into physical disk chunks to save RAM...")
    pairs_file = f"{processed_dir}/{mode}_candidate_pairs.parquet"
    import pyarrow.parquet as pq
    table = pq.read_table(pairs_file)
    n_rows = table.num_rows
    chunk_size = 2_000_000 # Process 2 million rows at a time
    
    os.makedirs(f"{processed_dir}/chunks_{mode}", exist_ok=True)
    chunk_files = []
    
    for i in range(0, n_rows, chunk_size):
        chunk_df = table.slice(i, chunk_size).to_pandas()
        c_file = f"{processed_dir}/chunks_{mode}/chunk_{i}.parquet"
        chunk_df.to_parquet(c_file, index=False)
        chunk_files.append(c_file)
        del chunk_df
        gc.collect()
        
    del table
    gc.collect()
    
    logging.info(f"Split {n_rows:,} pairs into {len(chunk_files)} disk chunks.")
    
    # 3. Process chunks using a strictly limited pool
    # Use only half the CPUs to ensure enough RAM for pandas joins
    n_workers = max(1, multiprocessing.cpu_count() // 2) 
    logging.info(f"Processing chunks with {n_workers} workers...")
    
    args_list = [(c, s1_df, s23_df) for c in chunk_files]
    
    completed = 0
    with multiprocessing.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(process_chunk_file, args_list):
            completed += 1
            logging.info(f"  Processed {completed}/{len(chunk_files)} chunks...")
            
    # Free memory
    del s1_df, s23_df
    gc.collect()
    
    # 4. Re-assemble final features
    logging.info("Re-assembling final features dataframe...")
    dfs = [pd.read_parquet(c) for c in chunk_files]
    df_feat = pd.concat(dfs, ignore_index=True)
    
    for c in chunk_files:
        os.remove(c)
    os.rmdir(f"{processed_dir}/chunks_{mode}")
    
    # 5. Add Contextual Features
    logging.info("Computing contextual features...")
    s1_max = df_feat.groupby('source1_entity_id')['name_tset'].transform('max')
    df_feat['name_sim_gap'] = s1_max - df_feat['name_tset']
    df_feat['cand_rank'] = df_feat.groupby('source1_entity_id')['tfidf_sim'].rank(method='dense', ascending=False)
    df_feat['n_candidates'] = df_feat.groupby('source1_entity_id')['candidate_entity_id'].transform('count')
    
    out_file = f"{processed_dir}/{mode}_features.parquet"
    df_feat.to_parquet(out_file, index=False)
    logging.info(f"Saved {len(df_feat):,} features to {out_file}")
    logging.info(f"Feature extraction completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        run_feature_extraction('train')
        
    if args.mode in ['test', 'both']:
        run_feature_extraction('test')
