import pandas as pd
import numpy as np
import time
import logging
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
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
        country = str(row.country_s1).upper()
        
        # === NAME FEATURES ===
        name_jaro = JaroWinkler.normalized_similarity(n1, n2)
        name_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
        name_tset = fuzz.token_set_ratio(n1, n2) / 100.0
        name_partial = fuzz.partial_ratio(n1, n2) / 100.0
        name_qratio = fuzz.QRatio(n1, n2) / 100.0
        name_exact = 1.0 if (n1 and n1 == n2) else 0.0
        fw_exact = 1.0 if (fw1 and fw1 == fw2 and len(fw1) >= 2) else 0.0
        name_tok_jaccard = safe_jaccard(nt1, nt2)
        name_lev = Levenshtein.normalized_similarity(n1, n2)
        name_len_diff = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
        
        # === ADDRESS FEATURES ===
        addr_jaro = JaroWinkler.normalized_similarity(a1, a2) if (a1 and a2) else 0.0
        addr_tset = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
        addr_lev = Levenshtein.normalized_similarity(a1, a2) if (a1 and a2) else 0.0
        addr_tok_jaccard = safe_jaccard(at1, at2)
        has_addr = 1.0 if a2 else 0.0
        addr_len_diff = abs(len(a1) - len(a2)) / max(1, max(len(a1), len(a2)))
        
        # === STRUCTURAL FEATURES ===
        num_jaccard = safe_jaccard(nums1, nums2)
        num_intersect = 1.0 if (nums1 & nums2) else 0.0
        pin_jaccard = safe_jaccard(pins1, pins2)
        pin_intersect = 1.0 if (pins1 & pins2) else 0.0
        
        # === TOKEN CONTAINMENT ===
        tok_contain = len(nt1 & nt2) / max(1, min(len(nt1), len(nt2))) if nt1 and nt2 else 0.0
        addr_contain = len(at1 & at2) / max(1, min(len(at1), len(at2))) if at1 and at2 else 0.0
        
        # === DOMAIN FEATURES (Country One-Hot for XGBoost logic branching) ===
        is_us = 1.0 if country == 'US' else 0.0
        is_in = 1.0 if country == 'INDIA' else 0.0
        is_fr = 1.0 if country == 'FRANCE' else 0.0
        is_uk = 1.0 if country == 'UK' else 0.0
        
        # === META ===
        is_s3 = 1.0 if str(row.c_id).startswith('S3') else 0.0
        
        feats = [
            name_jaro, name_tsort, name_tset, name_partial, name_qratio, name_exact, fw_exact, name_tok_jaccard, name_lev, name_len_diff,
            addr_jaro, addr_tset, addr_lev, addr_tok_jaccard, has_addr, addr_len_diff,
            num_jaccard, num_intersect, pin_jaccard, pin_intersect,
            tok_contain, addr_contain,
            is_us, is_in, is_fr, is_uk,
            is_s3
        ]
        results.append(feats)
    return results

def process_chunk_file(args):
    chunk_file, s1_df, s23_df = args
    df_pairs = pd.read_parquet(chunk_file)
    
    s1_cols = {'entity_id': 's1_id', 'clean_name': 'n1', 'clean_address': 'a1', 'numbers': 'nums1', 'pincodes': 'pins1', 'first_word': 'fw1', 'name_tokens': 'nt1', 'address_tokens': 'at1', 'country': 'country_s1'}
    s23_cols = {'entity_id': 'c_id', 'clean_name': 'n2', 'clean_address': 'a2', 'numbers': 'nums2', 'pincodes': 'pins2', 'first_word': 'fw2', 'name_tokens': 'nt2', 'address_tokens': 'at2'}
    
    merged = pd.merge(df_pairs, s1_df.rename(columns=s1_cols), left_on='source1_entity_id', right_on='s1_id', how='inner')
    merged = pd.merge(merged, s23_df.rename(columns=s23_cols), left_on='candidate_entity_id', right_on='c_id', how='inner')
    
    iter_cols = ['s1_id', 'c_id', 'tfidf_sim', 'n1', 'n2', 'a1', 'a2', 'nums1', 'nums2', 'pins1', 'pins2', 'fw1', 'fw2', 'nt1', 'nt2', 'at1', 'at2', 'country_s1']
    merged = merged[iter_cols]
    
    feats = _compute_chunk_features(merged)
    
    feat_cols = [
        'name_jaro', 'name_tsort', 'name_tset', 'name_partial', 'name_qratio', 'name_exact', 'fw_exact', 'name_tok_jaccard', 'name_lev', 'name_len_diff',
        'addr_jaro', 'addr_tset', 'addr_lev', 'addr_tok_jaccard', 'has_addr', 'addr_len_diff',
        'num_jaccard', 'num_intersect', 'pin_jaccard', 'pin_intersect',
        'tok_contain', 'addr_contain',
        'is_us', 'is_in', 'is_fr', 'is_uk',
        'is_s3'
    ]
    df_feats = pd.DataFrame(feats, columns=feat_cols)
    
    final_chunk = pd.concat([merged[['s1_id', 'c_id', 'tfidf_sim']].rename(columns={'s1_id': 'source1_entity_id', 'c_id': 'candidate_entity_id'}), df_feats], axis=1)
    final_chunk.to_parquet(chunk_file, index=False)
    
    del df_pairs, merged, df_feats, final_chunk
    gc.collect()
    
    return chunk_file

def run_feature_extraction(mode='train'):
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== EXTRACTING FEATURES V3 ({mode}) ===")
    
    s1_df = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet")
    s2_df = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet")
    s3_df = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet")
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    pairs_file = f"{processed_dir}/{mode}_candidate_pairs.parquet"
    import pyarrow.parquet as pq
    table = pq.read_table(pairs_file)
    n_rows = table.num_rows
    chunk_size = 2_000_000
    
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
    
    n_workers = max(1, multiprocessing.cpu_count() // 2) 
    args_list = [(c, s1_df, s23_df) for c in chunk_files]
    
    completed = 0
    with multiprocessing.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(process_chunk_file, args_list):
            completed += 1
            logging.info(f"  Processed {completed}/{len(chunk_files)} chunks...")
            
    del s1_df, s23_df
    gc.collect()
    
    dfs = [pd.read_parquet(c) for c in chunk_files]
    df_feat = pd.concat(dfs, ignore_index=True)
    
    for c in chunk_files:
        os.remove(c)
    os.rmdir(f"{processed_dir}/chunks_{mode}")
    
    s1_max = df_feat.groupby('source1_entity_id')['name_tset'].transform('max')
    df_feat['name_sim_gap'] = s1_max - df_feat['name_tset']
    df_feat['cand_rank'] = df_feat.groupby('source1_entity_id')['tfidf_sim'].rank(method='dense', ascending=False)
    df_feat['n_candidates'] = df_feat.groupby('source1_entity_id')['candidate_entity_id'].transform('count')
    
    out_file = f"{processed_dir}/{mode}_features.parquet"
    df_feat.to_parquet(out_file, index=False)
    logging.info(f"Feature extraction V3 completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        run_feature_extraction('train')
    if args.mode in ['test', 'both']:
        run_feature_extraction('test')
