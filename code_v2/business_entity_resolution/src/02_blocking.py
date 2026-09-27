import pandas as pd
import numpy as np
import time
import logging
from collections import defaultdict
import multiprocessing
import os
import argparse

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Globals for worker processes
G_PRUNED_INDEX = None
G_C23_IDS = None

def init_worker(pruned_index, c23_ids):
    global G_PRUNED_INDEX, G_C23_IDS
    G_PRUNED_INDEX = pruned_index
    G_C23_IDS = c23_ids

def query_chunk(args):
    """Query the inverted index for a chunk of S1 entities. Returns (s1_ids, c23_ids, scores)."""
    s1_ids_chunk, s1_text_chunk, top_k = args
    pruned_index = G_PRUNED_INDEX
    c23_ids = G_C23_IDS
    
    s1_res = []
    c23_res = []
    sim_res = []
    
    for s1_id, text in zip(s1_ids_chunk, s1_text_chunk):
        tokens = [w for w in str(text).split() if len(w) >= 2 and w in pruned_index]
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
            
        for c_idx, count in zip(top_cands, top_cnts):
            s1_res.append(s1_id)
            c23_res.append(c23_ids[c_idx])
            sim_res.append(float(count))
            
    return s1_res, c23_res, sim_res


def run_blocking(mode='train'):
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== STARTING PRECISION-FOCUSED BLOCKING ({mode}) ===")
    
    # Load S2/S3
    s2 = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet")
    s3 = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    
    logging.info(f"Loaded {len(s23):,} S2/S3 entities for indexing.")
    
    # Build combined text for indexing: name + address + numbers
    s23_text = (
        s23['clean_name'].fillna('') + ' ' + 
        s23['clean_address'].fillna('') + ' ' + 
        s23['numbers'].fillna('')
    ).values
    s23_ids = s23['entity_id'].values
    s23_country = s23['country'].fillna('UNKNOWN').values
    
    # Load S1
    s1 = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet")
    logging.info(f"Loaded {len(s1):,} S1 entities for querying.")
    
    s1_text = (
        s1['clean_name'].fillna('') + ' ' + 
        s1['clean_address'].fillna('') + ' ' +
        s1['numbers'].fillna('')
    ).values
    s1_ids = s1['entity_id'].values
    s1_country = s1['country'].fillna('UNKNOWN').values
    
    # Process per country (like V1) for precision
    countries = np.unique(s1_country)
    logging.info(f"Blocking across {len(countries)} countries: {list(countries)}")
    
    all_s1 = []
    all_c23 = []
    all_sim = []
    
    for country in countries:
        logging.info(f"Processing country block: '{country}'...")
        
        c_s1_mask = s1_country == country
        c_s23_mask = s23_country == country
        
        c_s1_ids = s1_ids[c_s1_mask]
        c_s1_text = s1_text[c_s1_mask]
        c_s23_ids = s23_ids[c_s23_mask]
        c_s23_text = s23_text[c_s23_mask]
        
        logging.info(f"  S1: {len(c_s1_ids):,}, S2/S3: {len(c_s23_ids):,}")
        
        if len(c_s1_ids) == 0 or len(c_s23_ids) == 0:
            continue
        
        # Build inverted index for this country's S2/S3
        index = defaultdict(list)
        for idx, text in enumerate(c_s23_text):
            tokens = set(w for w in str(text).split() if len(w) >= 2)
            for tok in tokens:
                index[tok].append(idx)
        
        # Prune: remove tokens appearing in > 0.5% of docs (too common)
        max_docs = max(int(len(c_s23_text) * 0.005), 50)
        pruned_index = {k: np.array(v, dtype=np.int32) for k, v in index.items() if 1 <= len(v) <= max_docs}
        del index
        
        logging.info(f"  Index: {len(pruned_index):,} token keys (pruned to max_df={max_docs})")
        
        # Query in parallel
        top_k = 30  # Slightly higher than V1's 15 to capture more candidates
        chunk_size = 50000
        chunks = [
            (c_s1_ids[i:i + chunk_size], c_s1_text[i:i + chunk_size], top_k)
            for i in range(0, len(c_s1_ids), chunk_size)
        ]
        
        with multiprocessing.Pool(
            processes=min(16, multiprocessing.cpu_count()),
            initializer=init_worker,
            initargs=(pruned_index, c_s23_ids)
        ) as pool:
            for chunk_idx, res in enumerate(pool.imap(query_chunk, chunks), 1):
                all_s1.extend(res[0])
                all_c23.extend(res[1])
                all_sim.extend(res[2])
                processed = min(chunk_idx * chunk_size, len(c_s1_ids))
                if chunk_idx % 5 == 0 or processed == len(c_s1_ids):
                    logging.info(f"    Query progress: {processed:,} / {len(c_s1_ids):,} ({processed/len(c_s1_ids)*100:.1f}%)")
    
    # Build output DataFrame
    pairs_df = pd.DataFrame({
        'source1_entity_id': all_s1,
        'candidate_entity_id': all_c23,
        'tfidf_sim': all_sim
    })
    
    logging.info(f"Total candidate pairs: {len(pairs_df):,}")
    
    out_path = f"{processed_dir}/{mode}_candidate_pairs.parquet"
    pairs_df.to_parquet(out_path, index=False)
    logging.info(f"Saved to {out_path}")
    logging.info(f"Blocking completed in {time.time() - t0:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        run_blocking('train')
    if args.mode in ['test', 'both']:
        run_blocking('test')
