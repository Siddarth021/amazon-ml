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
G_TEXT_INDEX = None
G_PIN_INDEX = None
G_C23_IDS = None

def init_worker(text_index, pin_index, c23_ids):
    global G_TEXT_INDEX, G_PIN_INDEX, G_C23_IDS
    G_TEXT_INDEX = text_index
    G_PIN_INDEX = pin_index
    G_C23_IDS = c23_ids

def query_chunk(args):
    """Query the inverted index for a chunk of S1 entities. Returns (s1_ids, c23_ids, scores)."""
    s1_ids_chunk, s1_text_chunk, s1_pin_chunk, top_k = args
    text_index = G_TEXT_INDEX
    pin_index = G_PIN_INDEX
    c23_ids = G_C23_IDS
    
    s1_res = []
    c23_res = []
    sim_res = []
    
    for s1_id, text, pins in zip(s1_ids_chunk, s1_text_chunk, s1_pin_chunk):
        cands = []
        
        # Strategy 1: Text Token Overlap (top_k)
        tokens = [w for w in str(text).split() if len(w) >= 2 and w in text_index]
        if tokens:
            matched_text = np.concatenate([text_index[tok] for tok in tokens])
            unq, cnts = np.unique(matched_text, return_counts=True)
            if len(unq) > top_k:
                top_idx = np.argpartition(cnts, -top_k)[-top_k:]
                cands.append(unq[top_idx])
            else:
                cands.append(unq)
                
        # Strategy 2: Exact Pincode Match (all matches)
        pin_tokens = [p for p in str(pins).split() if p in pin_index]
        if pin_tokens:
            matched_pins = np.concatenate([pin_index[p] for p in pin_tokens])
            unq_pins = np.unique(matched_pins)
            cands.append(unq_pins)
            
        if not cands:
            continue
            
        # Union of both strategies
        final_cands = np.unique(np.concatenate(cands))
        
        for c_idx in final_cands:
            s1_res.append(s1_id)
            c23_res.append(c23_ids[c_idx])
            sim_res.append(1.0) # We no longer rely on count for ML, just candidate retrieval
            
    return s1_res, c23_res, sim_res


def run_blocking(mode='train'):
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== STARTING MULTI-STRATEGY BLOCKING V4 ({mode}) ===")
    
    # Load S2/S3
    s2 = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet")
    s3 = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    
    logging.info(f"Loaded {len(s23):,} S2/S3 entities for indexing.")
    
    # Build text and pins
    s23_text = (s23['clean_name'].fillna('') + ' ' + s23['clean_address'].fillna('')).values
    s23_pins = s23['pincodes'].fillna('').values
    s23_ids = s23['entity_id'].values
    s23_country = s23['country'].fillna('UNKNOWN').values
    
    # Load S1
    s1 = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet")
    logging.info(f"Loaded {len(s1):,} S1 entities for querying.")
    
    s1_text = (s1['clean_name'].fillna('') + ' ' + s1['clean_address'].fillna('')).values
    s1_pins = s1['pincodes'].fillna('').values
    s1_ids = s1['entity_id'].values
    s1_country = s1['country'].fillna('UNKNOWN').values
    
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
        c_s1_pins = s1_pins[c_s1_mask]
        
        c_s23_ids = s23_ids[c_s23_mask]
        c_s23_text = s23_text[c_s23_mask]
        c_s23_pins = s23_pins[c_s23_mask]
        
        logging.info(f"  S1: {len(c_s1_ids):,}, S2/S3: {len(c_s23_ids):,}")
        
        if len(c_s1_ids) == 0 or len(c_s23_ids) == 0:
            continue
        
        # 1. Text Index
        t_idx = defaultdict(list)
        for idx, text in enumerate(c_s23_text):
            tokens = set(w for w in str(text).split() if len(w) >= 2)
            for tok in tokens:
                t_idx[tok].append(idx)
                
        max_docs = max(int(len(c_s23_text) * 0.005), 50)
        pruned_text_index = {k: np.array(v, dtype=np.int32) for k, v in t_idx.items() if 1 <= len(v) <= max_docs}
        del t_idx
        logging.info(f"  Text Index: {len(pruned_text_index):,} keys (pruned to max_df={max_docs})")
        
        # 2. Pincode Index
        p_idx = defaultdict(list)
        for idx, pins in enumerate(c_s23_pins):
            for p in set(str(pins).split()):
                p_idx[p].append(idx)
                
        pruned_pin_index = {k: np.array(v, dtype=np.int32) for k, v in p_idx.items()}
        del p_idx
        logging.info(f"  Pin Index: {len(pruned_pin_index):,} keys")
        
        top_k = 40
        chunk_size = 50000
        chunks = [
            (c_s1_ids[i:i + chunk_size], c_s1_text[i:i + chunk_size], c_s1_pins[i:i + chunk_size], top_k)
            for i in range(0, len(c_s1_ids), chunk_size)
        ]
        
        with multiprocessing.Pool(
            processes=min(16, multiprocessing.cpu_count()),
            initializer=init_worker,
            initargs=(pruned_text_index, pruned_pin_index, c_s23_ids)
        ) as pool:
            for chunk_idx, res in enumerate(pool.imap(query_chunk, chunks), 1):
                all_s1.extend(res[0])
                all_c23.extend(res[1])
                all_sim.extend(res[2])
                processed = min(chunk_idx * chunk_size, len(c_s1_ids))
                if chunk_idx % 5 == 0 or processed == len(c_s1_ids):
                    logging.info(f"    Query progress: {processed:,} / {len(c_s1_ids):,} ({processed/len(c_s1_ids)*100:.1f}%)")
    
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
