import pandas as pd
import numpy as np
import time
import logging
from collections import defaultdict
import multiprocessing
import os
import argparse
import math

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Globals for worker processes
G_TEXT_INDEX = None
G_IDF_MAP = None
G_PIN_INDEX = None
G_NUM_INDEX = None
G_C23_IDS = None

def init_worker(text_index, idf_map, pin_index, num_index, c23_ids):
    global G_TEXT_INDEX, G_IDF_MAP, G_PIN_INDEX, G_NUM_INDEX, G_C23_IDS
    G_TEXT_INDEX = text_index
    G_IDF_MAP = idf_map
    G_PIN_INDEX = pin_index
    G_NUM_INDEX = num_index
    G_C23_IDS = c23_ids

def query_chunk(args):
    """Query the inverted index for a chunk of S1 entities. Returns (s1_ids, c23_ids, scores)."""
    s1_ids_chunk, s1_text_chunk, s1_pin_chunk, s1_num_chunk, top_k = args
    text_index = G_TEXT_INDEX
    idf_map = G_IDF_MAP
    pin_index = G_PIN_INDEX
    num_index = G_NUM_INDEX
    c23_ids = G_C23_IDS
    
    s1_res = []
    c23_res = []
    sim_res = []
    
    for s1_id, text, pins, nums in zip(s1_ids_chunk, s1_text_chunk, s1_pin_chunk, s1_num_chunk):
        cands = []
        
        # Strategy 1: Trigram Token Overlap with exact IDF Weighting (via np.bincount)
        text_str = str(text).replace(' ', '_').lower()
        if len(text_str) >= 3:
            # Generate unique trigrams for this query
            tokens = set(text_str[i:i+3] for i in range(len(text_str)-2))
            valid_tokens = [tg for tg in tokens if tg in text_index]
            
            if valid_tokens:
                # Gather all matching document IDs and their corresponding IDF weights
                matched_docs_list = []
                weights_list = []
                for tg in valid_tokens:
                    docs = text_index[tg]
                    matched_docs_list.append(docs)
                    weights_list.append(np.full(len(docs), idf_map[tg], dtype=np.float32))
                
                matched_docs = np.concatenate(matched_docs_list)
                weights = np.concatenate(weights_list)
                
                # np.bincount is a blazing fast C-level grouping function
                # It sums the IDF weights for each unique document ID perfectly!
                scores = np.bincount(matched_docs, weights=weights)
                
                # Get the IDs of documents that had at least one match
                unq = np.nonzero(scores)[0]
                unq_scores = scores[unq]
                
                if len(unq) > top_k:
                    top_idx = np.argpartition(unq_scores, -top_k)[-top_k:]
                    cands.append(unq[top_idx])
                else:
                    cands.append(unq)
                
        # Strategy 2: Exact Pincode Match (all matches)
        pin_tokens = [p for p in str(pins).split() if p in pin_index]
        if pin_tokens:
            matched_pins = np.concatenate([pin_index[p] for p in pin_tokens])
            unq_pins = np.unique(matched_pins)
            cands.append(unq_pins)
            
        # Strategy 3: Exact Numbers Match (all matches)
        num_tokens = [n for n in str(nums).split() if len(n) >= 5 and n in num_index] # Only match on long numbers
        if num_tokens:
            matched_nums = np.concatenate([num_index[n] for n in num_tokens])
            unq_nums = np.unique(matched_nums)
            cands.append(unq_nums)
            
        if not cands:
            continue
            
        # Union of all strategies
        final_cands = np.unique(np.concatenate(cands))
        
        for c_idx in final_cands:
            s1_res.append(s1_id)
            c23_res.append(c23_ids[c_idx])
            sim_res.append(1.0)
            
    return s1_res, c23_res, sim_res


def run_blocking(mode='train'):
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== STARTING WEIGHTED TRIGRAM BLOCKING V8 ({mode}) ===")
    
    # Load S2/S3
    s2 = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet")
    s3 = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    
    logging.info(f"Loaded {len(s23):,} S2/S3 entities for indexing.")
    
    # Build text and pins and numbers
    s23_text = (s23['clean_name'].fillna('') + ' ' + s23['clean_address'].fillna('') + ' ' + s23['numbers'].fillna('')).values
    s23_pins = s23['pincodes'].fillna('').values
    s23_nums = s23['numbers'].fillna('').values
    s23_ids = s23['entity_id'].values
    s23_country = s23['country'].fillna('UNKNOWN').values
    
    # Load S1
    s1 = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet")
    logging.info(f"Loaded {len(s1):,} S1 entities for querying.")
    
    s1_text = (s1['clean_name'].fillna('') + ' ' + s1['clean_address'].fillna('') + ' ' + s1['numbers'].fillna('')).values
    s1_pins = s1['pincodes'].fillna('').values
    s1_nums = s1['numbers'].fillna('').values
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
        c_s1_nums = s1_nums[c_s1_mask]
        
        c_s23_ids = s23_ids[c_s23_mask]
        c_s23_text = s23_text[c_s23_mask]
        c_s23_pins = s23_pins[c_s23_mask]
        c_s23_nums = s23_nums[c_s23_mask]
        
        logging.info(f"  S1: {len(c_s1_ids):,}, S2/S3: {len(c_s23_ids):,}")
        
        if len(c_s1_ids) == 0 or len(c_s23_ids) == 0:
            continue
        
        # 1. Trigram Text Index + exact IDF
        t_idx = defaultdict(list)
        for idx, text in enumerate(c_s23_text):
            text_str = str(text).replace(' ', '_').lower()
            if len(text_str) >= 3:
                trigrams = set(text_str[i:i+3] for i in range(len(text_str)-2))
                for tg in trigrams:
                    t_idx[tg].append(idx)
                    
        # Compute proper IDF weighting (like TfidfVectorizer does)
        # IDF = log((N + 1) / (df + 1)) + 1
        N = len(c_s23_text)
        idf_map = {}
        
        # We prune extreme noise (trigrams in > 1% of docs) because they bloat RAM
        # but they have near-zero IDF anyway, so deleting them doesn't hurt accuracy.
        max_docs = max(int(N * 0.01), 100) 
        
        pruned_text_index = {}
        for k, v in t_idx.items():
            df = len(v)
            if 1 <= df <= max_docs:
                pruned_text_index[k] = np.array(v, dtype=np.int32)
                idf_map[k] = math.log((N + 1) / (df + 1)) + 1.0
                
        del t_idx
        logging.info(f"  Trigram Index: {len(pruned_text_index):,} keys (pruned to max_df={max_docs})")
        
        # 2. Pincode Index
        p_idx = defaultdict(list)
        for idx, pins in enumerate(c_s23_pins):
            for p in set(str(pins).split()):
                p_idx[p].append(idx)
                
        pruned_pin_index = {k: np.array(v, dtype=np.int32) for k, v in p_idx.items()}
        del p_idx
        logging.info(f"  Pin Index: {len(pruned_pin_index):,} keys")
        
        # 3. Numbers Index
        n_idx = defaultdict(list)
        for idx, nums in enumerate(c_s23_nums):
            for n in set(str(nums).split()):
                if len(n) >= 5: # Only map long numbers like phones
                    n_idx[n].append(idx)
                    
        pruned_num_index = {k: np.array(v, dtype=np.int32) for k, v in n_idx.items()}
        del n_idx
        logging.info(f"  Num Index: {len(pruned_num_index):,} keys")
        
        top_k = 50
        chunk_size = 50000
        chunks = [
            (c_s1_ids[i:i + chunk_size], c_s1_text[i:i + chunk_size], c_s1_pins[i:i + chunk_size], c_s1_nums[i:i + chunk_size], top_k)
            for i in range(0, len(c_s1_ids), chunk_size)
        ]
        
        with multiprocessing.Pool(
            processes=min(16, multiprocessing.cpu_count()),
            initializer=init_worker,
            initargs=(pruned_text_index, idf_map, pruned_pin_index, pruned_num_index, c_s23_ids)
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
