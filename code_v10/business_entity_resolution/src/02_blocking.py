"""Multi-strategy blocking: Token Inverted Index + FAISS ANN + Pincode + Numbers.

Unions all four engines' candidates per S1 entity, deduplicates, and caps at 80."""

import pandas as pd
import numpy as np
import time
import logging
from collections import defaultdict
import multiprocessing
import os
import argparse
import gc

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Token Inverted Index Engine (same as V5 but refined) ---

G_TEXT_INDEX = None
G_PIN_INDEX = None
G_NUM_INDEX = None
G_C23_IDS = None

def init_worker(text_index, pin_index, num_index, c23_ids):
    global G_TEXT_INDEX, G_PIN_INDEX, G_NUM_INDEX, G_C23_IDS
    G_TEXT_INDEX = text_index
    G_PIN_INDEX = pin_index
    G_NUM_INDEX = num_index
    G_C23_IDS = c23_ids

def query_chunk(args):
    s1_ids_chunk, s1_text_chunk, s1_pin_chunk, s1_num_chunk, top_k = args
    text_index = G_TEXT_INDEX
    pin_index = G_PIN_INDEX
    num_index = G_NUM_INDEX
    c23_ids = G_C23_IDS
    
    s1_res = []
    c23_res = []
    sim_res = []
    
    for s1_id, text, pins, nums in zip(s1_ids_chunk, s1_text_chunk, s1_pin_chunk, s1_num_chunk):
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
                
        # Strategy 2: Exact Pincode Match
        pin_tokens = [p for p in str(pins).split() if p in pin_index]
        if pin_tokens:
            matched_pins = np.concatenate([pin_index[p] for p in pin_tokens])
            unq_pins = np.unique(matched_pins)
            cands.append(unq_pins)
            
        # Strategy 3: Exact Numbers Match
        num_tokens = [n for n in str(nums).split() if len(n) >= 5 and n in num_index]
        if num_tokens:
            matched_nums = np.concatenate([num_index[n] for n in num_tokens])
            unq_nums = np.unique(matched_nums)
            cands.append(unq_nums)
            
        if not cands:
            continue
            
        final_cands = np.unique(np.concatenate(cands))
        
        for c_idx in final_cands:
            s1_res.append(s1_id)
            c23_res.append(c23_ids[c_idx])
            sim_res.append(1.0)
            
    return s1_res, c23_res, sim_res


def run_token_blocking(mode='train'):
    """Run the token inverted index blocking engine."""
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== TOKEN BLOCKING ENGINE ({mode}) ===")
    
    s2 = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet")
    s3 = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    
    logging.info(f"Loaded {len(s23):,} S2/S3 entities.")
    
    s23_text = (s23['clean_name'].fillna('') + ' ' + s23['clean_address'].fillna('') + ' ' + s23['numbers'].fillna('')).values
    s23_pins = s23['pincodes'].fillna('').values
    s23_nums = s23['numbers'].fillna('').values
    s23_ids = s23['entity_id'].values
    s23_country = s23['country'].fillna('UNKNOWN').values
    
    s1 = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet")
    logging.info(f"Loaded {len(s1):,} S1 entities.")
    
    s1_text = (s1['clean_name'].fillna('') + ' ' + s1['clean_address'].fillna('') + ' ' + s1['numbers'].fillna('')).values
    s1_pins = s1['pincodes'].fillna('').values
    s1_nums = s1['numbers'].fillna('').values
    s1_ids = s1['entity_id'].values
    s1_country = s1['country'].fillna('UNKNOWN').values
    
    countries = np.unique(s1_country)
    logging.info(f"Countries: {list(countries)}")
    
    all_s1, all_c23, all_sim = [], [], []
    
    for country in countries:
        logging.info(f"Processing country: '{country}'...")
        
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
        
        # Build text index
        t_idx = defaultdict(list)
        for idx, text in enumerate(c_s23_text):
            tokens = set(w for w in str(text).split() if len(w) >= 2)
            for tok in tokens:
                t_idx[tok].append(idx)
        max_docs = max(int(len(c_s23_text) * 0.005), 50)
        pruned_text = {k: np.array(v, dtype=np.int32) for k, v in t_idx.items() if 1 <= len(v) <= max_docs}
        del t_idx
        logging.info(f"  Text Index: {len(pruned_text):,} keys")
        
        # Build pin index
        p_idx = defaultdict(list)
        for idx, pins in enumerate(c_s23_pins):
            for p in set(str(pins).split()):
                p_idx[p].append(idx)
        pruned_pin = {k: np.array(v, dtype=np.int32) for k, v in p_idx.items()}
        del p_idx
        
        # Build num index
        n_idx = defaultdict(list)
        for idx, nums in enumerate(c_s23_nums):
            for n in set(str(nums).split()):
                if len(n) >= 5:
                    n_idx[n].append(idx)
        pruned_num = {k: np.array(v, dtype=np.int32) for k, v in n_idx.items()}
        del n_idx
        
        top_k = 100
        chunk_size = 50000
        chunks = [
            (c_s1_ids[i:i+chunk_size], c_s1_text[i:i+chunk_size], c_s1_pins[i:i+chunk_size], c_s1_nums[i:i+chunk_size], top_k)
            for i in range(0, len(c_s1_ids), chunk_size)
        ]
        
        with multiprocessing.Pool(
            processes=min(16, multiprocessing.cpu_count()),
            initializer=init_worker,
            initargs=(pruned_text, pruned_pin, pruned_num, c_s23_ids)
        ) as pool:
            for chunk_idx, res in enumerate(pool.imap(query_chunk, chunks), 1):
                all_s1.extend(res[0])
                all_c23.extend(res[1])
                all_sim.extend(res[2])
                processed = min(chunk_idx * chunk_size, len(c_s1_ids))
                if chunk_idx % 5 == 0 or processed == len(c_s1_ids):
                    logging.info(f"    Progress: {processed:,} / {len(c_s1_ids):,}")
    
    token_pairs = pd.DataFrame({
        'source1_entity_id': all_s1,
        'candidate_entity_id': all_c23,
        'tfidf_sim': all_sim
    })
    logging.info(f"Token blocking pairs: {len(token_pairs):,}")
    
    out_path = f"{processed_dir}/{mode}_token_pairs.parquet"
    token_pairs.to_parquet(out_path, index=False)
    logging.info(f"Token blocking completed in {time.time() - t0:.2f}s")
    return token_pairs


def union_candidates(mode='train'):
    """Union token blocking + ANN blocking candidates, dedup, cap at 80 per S1 by score."""
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== UNION CANDIDATES ({mode}) ===")
    
    # Load token pairs
    token_path = f"{processed_dir}/{mode}_token_pairs.parquet"
    ann_path = f"{processed_dir}/{mode}_ann_pairs.parquet"
    
    token_df = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'tfidf_sim'])
    ann_df = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'ann_cosine_sim'])
    
    if os.path.exists(token_path):
        token_df = pd.read_parquet(token_path)
        token_df = token_df[['source1_entity_id', 'candidate_entity_id']].drop_duplicates()
        token_df['token_score'] = 0.5
        logging.info(f"  Token pairs: {len(token_df):,}")
    
    if os.path.exists(ann_path):
        ann_df = pd.read_parquet(ann_path)
        ann_df = ann_df.drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'])
        logging.info(f"  ANN pairs: {len(ann_df):,}")
    
    if len(token_df) == 0 and len(ann_df) == 0:
        logging.error("No candidate pairs found!")
        return
    
    # Outer join
    logging.info("  Merging candidates...")
    combined = pd.merge(token_df, ann_df, on=['source1_entity_id', 'candidate_entity_id'], how='outer')
    
    combined['token_score'] = combined['token_score'].fillna(0.0)
    combined['ann_cosine_sim'] = combined['ann_cosine_sim'].fillna(0.0)
    
    # Compute priority score:
    # Both = 3, Token Only = 2, ANN Only = 1
    combined['priority'] = 0
    mask_both = (combined['token_score'] > 0) & (combined['ann_cosine_sim'] > 0)
    mask_token_only = (combined['token_score'] > 0) & (combined['ann_cosine_sim'] == 0)
    mask_ann_only = (combined['token_score'] == 0) & (combined['ann_cosine_sim'] > 0)
    
    combined.loc[mask_both, 'priority'] = 3
    combined.loc[mask_token_only, 'priority'] = 2
    combined.loc[mask_ann_only, 'priority'] = 1
    
    # Within each priority level, sort by ann_cosine_sim to break ties
    combined = combined.sort_values(['source1_entity_id', 'priority', 'ann_cosine_sim'], ascending=[True, False, False])
    
    # Cap at 150 candidates per S1 entity
    MAX_CANDS = 150
    logging.info(f"  Capping at {MAX_CANDS} candidates per S1 based on priority...")
    combined = combined.groupby('source1_entity_id').head(MAX_CANDS).reset_index(drop=True)
    
    # Add a placeholder tfidf_sim column for compatibility with downstream
    combined['tfidf_sim'] = combined['priority'].astype(float)
    combined = combined[['source1_entity_id', 'candidate_entity_id', 'tfidf_sim']]
    
    out_path = f"{processed_dir}/{mode}_candidate_pairs.parquet"
    combined.to_parquet(out_path, index=False)
    logging.info(f"  Final candidates: {len(combined):,}")
    logging.info(f"  Saved to {out_path}")
    logging.info(f"Union completed in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='both')
    parser.add_argument("--stage", type=str, choices=['token', 'union', 'all'], default='all')
    args = parser.parse_args()
    
    modes = ['train', 'test'] if args.mode == 'both' else [args.mode]
    
    for m in modes:
        if args.stage in ['token', 'all']:
            run_token_blocking(m)
        if args.stage in ['union', 'all']:
            union_candidates(m)
