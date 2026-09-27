"""FAISS ANN index builder and query module.

Builds per-country HNSW indices on S2+S3 embeddings and queries S1 against them."""

import pandas as pd
import numpy as np
import faiss
import time
import logging
import os
import gc

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

EMBEDDING_DIM = 384
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
ANN_TOP_K = 150


def build_and_query_ann(mode='train'):
    """Build FAISS HNSW indices per country on S2+S3, query S1 against them.
    
    Returns a DataFrame of (source1_entity_id, candidate_entity_id, ann_cosine_sim).
    """
    t0 = time.time()
    processed_dir = "processed"
    embed_dir = f"{processed_dir}/embeddings"
    
    logging.info(f"=== BUILDING FAISS ANN INDEX ({mode}) ===")
    
    # Load entity metadata
    s1_df = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet", columns=['entity_id', 'country'])
    s2_df = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet", columns=['entity_id', 'country'])
    s3_df = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet", columns=['entity_id', 'country'])
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    
    # Load embeddings
    s1_embeds = np.load(f"{embed_dir}/{mode}_s1_embeddings.npy").astype(np.float32)
    s2_embeds = np.load(f"{embed_dir}/{mode}_s2_embeddings.npy").astype(np.float32)
    s3_embeds = np.load(f"{embed_dir}/{mode}_s3_embeddings.npy").astype(np.float32)
    s23_embeds = np.vstack([s2_embeds, s3_embeds])
    del s2_embeds, s3_embeds
    gc.collect()
    
    logging.info(f"  S1: {len(s1_df):,} entities, S2+S3: {len(s23_df):,} entities")
    
    s1_ids = s1_df['entity_id'].values
    s1_countries = s1_df['country'].fillna('UNKNOWN').values
    s23_ids = s23_df['entity_id'].values
    s23_countries = s23_df['country'].fillna('UNKNOWN').values
    
    countries = np.unique(s1_countries)
    logging.info(f"  Countries: {list(countries)}")
    
    all_s1_res = []
    all_c23_res = []
    all_sim_res = []
    
    for country in countries:
        logging.info(f"  Processing country: '{country}'...")
        
        c_s1_mask = s1_countries == country
        c_s23_mask = s23_countries == country
        
        c_s1_ids = s1_ids[c_s1_mask]
        c_s1_embeds = s1_embeds[c_s1_mask]
        c_s23_ids = s23_ids[c_s23_mask]
        c_s23_embeds = s23_embeds[c_s23_mask]
        
        logging.info(f"    S1: {len(c_s1_ids):,}, S2+S3: {len(c_s23_ids):,}")
        
        if len(c_s1_ids) == 0 or len(c_s23_ids) == 0:
            continue
        
        # Normalize embeddings (should already be L2-normalized, but ensure)
        faiss.normalize_L2(c_s23_embeds)
        faiss.normalize_L2(c_s1_embeds)
        
        # Build HNSW index
        n_s23 = len(c_s23_ids)
        if n_s23 < 10000:
            # Small shard: use flat index
            logging.info(f"    Using Flat index (small shard)")
            index = faiss.IndexFlatIP(EMBEDDING_DIM)
        else:
            # Large shard: use HNSW
            logging.info(f"    Building HNSW(M={HNSW_M}) index...")
            index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
            index.hnsw.efSearch = HNSW_EF_SEARCH
        
        index.add(c_s23_embeds)
        logging.info(f"    Index built with {index.ntotal:,} vectors")
        
        # Query in chunks
        top_k = min(ANN_TOP_K, n_s23)
        chunk_size = 50000
        
        for i in range(0, len(c_s1_ids), chunk_size):
            end = min(i + chunk_size, len(c_s1_ids))
            q_embeds = c_s1_embeds[i:end]
            q_ids = c_s1_ids[i:end]
            
            similarities, indices = index.search(q_embeds, top_k)
            
            for j in range(len(q_ids)):
                for k in range(top_k):
                    idx = indices[j, k]
                    if idx >= 0:  # -1 means no result
                        all_s1_res.append(q_ids[j])
                        all_c23_res.append(c_s23_ids[idx])
                        all_sim_res.append(float(similarities[j, k]))
            
            if (i // chunk_size + 1) % 5 == 0 or end == len(c_s1_ids):
                logging.info(f"    Query progress: {end:,} / {len(c_s1_ids):,}")
        
        del index, c_s23_embeds, c_s1_embeds
        gc.collect()
    
    pairs_df = pd.DataFrame({
        'source1_entity_id': all_s1_res,
        'candidate_entity_id': all_c23_res,
        'ann_cosine_sim': all_sim_res
    })
    
    logging.info(f"  ANN candidate pairs: {len(pairs_df):,}")
    
    out_path = f"{processed_dir}/{mode}_ann_pairs.parquet"
    pairs_df.to_parquet(out_path, index=False)
    logging.info(f"  Saved to {out_path}")
    logging.info(f"FAISS ANN completed in {time.time() - t0:.2f}s")
    
    return pairs_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        build_and_query_ann('train')
    if args.mode in ['test', 'both']:
        build_and_query_ann('test')
