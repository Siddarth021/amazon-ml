import pandas as pd
import numpy as np
import time
import logging
import multiprocessing
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul_topn

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_blocking(mode='train'):
    t0 = time.time()
    processed_dir = "processed"
    
    logging.info(f"=== STARTING TF-IDF COSINE BLOCKING V7 ({mode}) ===")
    
    # Load S2/S3
    s2 = pd.read_parquet(f"{processed_dir}/{mode}_s2.parquet")
    s3 = pd.read_parquet(f"{processed_dir}/{mode}_s3.parquet")
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    
    logging.info(f"Loaded {len(s23):,} S2/S3 entities for indexing.")
    
    s23_text = (s23['clean_name'].fillna('') + ' ' + s23['clean_address'].fillna('')).values
    s23_ids = s23['entity_id'].values
    s23_country = s23['country'].fillna('UNKNOWN').values
    
    # Load S1
    s1 = pd.read_parquet(f"{processed_dir}/{mode}_s1.parquet")
    logging.info(f"Loaded {len(s1):,} S1 entities for querying.")
    
    s1_text = (s1['clean_name'].fillna('') + ' ' + s1['clean_address'].fillna('')).values
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
        
        c_s23_ids = s23_ids[c_s23_mask]
        c_s23_text = s23_text[c_s23_mask]
        
        logging.info(f"  S1: {len(c_s1_ids):,}, S2/S3: {len(c_s23_ids):,}")
        
        if len(c_s1_ids) == 0 or len(c_s23_ids) == 0:
            continue
            
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, max_df=0.01)
        logging.info("  Fitting TF-IDF Vectorizer...")
        s23_tfidf = vectorizer.fit_transform(c_s23_text)
        logging.info("  Transforming S1 text...")
        s1_tfidf = vectorizer.transform(c_s1_text)
        
        top_k = 50
        
        # We must multiply in chunks to avoid OOM
        chunk_size = 50000
        n_chunks = (s1_tfidf.shape[0] + chunk_size - 1) // chunk_size
        
        s23_tfidf_t = s23_tfidf.T.tocsr() # sparse_dot_topn requires both to be CSR
        s1_tfidf = s1_tfidf.tocsr()
        
        logging.info(f"  Starting Sparse Dot Top-N Multiplication (Chunk size: {chunk_size})...")
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, s1_tfidf.shape[0])
            
            s1_chunk = s1_tfidf[start:end]
            
            # Fast cosine similarity returning only top K per row
            res = sp_matmul_topn(s1_chunk, s23_tfidf_t, top_n=top_k)
            
            # Extract non-zero elements
            non_zeros = res.nonzero()
            row_indices = non_zeros[0]
            col_indices = non_zeros[1]
            data = res.data
            
            for j in range(len(row_indices)):
                s1_idx = start + row_indices[j]
                c23_idx = col_indices[j]
                
                all_s1.append(c_s1_ids[s1_idx])
                all_c23.append(c_s23_ids[c23_idx])
                all_sim.append(float(data[j]))
                
            if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
                logging.info(f"    Processed {end:,} / {s1_tfidf.shape[0]:,} rows")
                
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
