"""Bi-encoder embedding module: encode all entities using a multilingual sentence transformer,
cache to disk as memory-mapped numpy arrays, and provide cosine similarity lookup."""

import pandas as pd
import numpy as np
import time
import logging
import os
import gc
import torch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 512

def encode_entities(mode='train'):
    """Encode all S1/S2/S3 entities for a given mode using the bi-encoder.
    
    Saves embeddings as .npy files, keyed by entity_id order in the parquet files.
    Deduplicates identical text strings before encoding for efficiency.
    """
    from sentence_transformers import SentenceTransformer
    
    t0 = time.time()
    processed_dir = "processed"
    embed_dir = f"{processed_dir}/embeddings"
    os.makedirs(embed_dir, exist_ok=True)
    
    logging.info(f"=== ENCODING ENTITIES WITH BI-ENCODER ({mode}) ===")
    logging.info(f"Model: {MODEL_NAME}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    
    # Load all source files
    sources = ['s1', 's2', 's3']
    for source in sources:
        parquet_file = f"{processed_dir}/{mode}_{source}.parquet"
        embed_file = f"{embed_dir}/{mode}_{source}_embeddings.npy"
        
        if os.path.exists(embed_file):
            logging.info(f"  Embeddings already cached: {embed_file}, skipping.")
            continue
            
        logging.info(f"  Loading {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        # Build text for encoding: original_name + " [SEP] " + original_address
        # Use Unicode-preserving versions for the bi-encoder!
        texts = (df['original_name'].fillna('') + ' [SEP] ' + df['original_address'].fillna('')).values
        
        # Deduplicate for speed
        unique_texts, inverse_idx = np.unique(texts, return_inverse=True)
        logging.info(f"  Total texts: {len(texts):,} | Unique: {len(unique_texts):,} ({len(unique_texts)/len(texts)*100:.1f}%)")
        
        # Encode unique texts
        logging.info(f"  Encoding {len(unique_texts):,} unique texts (batch_size={BATCH_SIZE})...")
        unique_embeddings = model.encode(
            unique_texts.tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
            convert_to_numpy=True,
        )
        
        # Expand back to full set via inverse index
        embeddings = unique_embeddings[inverse_idx]
        
        # Save
        np.save(embed_file, embeddings)
        logging.info(f"  Saved embeddings to {embed_file}: shape={embeddings.shape}, dtype={embeddings.dtype}")
        
        del df, texts, unique_texts, inverse_idx, unique_embeddings, embeddings
        gc.collect()
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    logging.info(f"Bi-encoder encoding completed in {time.time() - t0:.2f}s")


def load_embeddings(mode, source):
    """Load cached embeddings for a given mode and source."""
    embed_file = f"processed/embeddings/{mode}_{source}_embeddings.npy"
    return np.load(embed_file)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'both'], default='both')
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        encode_entities('train')
    if args.mode in ['test', 'both']:
        encode_entities('test')
