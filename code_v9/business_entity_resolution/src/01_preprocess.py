import pandas as pd
import numpy as np
import re
import unicodedata
import os
import csv
from multiprocessing import Pool, cpu_count
from functools import partial
import logging
import json
import time

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_directories(processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

# Precompile regexes
RE_PUNCT_DASH = re.compile(r"['.,&\-\(\)\[\]\{\}:;\"/\\]")
RE_DOMAIN = re.compile(r"\.(com|in|org|net|co|biz|info)\b")
RE_WWW = re.compile(r"\bwww\.")
RE_SPACES = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\d+")
RE_PINCODE = re.compile(r"\b\d{5,6}\b")
RE_LEADING_NOISE = re.compile(r"^[\s\-_.#*]+")

STOPWORDS = {"the", "a", "an", "mr", "shri", "mrs", "ms"}

# Legal suffix canonicalization (from Qbyte's approach)
LEGAL_SUFFIXES = {
    "pvt": "pvt", "private": "pvt",
    "ltd": "ltd", "limited": "ltd",
    "corp": "corp", "corporation": "corp",
    "inc": "inc", "incorporated": "inc",
    "llc": "llc", "llp": "llp",
    "co": "co", "company": "co",
}

def normalize_text_unicode_preserving(text):
    """Normalize text while PRESERVING non-Latin scripts (Devanagari, Kannada, etc.)
    for the bi-encoder embedding stage. Never destroy Unicode."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # NFKC normalization (collapses ligatures, normalizes width variants)
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    
    # Strip domain noise
    text = RE_WWW.sub("", text)
    text = RE_DOMAIN.sub(r" \1 ", text)
    
    # Strip leading punctuation noise
    text = RE_LEADING_NOISE.sub("", text)
    
    # Collapse whitespace
    text = RE_SPACES.sub(" ", text).strip()
    return text

def normalize_text_ascii(text):
    """Normalize text with full ASCII folding for classical string similarity features."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = RE_DOMAIN.sub(r" \1 ", text)
    text = RE_PUNCT_DASH.sub(" ", text)
    
    # Canonicalize legal suffixes
    tokens = text.split()
    tokens = [LEGAL_SUFFIXES.get(t, t) for t in tokens]
    text = " ".join(tokens)
    
    text = RE_SPACES.sub(" ", text).strip()
    return text

def extract_features(row):
    entity_id, name, address, country = row
    
    # Unicode-preserving normalization (for bi-encoder)
    original_name = normalize_text_unicode_preserving(name)
    original_addr = normalize_text_unicode_preserving(address)
    
    # ASCII-folded normalization (for classical features)
    clean_name = normalize_text_ascii(name)
    clean_addr = normalize_text_ascii(address)
    
    # Tokens from ASCII-folded version
    name_tokens = clean_name.split()
    addr_tokens = clean_addr.split()
    
    # First word extraction (skip stopwords)
    first_word = ""
    for token in name_tokens:
        if token not in STOPWORDS:
            first_word = token
            break
            
    # Number extraction from address
    numbers = RE_NUMBERS.findall(clean_addr)
    pincodes = RE_PINCODE.findall(clean_addr)
    
    return {
        'entity_id': entity_id,
        'country': country if pd.notna(country) else "",
        'original_name': original_name,       # Unicode-preserving for bi-encoder
        'original_address': original_addr,     # Unicode-preserving for bi-encoder
        'clean_name': clean_name,              # ASCII-folded for classical features
        'clean_address': clean_addr,           # ASCII-folded for classical features
        'first_word': first_word,
        'name_tokens': " ".join(name_tokens),
        'address_tokens': " ".join(addr_tokens),
        'numbers': " ".join(numbers),
        'pincodes': " ".join(pincodes)
    }

def process_chunk(chunk_data):
    return [extract_features(row) for row in chunk_data]

def preprocess_file(input_file, output_file, num_workers):
    logging.info(f"Processing {input_file} ...")
    
    df = pd.read_csv(input_file, sep='\t', quoting=csv.QUOTE_NONE, low_memory=False, 
                     dtype={'entity_id': str, 'business_name': str, 'business_address': str, 'country': str})
    
    df['business_name'] = df['business_name'].fillna("")
    df['business_address'] = df['business_address'].fillna("")
    
    data = list(zip(df['entity_id'], df['business_name'], df['business_address'], df['country']))
    
    chunk_size = max(10000, len(data) // (num_workers * 4))
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    processed_rows = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap(process_chunk, chunks):
            processed_rows.extend(result)
            
    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_parquet(output_file, index=False)
    logging.info(f"Saved {len(processed_df)} rows to {output_file}")

if __name__ == "__main__":
    t0 = time.time()
    
    dataset_dir = "../../../Dataset/student_resource/dataset"
    processed_dir = "processed"
    num_workers = min(16, cpu_count())
    
    setup_directories(processed_dir)
    logging.info(f"Starting Preprocessing V9 using {num_workers} workers.")
    
    files_to_process = [
        (f"{dataset_dir}/train/train_source1.tsv", f"{processed_dir}/train_s1.parquet"),
        (f"{dataset_dir}/train/train_source2.tsv", f"{processed_dir}/train_s2.parquet"),
        (f"{dataset_dir}/train/train_source3.tsv", f"{processed_dir}/train_s3.parquet"),
        (f"{dataset_dir}/test/test_source1.tsv", f"{processed_dir}/test_s1.parquet"),
        (f"{dataset_dir}/test/test_source2.tsv", f"{processed_dir}/test_s2.parquet"),
        (f"{dataset_dir}/test/test_source3.tsv", f"{processed_dir}/test_s3.parquet")
    ]
    
    for in_f, out_f in files_to_process:
        if os.path.exists(in_f):
            preprocess_file(in_f, out_f, num_workers)
        else:
            logging.warning(f"File not found: {in_f}")
            
    logging.info(f"Preprocessing V9 completed in {time.time() - t0:.2f} seconds.")
