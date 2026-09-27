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

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def setup_directories(processed_dir):
    os.makedirs(processed_dir, exist_ok=True)

# Precompile regexes
RE_PUNCT_DASH = re.compile(r"['.,&\-()\[\]{}:;\"/\\]")
RE_DOMAIN = re.compile(r"\.(com|in|org|net|co|biz|info)\b")
RE_SPACES = re.compile(r"\s+")
RE_NUMBERS = re.compile(r"\d+")
RE_PINCODE = re.compile(r"\b\d{5,6}\b") # 5 (US/FR) or 6 (IN) digit PIN/ZIP

STOPWORDS = {"the", "a", "an", "mr", "shri", "mrs", "ms"}

LEGAL_SUFFIXES = {
    r"\bpvt\b": "private",
    r"\bltd\b": "limited",
    r"\bpltd\b": "private limited",
    r"\bllp\b": "limited liability partnership",
    r"\binc\b": "incorporated",
    r"\bcorp\b": "corporation",
    r"\bco\b": "company"
}
LEGAL_RE = {re.compile(k): v for k, v in LEGAL_SUFFIXES.items()}

def normalize_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase and normalize unicode
    text = text.lower()
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    
    # Handle domain names: replace .com with space com
    text = RE_DOMAIN.sub(r" \1 ", text)
    
    # Replace punctuation with spaces
    text = RE_PUNCT_DASH.sub(" ", text)
    
    # Normalize legal suffixes
    for rx, repl in LEGAL_RE.items():
        text = rx.sub(repl, text)
        
    # Standardize whitespace
    text = RE_SPACES.sub(" ", text).strip()
    return text

def extract_features(row):
    entity_id, name, address, country = row
    
    # Normalize strings
    clean_name = normalize_text(name)
    clean_addr = normalize_text(address)
    
    # Tokens
    name_tokens = clean_name.split()
    addr_tokens = clean_addr.split()
    
    # First word extraction (skip stopwords)
    first_word = ""
    for token in name_tokens:
        if token not in STOPWORDS:
            first_word = token
            break
            
    # Number extraction from address (very high signal)
    numbers = RE_NUMBERS.findall(clean_addr)
    # Get all 5 or 6 digit sequences as potential pincodes/zipcodes
    pincodes = RE_PINCODE.findall(clean_addr)
    
    return {
        'entity_id': entity_id,
        'country': country if pd.notna(country) else "",
        'clean_name': clean_name,
        'clean_address': clean_addr,
        'first_word': first_word,
        'name_tokens': " ".join(name_tokens), # Save as string for parquet
        'address_tokens': " ".join(addr_tokens),
        'numbers': " ".join(numbers),
        'pincodes': " ".join(pincodes)
    }

def process_chunk(chunk_data):
    return [extract_features(row) for row in chunk_data]

def preprocess_file(input_file, output_file, num_workers):
    logging.info(f"Processing {input_file} ...")
    
    # Load dataset
    df = pd.read_csv(input_file, sep='\t', quoting=csv.QUOTE_NONE, low_memory=False, 
                     dtype={'entity_id': str, 'business_name': str, 'business_address': str, 'country': str})
    
    # Fill NAs
    df['business_name'] = df['business_name'].fillna("")
    df['business_address'] = df['business_address'].fillna("")
    
    # Convert to list of tuples for fast multiprocessing
    data = list(zip(df['entity_id'], df['business_name'], df['business_address'], df['country']))
    
    chunk_size = max(10000, len(data) // (num_workers * 4))
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    processed_rows = []
    with Pool(processes=num_workers) as pool:
        for result in pool.imap(process_chunk, chunks):
            processed_rows.extend(result)
            
    # Create new dataframe
    processed_df = pd.DataFrame(processed_rows)
    
    # Save as parquet for fast reading in next steps
    processed_df.to_parquet(output_file, index=False)
    logging.info(f"Saved {len(processed_df)} rows to {output_file}")

if __name__ == "__main__":
    t0 = time.time()
    
    # Config
    dataset_dir = "../../../Dataset/student_resource/dataset"
    processed_dir = "processed"
    num_workers = min(16, cpu_count())
    
    setup_directories(processed_dir)
    logging.info(f"Starting Preprocessing using {num_workers} workers.")
    
    # We will process train and test
    files_to_process = [
        (f"{dataset_dir}/train/train_source1.tsv", f"{processed_dir}/train_s1.parquet"),
        (f"{dataset_dir}/train/train_source2.tsv", f"{processed_dir}/train_s2.parquet"),
        (f"{dataset_dir}/train/train_source3.tsv", f"{processed_dir}/train_s3.parquet"),
        
        # We process test early to make inference fast later
        (f"{dataset_dir}/test/test_source1.tsv", f"{processed_dir}/test_s1.parquet"),
        (f"{dataset_dir}/test/test_source2.tsv", f"{processed_dir}/test_s2.parquet"),
        (f"{dataset_dir}/test/test_source3.tsv", f"{processed_dir}/test_s3.parquet")
    ]
    
    for in_f, out_f in files_to_process:
        if os.path.exists(in_f):
            preprocess_file(in_f, out_f, num_workers)
        else:
            logging.warning(f"File not found: {in_f}")
            
    logging.info(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")
