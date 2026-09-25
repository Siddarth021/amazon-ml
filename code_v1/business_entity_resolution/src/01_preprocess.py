import os
import re
import csv
import sys
import time
import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, log_file=None):
    formatted_msg = f"[{get_timestamp()}] {msg}"
    print(formatted_msg, flush=True)
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(formatted_msg + "\n")
        except Exception:
            pass

import unicodedata

# Common entity name abbreviation expansion map (US, India, France)
NAME_MAP = {
    # English & Indian commercial abbreviations
    'ltd': 'limited', 'corp': 'corporation', 'inc': 'incorporated',
    'co': 'company', 'grp': 'group', 'pvt': 'private', 'assocs': 'associates',
    'llc': 'limited liability company', 'llp': 'limited liability partnership',
    'prop': 'proprietor', 'proprietorship': 'proprietorship',
    # French & European legal entities
    'sa': 'societe anonyme', 'sarl': 'societe a responsabilite limitee',
    'sasu': 'societe par actions simplifiee unipersonnelle',
    'sas': 'societe par actions simplifiee',
    'eurl': 'entreprise unipersonnelle a responsabilite limitee',
    'sci': 'societe civile immobiliere',
    'snc': 'societe en nom collectif',
    'gie': 'groupement d interet economique',
    'ste': 'societe', 'assoc': 'association',
    'gmbh': 'gesellschaft mit beschrankter haftung'
}

# Common address abbreviation expansion map (US, India, France)
ADDRESS_MAP = {
    # English & Indian address terms
    'st': 'street', 'rd': 'road', 'ave': 'avenue', 'dr': 'drive',
    'ln': 'lane', 'apt': 'apartment', 'ste': 'suite', 'fl': 'floor',
    'no': 'number', 'blvd': 'boulevard', 'ct': 'court', 'pl': 'place',
    'sq': 'square', 'hwy': 'highway', 'pkwy': 'parkway',
    'opp': 'opposite', 'near': 'near', 'hno': 'house number',
    # French address terms
    'bd': 'boulevard', 'av': 'avenue', 'imp': 'impasse', 'rte': 'route',
    'all': 'allee', 'che': 'chemin', 'bat': 'batiment', 'etg': 'etage',
    'res': 'residence', 'pl': 'place', 'bis': 'bis'
}

def strip_accents(text):
    if not isinstance(text, str) or not text:
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def build_regex_pattern(mapping):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in mapping.keys()) + r')\b', flags=re.IGNORECASE)
    return pattern, mapping

NAME_REGEX, NAME_DICT = build_regex_pattern(NAME_MAP)
ADDRESS_REGEX, ADDRESS_DICT = build_regex_pattern(ADDRESS_MAP)
CLEAN_CHARS_REGEX = re.compile(r'[^\w\s]', flags=re.UNICODE)
WHITESPACE_REGEX = re.compile(r'\s+')

def clean_text_name(text):
    if not isinstance(text, str) or not text:
        return ""
    text = strip_accents(text.lower())
    text = CLEAN_CHARS_REGEX.sub(' ', text)
    text = NAME_REGEX.sub(lambda m: NAME_DICT.get(m.group(0).lower(), m.group(0)), text)
    return WHITESPACE_REGEX.sub(' ', text).strip()

def clean_text_address(text):
    if not isinstance(text, str) or not text:
        return ""
    text = strip_accents(text.lower())
    text = CLEAN_CHARS_REGEX.sub(' ', text)
    text = ADDRESS_REGEX.sub(lambda m: ADDRESS_DICT.get(m.group(0).lower(), m.group(0)), text)
    return WHITESPACE_REGEX.sub(' ', text).strip()

def process_chunk(df):
    df['clean_name'] = df['business_name'].astype(str).apply(clean_text_name)
    df['clean_address'] = df['business_address'].astype(str).apply(clean_text_address)
    return df

def process_file_parallel(input_path, output_path, log_file, num_workers=16):
    log(f"Starting dataset load from: {input_path}", log_file)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    start_t = time.time()
    df = pd.read_csv(input_path, sep='\t', quoting=csv.QUOTE_NONE, low_memory=False)
    log(f"Successfully loaded {len(df):,} rows from {os.path.basename(input_path)} in {time.time() - start_t:.2f} seconds.", log_file)
    
    log(f"Splitting into chunks for parallel text normalization ({num_workers} workers)...", log_file)
    chunks = [df.iloc[i:i + 100000].copy() for i in range(0, len(df), 100000)]
    log(f"Created {len(chunks)} chunks of ~100,000 rows each.", log_file)
    
    norm_start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_chunks = list(executor.map(process_chunk, chunks))
        
    result_df = pd.concat(processed_chunks, ignore_index=True)
    norm_elapsed = time.time() - norm_start
    log(f"Normalized {len(result_df):,} rows in {norm_elapsed:.2f} seconds ({len(result_df)/norm_elapsed:.0f} rows/sec).", log_file)
    
    save_start = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, sep='\t', index=False)
    log(f"Saved preprocessed dataset to {output_path} in {time.time() - save_start:.2f} seconds.\n", log_file)

if __name__ == "__main__":
    src_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(src_dir, "..", "..", ".."))
    base_dir = os.path.abspath(os.path.join(src_dir, ".."))
    dataset_dir = os.path.join(root_dir, "student_resource", "dataset")
    if not os.path.exists(dataset_dir):
        alt_dataset = os.path.abspath(os.path.join(root_dir, "..", "Dataset", "student_resource", "dataset"))
        if os.path.exists(alt_dataset):
            dataset_dir = alt_dataset
    out_dir = os.path.join(base_dir, "processed")
    log_file = os.path.join(base_dir, "train.log")
    
    os.makedirs(out_dir, exist_ok=True)

    log("=" * 60, log_file)
    log("STEP 1: PREPROCESSING STARTED", log_file)
    log("=" * 60, log_file)
    log(f"Dataset root directory: {dataset_dir}", log_file)
    log(f"Output directory: {out_dir}", log_file)
    log(f"Log file: {log_file}", log_file)
    
    step_start = time.time()
    process_file_parallel(
        os.path.join(dataset_dir, "train", "train_source1.tsv"),
        os.path.join(out_dir, "clean_train_s1.tsv"),
        log_file
    )
    process_file_parallel(
        os.path.join(dataset_dir, "train", "train_source2.tsv"),
        os.path.join(out_dir, "clean_train_s2.tsv"),
        log_file
    )
    process_file_parallel(
        os.path.join(dataset_dir, "train", "train_source3.tsv"),
        os.path.join(out_dir, "clean_train_s3.tsv"),
        log_file
    )
    
    log(f"STEP 1: PREPROCESSING COMPLETED IN {time.time() - step_start:.2f} SECONDS.", log_file)
