# Code V2: 98% Target Architecture

This directory contains the Version 2 implementation of the Entity Resolution pipeline, targeting ~98% Macro F0.5.

## Architecture & Data Flow

1. **`01_preprocess.py`**: Reads raw TSV files, performs normalization (lowercasing, unicode NFD, legal suffix canonization), extracts structured fields (first word, pincodes, numbers), and saves to fast Parquet format.
2. **`02_blocking.py`**: High-recall multi-strategy blocking. Generates candidates using 4 distinct indices: 
    * Name + Address token inverted index
    * Name 3-gram index (for typos and domain names)
    * Address-only index (critical for cross-script matches like Hindi/English)
    * First-word index
3. **`03_features.py`**: Extracts 25+ rich features per candidate pair using RapidFuzz string metrics (Jaro, Token Sort/Set, Partial Ratio) and exact matches on extracted fields (numbers, pincodes). Also computes contextual features (rank, max similarity gap).
4. **`04_train_and_val.py`**: Implements Hard Negative Mining and trains a 10-fold CV XGBoost ensemble. GroupKFold on `source1_entity_id` ensures zero data leakage across folds.
5. **`05_inference.py`**: Generates probabilities on the test set, applies thresholding, and critically, enforces a 1-to-1 strict mapping constraint where a candidate can only match one S1 entity. Outputs `output/matching_results.tsv`.

## How to Run

Install requirements:
```bash
pip install -r requirements.txt
```

Run the full end-to-end pipeline:
```bash
python3 run_pipeline.py
```
This single orchestrator script will run steps 01-05 in sequence, utilizing multiprocessing to process all 12.5M rows efficiently. Output will be generated in the `output/` directory.
