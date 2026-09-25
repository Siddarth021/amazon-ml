# Business Entity Resolution Pipeline

This repository contains the production-grade, end-to-end Machine Learning pipeline for the **Amazon ML Challenge: Business Entity Resolution**. 

Given business records across 3 independent data sources (`Source 1` reference, `Source 2`, and `Source 3`) with noisy names and inconsistent addresses, the pipeline determines which records across sources refer to the same real-world business entity.

---

## Performance Summary

| Split / Dataset | Precision | Recall | $F_{0.5}$ Score ($\beta = 0.5$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **80/20 Out-of-Fold (OOF) Validation** | **95.18%** | **78.01%** | **0.9117 (91.17%)** | Zero Leakage |
| **Full Train Evaluation** | **95.30%** | **78.63%** | **0.9143 (91.43%)** | Full Model |
| **Official Test Validator (`validate_submission.py`)** | — | — | — | **PASS (100% Valid)** |

---

## Technical Architecture

```
[Raw TSV Datasets]
       │
       ▼
1. Preprocessing & Normalization (01_preprocess.py)
   ├── Unicode Normalization & Special Character Removal
   ├── Legal Suffix Normalization ('corp' -> 'corporation', 'pvt' -> 'private', etc.)
   └── Address Keyword Expansion ('st' -> 'street', 'rd' -> 'road', 'fl' -> 'floor')
       │
       ▼
2. High-Recall Candidate Blocking (02_blocking.py)
   ├── Dynamic Country Partitioning (US, India, France, Open-Set)
   ├── 16-Worker Parallel Inverted Token Indexing (max_df = 0.005)
   └── Candidate Space Reduction: 99.9985% Reduction Ratio
       │
       ▼
3. Feature Engineering & RapidFuzz Similarity (03_train_and_match.py / 04_inference.py)
   ├── name_jaro, name_token_sort, name_partial
   └── address_jaro, address_token_set, exact_name
       │
       ▼
4. XGBoost Classifier Scoring & Precision Optimization
   ├── Model: XGBoost Gradient Boosted Trees (300 trees, max depth 6, lr 0.08)
   └── Decision Threshold: p >= 0.90 (Precision-heavy optimization for F_0.5)
       │
       ▼
5. Formatted TSV Output Generation & Validation
   ├── output/candidate_pairs.tsv (Blocking set)
   └── output/matching_results.tsv (Final entity matches)
```

---

## Directory Structure

```text
business_entity_resolution/
├── src/
│   ├── 01_preprocess.py          # Parallel text normalization & abbreviation expansion
│   ├── 02_blocking.py            # High-recall parallel Inverted Token Index blocking
│   ├── 03_train_and_match.py     # Feature extraction, XGBoost training & threshold search
│   └── 04_inference.py           # End-to-end inference runner for the test dataset
├── run_pipeline.py               # Orchestrator script to execute full training pipeline
├── README.md                     # Technical pipeline documentation & reproduction guide
└── requirements.txt              # Pinned Python dependencies
```

---

## Setup & Installation

1. **Clone Repository & Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Directory Setup**:
   Ensure dataset files are located in `student_resource/dataset/`:
   - `student_resource/dataset/train/` (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`)
   - `student_resource/dataset/test/` (`test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`)

---

## Step-by-Step Execution Guide

### Option A: Run Full Test Inference Pipeline
To generate official leaderboard & submission outputs (`output/matching_results.tsv` and `output/candidate_pairs.tsv`) on the **test dataset**:

```bash
python src/04_inference.py
```

### Option B: Run Training Pipeline Step-by-Step

1. **Step 1: Text Preprocessing & Normalization**:
   ```bash
   python src/01_preprocess.py
   ```
2. **Step 2: Candidate Blocking**:
   ```bash
   python src/02_blocking.py
   ```
3. **Step 3: Feature Extraction, XGBoost Training & Matching**:
   ```bash
   python src/03_train_and_match.py
   ```
4. **Validate Generated Outputs**:
   ```bash
   python ../../student_resource/utils/validate_submission.py \
       --matching output/matching_results.tsv \
       --candidate output/candidate_pairs.tsv \
       --test-dir ../../student_resource/dataset/test \
       --check-ids
   ```
