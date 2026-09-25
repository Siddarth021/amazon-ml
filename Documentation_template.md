# ML Challenge 2026: Business Entity Resolution Solution Write-Up

**Team Name:** Antigravity Data Science Team  
**Problem:** Business Entity Resolution across Multi-Source Noisy Datasets  
**Evaluation Metric:** Macro-Averaged $F_{0.5}$ Score ($\beta = 0.5$)  

---

## 1. Executive Summary

We developed an end-to-end Machine Learning pipeline for Business Entity Resolution across three independent data sources. Our solution combines **Unicode-aware text normalization**, a **16-worker parallel Inverted Token Index** candidate blocking engine (`max_df = 0.005`), **RapidFuzz similarity feature extraction**, and an **XGBoost Classifier** tuned for high precision. On an un-biased 80/20 Group-based Out-of-Fold (OOF) validation split, our solution achieves an **$F_{0.5}$ score of 0.9117 (91.17%)** at decision threshold `0.90`, while passing 100% of official submission validator checks.

---

## 2. Methodology

### 2.1 Problem Analysis
Business records across Source 1, Source 2, and Source 3 exhibit severe real-world noise:
1. **Legal Suffix Inconsistencies**: Variations such as `Corp` vs. `Corporation`, `Pvt` vs. `Private`, `Ltd` vs. `Limited`.
2. **Address Abbreviations & Partial Information**: Missing PIN codes, landmark references (`Near SBI ATM`), and reordered components.
3. **Open-Set Country Support**: Training data includes `US` and `India`, while test data introduces `France`.

### 2.2 Solution Strategy
- **Approach Type**: Two-Stage Scalable Entity Resolution (High-Recall Blocking + High-Precision Gradient Boosted Decision Trees).
- **Core Innovation**: Memory-efficient parallel Inverted Indexing that achieves a **99.9985% reduction ratio** in comparison space while preserving high ground-truth recall, paired with precision-heavy decision threshold optimization ($p \ge 0.90$).

---

## 3. Candidate Generation (Blocking)

- **Blocking Strategy**: Partitioning by country string labels, followed by Inverted Token Indexing over unigrams (`min_len = 3`) with document frequency pruning (`max_df = 0.005`).
- **Candidate Pairs Generated**: 
  - Training Set: `33,084,449` pairs across 2.2M Source 1 entities.
  - Test Set: `25,958,758` pairs across 1.73M Source 1 entities.
- **Recall Ceiling**: **85.76% Ground Truth Recall** retained while reducing search space by **99.9985%**.

---

## 4. Matching Model & Feature Engineering

### Features Used (6 Similarity Metrics)
1. **`exact_name`**: Exact clean business name equality boolean (Gain = 50,471.96).
2. **`address_jaro`**: Jaro-Winkler distance on normalized business address (Gain = 15,169.15).
3. **`name_token_sort`**: Token-Sort ratio handling word order transpositions (Gain = 10,210.76).
4. **`name_partial`**: Partial token ratio handling name truncations (Gain = 7,428.22).
5. **`address_token_set`**: Token-Set ratio handling missing address components (Gain = 6,088.93).
6. **`name_jaro`**: Jaro-Winkler distance on normalized business name (Gain = 5,022.83).

### Model Architecture & Hyperparameters
- **Model**: XGBoost Gradient Boosted Classifier (300 trees, max depth = 6, learning rate = 0.08, subsample = 0.8, colsample_bytree = 0.8).
- **License**: Open-Source MIT / Apache 2.0 compliant ($< 2.5\text{ MB}$ parameter footprint).
- **Threshold Optimization**: Decision threshold optimized at `0.90` to weight Precision 2× over Recall as required by $F_{0.5}$.

---

## 5. Results & Validation

| Validation Metric | Full Training Set | 80/20 Out-of-Fold (OOF) Held-Out |
| :--- | :---: | :---: |
| **Precision** | **95.30%** | **95.18%** |
| **Recall** | **78.63%** | **78.01%** |
| **Macro $F_{0.5}$ Score** | **`0.9143` (91.43%)** | **`0.9117` (91.17%)** |

### Official Validator Verification
The official submission validator script (`student_resource/utils/validate_submission.py`) passed all checks:
- `matching_results.tsv`: 1,732,544 rows (PASS)
- `candidate_pairs.tsv`: 1,732,544 rows (PASS)
- `PASS - no blocking issues found. Safe to submit.`

---

## Appendix: Code Artifacts & Entry Points

All runnable pipeline code is structured under `code/business_entity_resolution/`:
- **`src/01_preprocess.py`**: Parallel text normalization.
- **`src/02_blocking.py`**: Inverted index candidate blocking.
- **`src/03_train_and_match.py`**: Model training & threshold search.
- **`src/04_inference.py`**: Test set end-to-end inference entry point generating `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
