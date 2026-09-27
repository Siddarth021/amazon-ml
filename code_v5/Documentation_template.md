# Methodology Documentation Template

## Approach Overview
We employed a purely machine-learning and data-engineering approach to solve the Entity Resolution challenge, deliberately avoiding deep learning/LLMs to maximize execution speed and scalability on CPU. Our architecture is a 3-stage pipeline:
1. **Enhanced Preprocessing & Field Parsing**: Clean data and extract structured signals (numbers, pincodes, first word) out of raw text.
2. **Multi-Strategy Blocking (Inverted Indices)**: A massive-recall candidate generator that uses 4 overlapping strategies to achieve >98% recall.
3. **Rich Feature XGBoost with 10-Fold Ensemble**: A gradient boosted tree model using 25+ string and contextual features, trained with Hard Negative Mining, and evaluated through rigorous 10-fold CV to prevent data leakage.

## Candidate Generation / Blocking Strategy
Our deep ground truth analysis revealed that the naive token-based blocking missed ~1.07 million positive pairs (85.96% recall ceiling). To fix this without exploding candidate counts, we deployed a union of orthogonal strategies:
1. **Name + Address Token Index**: Main index. When a name query returns 0 candidates (often due to null S2/S3 addresses), we dynamically fall back to address-only queries.
2. **Address-Only + Numbers Index**: Critical for cross-script entities (e.g., Hindi vs English names). Indic script names share ZERO character overlap with English names, but their addresses are written in English and share street numbers and localities.
3. **Character 3-Gram Index (Name)**: To catch heavy typos and domain-style names (`cardiologyspecialists.com` vs `Cardiology Specialists`).
4. **First-Word Blocking**: Retrieves entities sharing the exact first meaningful word in the same country.

This pushed blocking recall from 85.96% to over 98%, removing the primary bottleneck.

## Model Architecture and Feature Engineering
**Model**: We used XGBoost (`tree_method='hist'`) for fast gradient boosting, trained as a binary classifier to predict `P(match | candidate_pair)`.

**Hard Negative Mining**: We subsampled easy negatives but retained 100% of "hard" negatives (those that had high blocking overlap but were not ground truth matches). This forces the tree splits to focus on the hardest distinctions.

**Features (25 Total)**:
- **String Similarity**: RapidFuzz metrics (Jaro-Winkler, Token Sort, Token Set, Partial Ratio) on both name and address.
- **Structural Matches**: Exact match binary flags for first word, extracted city, and extracted numbers.
- **Number & Pincode Jaccard**: Extracted all digit sequences from addresses and computed overlap. This was the single highest-signal feature (house numbers rarely collide).
- **Contextual/Ranking Features**: Instead of scoring pairs in a vacuum, we included the candidate's blocking rank and the gap between this candidate's name similarity and the max name similarity among all candidates for the given S1.

## Post-Processing
We enforced a strict **1-to-1 Mapping Constraint**. Since an S2 or S3 entity can only ever match one S1 entity, if the model predicted multiple matches for an S2/S3 ID, we resolved the conflict by picking the S1 with the highest predicted probability. This dramatically reduced false positives.
