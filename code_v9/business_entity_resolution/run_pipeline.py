"""Pipeline orchestrator for Code V9 — DL-augmented Entity Resolution.

Runs all stages in sequence:
1. Preprocessing (CPU)
2. Bi-Encoder Embedding (GPU)
3. FAISS ANN Index + Query (GPU/CPU)
4. Token Blocking (CPU)
5. Candidate Union (CPU)
6. Feature Extraction + Bi-Encoder Cosine (CPU)
7. Training / 10-Fold CV (CPU)
8. Inference + Export (CPU)
"""

import subprocess
import sys
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_step(name, script, args=None):
    """Run a pipeline step as a subprocess."""
    cmd = [sys.executable, f"src/{script}"]
    if args:
        cmd.extend(args)
    
    logging.info("=" * 60)
    logging.info(f"  RUNNING: {script}")
    logging.info("=" * 60)
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        logging.error(f"FAILED: {script} (exit code {result.returncode})")
        sys.exit(1)
    
    logging.info(f"Completed {script} in {elapsed:.2f}s")
    return elapsed

def main():
    t_total = time.time()
    
    logging.info("=" * 70)
    logging.info("  CODE V9 — DL-AUGMENTED ENTITY RESOLUTION PIPELINE")
    logging.info(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("=" * 70)
    
    timings = {}
    
    # Stage 1: Preprocessing
    timings['preprocess'] = run_step("Preprocessing", "01_preprocess.py")
    
    # Stage 2: Bi-Encoder Embedding (GPU)
    timings['embeddings'] = run_step("Bi-Encoder Embedding", "embeddings.py", ["--mode", "both"])
    
    # Stage 3: FAISS ANN Blocking (GPU/CPU)
    timings['ann'] = run_step("FAISS ANN Blocking", "ann_index.py", ["--mode", "both"])
    
    # Stage 4: Token Blocking (CPU)
    timings['token_blocking'] = run_step("Token Blocking", "02_blocking.py", ["--mode", "both", "--stage", "token"])
    
    # Stage 5: Candidate Union
    timings['union'] = run_step("Candidate Union", "02_blocking.py", ["--mode", "both", "--stage", "union"])
    
    # Stage 6: Feature Extraction
    timings['features'] = run_step("Feature Extraction", "03_features.py", ["--mode", "both"])
    
    # Stage 6b: Add bi-encoder cosine feature
    timings['cosine_feature'] = run_step("Bi-Encoder Cosine Feature", "03_features.py", ["--mode", "both", "--add-cosine"])
    
    # Stage 7: Training
    timings['training'] = run_step("10-Fold CV Training", "04_train_and_val.py")
    
    # Stage 8: Inference
    timings['inference'] = run_step("Inference + Export", "05_inference.py")
    
    # Summary
    total_time = time.time() - t_total
    logging.info("\n" + "=" * 70)
    logging.info("  PIPELINE COMPLETE")
    logging.info("=" * 70)
    logging.info(f"  Total time: {total_time/60:.2f} minutes")
    logging.info("")
    for step, elapsed in timings.items():
        logging.info(f"  {step:30s} {elapsed:8.2f}s ({elapsed/60:.1f}m)")
    logging.info("")
    logging.info("  Output files:")
    logging.info("    output/matching_results.tsv")
    logging.info("    output/candidate_pairs.tsv")

if __name__ == "__main__":
    main()
