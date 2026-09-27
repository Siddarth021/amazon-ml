"""Pipeline orchestrator for Code V9.1 — 90/10 Validation & 1-to-1 Inference.

Runs the optimized stages in sequence:
1. FAISS ANN Indexing (Top-K=150)
2. Semantic Fused-Score Candidate Union (Cap=80)
3. Feature Extraction
4. Feature Extraction (Cosine)
5. Rapid 90/10 Holdout Validation & Final Training
6. Inference + Export (Strict 1-to-1)
"""

import subprocess
import sys
import time
import logging
import os
from datetime import datetime

def setup_logging():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/pipeline_{run_timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = f"{run_dir}/master_pipeline.log"
    
    # Configure logging to write to both stdout and the log file
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return run_dir, log_file

def run_step(name, script, args=None):
    """Run a pipeline step as a subprocess, streaming its output directly to stdout so it gets captured."""
    cmd = [sys.executable, f"src/{script}"]
    if args:
        cmd.extend(args)
    
    logging.info("=" * 60)
    logging.info(f"  RUNNING: {script}")
    if args:
        logging.info(f"  ARGS: {' '.join(args)}")
    logging.info("=" * 60)
    
    t0 = time.time()
    # We use subprocess.run without capture_output so it prints to the terminal 
    # (and the user can see it). But since we want to capture it in our log file too,
    # we can pipe stdout and stderr and log them.
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in iter(process.stdout.readline, ''):
        # We strip the newline because logging adds its own, 
        # and we don't prepend our own timestamp if the script already prints one.
        line = line.rstrip('\n')
        if line:
            logging.info(f"[{script}] {line}")
            
    process.wait()
    elapsed = time.time() - t0
    
    if process.returncode != 0:
        logging.error(f"FAILED: {script} (exit code {process.returncode})")
        sys.exit(1)
    
    logging.info(f"Completed {script} in {elapsed:.2f}s\n")
    return elapsed

def main():
    run_dir, log_file = setup_logging()
    t_total = time.time()
    
    logging.info("=" * 70)
    logging.info("  CODE V9.1 — 90/10 VALIDATION PIPELINE")
    logging.info(f"  Run Directory: {run_dir}")
    logging.info("=" * 70)
    
    timings = {}
    
    # Stage 1: Token Blocking (CPU) - Re-running because we increased top_k to 100
    timings['token_blocking'] = run_step("Token Blocking", "02_blocking.py", ["--mode", "both", "--stage", "token"])
    
    # Stage 2: FAISS ANN Blocking (GPU/CPU) - Re-running because we increased ANN_TOP_K to 150
    timings['ann'] = run_step("FAISS ANN Blocking", "ann_index.py", ["--mode", "both"])
    
    # Stage 3: Candidate Union (Fused Score)
    timings['union'] = run_step("Candidate Union", "02_blocking.py", ["--mode", "both", "--stage", "union"])
    
    # Stage 3: Feature Extraction
    timings['features'] = run_step("Feature Extraction", "03_features.py", ["--mode", "both"])
    
    # Stage 4: Add bi-encoder cosine feature
    timings['cosine_feature'] = run_step("Bi-Encoder Cosine Feature", "03_features.py", ["--mode", "both", "--add-cosine"])
    
    # Stage 5: Training & 90/10 Validation
    timings['training'] = run_step("90/10 Validation & Final Training", "04_train_and_val.py")
    
    # Stage 6: Inference (1-to-1 strict)
    timings['inference'] = run_step("Inference + Export", "05_inference.py")
    
    # Summary
    total_time = time.time() - t_total
    logging.info("=" * 70)
    logging.info("  PIPELINE V9.1 COMPLETE")
    logging.info("=" * 70)
    logging.info(f"  Total time: {total_time/60:.2f} minutes")
    logging.info("")
    for step, elapsed in timings.items():
        logging.info(f"  {step:30s} {elapsed:8.2f}s ({elapsed/60:.1f}m)")
    logging.info("")
    logging.info("  Output files:")
    logging.info("    output/matching_results.tsv")
    logging.info("    output/candidate_pairs.tsv")
    logging.info(f"  Master log saved to: {log_file}")

if __name__ == "__main__":
    main()
