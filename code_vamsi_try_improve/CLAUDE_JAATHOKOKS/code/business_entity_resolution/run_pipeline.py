import subprocess
import sys
import time
import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging():
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/pipeline_{run_timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/logs", exist_ok=True)
    os.makedirs(f"{run_dir}/output", exist_ok=True)
    
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

def main():
    run_dir, log_file = setup_logging()
    
    logging.info("=" * 70)
    logging.info("  CODE VAMSI — CLAUDE JAATHOKOKS PIPELINE")
    logging.info(f"  Run Directory: {run_dir}")
    logging.info("=" * 70)

    # Set up environment for the bash script to inherit
    env = os.environ.copy()
    
    # Convert run_dir to an absolute path so child scripts resolve it correctly
    abs_run_dir = Path(run_dir).resolve()
    env["ER_OUTPUT"] = str(abs_run_dir / "output")
    env["ER_LOGS"] = str(abs_run_dir / "logs")

    t_total = time.time()
    
    cmd = ["bash", "scripts/run_all.sh"]
    logging.info("=" * 60)
    logging.info("  RUNNING: scripts/run_all.sh")
    logging.info("=" * 60)
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    
    for line in iter(process.stdout.readline, ''):
        line = line.rstrip('\n')
        if line:
            logging.info(f"[run_all.sh] {line}")
            
    process.wait()
    total_time = time.time() - t_total
    
    if process.returncode != 0:
        logging.error(f"FAILED: scripts/run_all.sh (exit code {process.returncode})")
        sys.exit(1)
        
    logging.info("=" * 70)
    logging.info("  PIPELINE COMPLETE")
    logging.info("=" * 70)
    logging.info(f"  Total time: {total_time/60:.2f} minutes")
    logging.info("")
    logging.info("  Output files:")
    logging.info(f"    {abs_run_dir}/output/matching_results.tsv")
    logging.info(f"    {abs_run_dir}/output/candidate_pairs.tsv")
    logging.info(f"  Master log saved to: {log_file}")

if __name__ == "__main__":
    main()
