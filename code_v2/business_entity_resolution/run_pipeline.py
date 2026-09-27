import os
import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def run_script(script_name):
    logging.info(f"============================================================")
    logging.info(f"  RUNNING: {script_name}")
    logging.info(f"============================================================")
    
    result = subprocess.run([sys.executable, f"src/{script_name}"])
    if result.returncode != 0:
        logging.error(f"Script {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    scripts = [
        # "01_preprocess.py", # Already done
        # "02_blocking.py", # Already done before server restart
        "03_features.py",
        "04_train_and_val.py",
        "05_inference.py"
    ]
    
    for script in scripts:
        run_script(script)
        
    logging.info("PIPELINE COMPLETED SUCCESSFULLY.")
