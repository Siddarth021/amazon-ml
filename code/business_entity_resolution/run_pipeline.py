import os
import sys
import time
import datetime
import subprocess

def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg, log_file):
    formatted_msg = f"[{get_timestamp()}] {msg}"
    print(formatted_msg, flush=True)
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")
    except Exception:
        pass

def run_step(step_name, script_path, log_file):
    log("=" * 70, log_file)
    log(f"LAUNCHING {step_name}: {os.path.basename(script_path)}", log_file)
    log("=" * 70, log_file)
    start_time = time.time()
    
    cmd = [sys.executable, script_path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    for line in iter(process.stdout.readline, ''):
        clean_line = line.rstrip()
        if clean_line:
            print(clean_line, flush=True)
            try:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(clean_line + "\n")
            except Exception:
                pass
        
    process.wait()
    elapsed = time.time() - start_time
    
    if process.returncode != 0:
        log(f"[ERROR] {step_name} failed with return code {process.returncode} after {elapsed:.2f} seconds.", log_file)
        sys.exit(process.returncode)
    else:
        log(f"[SUCCESS] {step_name} finished execution cleanly in {elapsed:.2f} seconds.\n", log_file)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "src")
    processed_dir = os.path.join(base_dir, "processed")
    log_file = os.path.join(base_dir, "train.log")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== RESUMING PIPELINE AT STEP 3 - {get_timestamp()} ===\n\n")
    
    log("BUSINESS ENTITY RESOLUTION PIPELINE RESUMED AT STEP 3", log_file)
    overall_start = time.time()
    
    s1_clean = os.path.join(processed_dir, "clean_train_s1.tsv")
    s2_clean = os.path.join(processed_dir, "clean_train_s2.tsv")
    s3_clean = os.path.join(processed_dir, "clean_train_s3.tsv")
    cands_file = os.path.join(processed_dir, "candidate_pairs.parquet")
    
    if os.path.exists(s1_clean) and os.path.exists(s2_clean) and os.path.exists(s3_clean):
        log("Preprocessed clean dataset files found in processed/. Skipping Step 1.", log_file)
    else:
        run_step("STEP 1: PREPROCESSING", os.path.join(src_dir, "01_preprocess.py"), log_file)
        
    if os.path.exists(cands_file):
        log("Candidate pairs file found in processed/candidate_pairs.parquet (33,084,449 candidate pairs). Skipping Step 2.", log_file)
    else:
        run_step("STEP 2: BLOCKING", os.path.join(src_dir, "02_blocking.py"), log_file)
        
    run_step("STEP 3: TRAINING & MATCHING", os.path.join(src_dir, "03_train_and_match.py"), log_file)
    
    overall_elapsed = time.time() - overall_start
    log("=" * 70, log_file)
    log(f"ALL PIPELINE STAGES COMPLETED SUCCESSFULLY IN {overall_elapsed / 60:.2f} MINUTES!", log_file)
    log("=" * 70, log_file)
