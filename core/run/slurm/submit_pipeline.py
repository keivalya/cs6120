#!/usr/bin/env python3
import subprocess
import re
import sys
from pathlib import Path

# Cases to submit: (task_id, cond, axis)
# Case 1 (7, original) is already running as 8533216.
# Case 4 (0, original) is already running as 8531910.
CASES = [
    (7, "blank", None),
    (7, "nonsense", None),
    (0, "wrong_action", None),
    (0, "wrong_task", None),
    (8, "original", None),
    (8, "wrong_task", None),
    (1, "original", None),
    (1, None, "para_action"),
    (1, None, "para_object"),
]

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return res.returncode, res.stdout.strip(), res.stderr.strip()

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    # Start chain from Case 1 which is already running
    prev_job_id = "8533216"
    print(f"Chaining pipeline starting from active job {prev_job_id}...", flush=True)

    for tid, cond, axis in CASES:
        label = axis if axis else cond
        dep_arg = f"--dependency=afterany:{prev_job_id}"
        
        if axis:
            cmd = f"sbatch {dep_arg} --array={tid}-{tid} core/run/slurm/run_video_array.sbatch --paraphrase_axis {axis}"
        else:
            cmd = f"sbatch {dep_arg} --array={tid}-{tid} core/run/slurm/run_video_array.sbatch --condition {cond}"
            
        print(f"Submitting: {cmd}", flush=True)
        ret, stdout, stderr = run_cmd(cmd, cwd=str(repo_root))
        if ret != 0:
            print(f"Submission failed: {stderr}", flush=True)
            sys.exit(1)
            
        m = re.search(r"Submitted batch job (\d+)", stdout)
        if not m:
            print(f"Could not parse job ID from: {stdout}", flush=True)
            sys.exit(1)
            
        job_id = m.group(1)
        print(f"  --> Job {job_id} submitted. Chaining next job to it.", flush=True)
        prev_job_id = job_id

    print("\n=== Pipeline submission complete! ===", flush=True)

if __name__ == "__main__":
    main()
