#!/usr/bin/env python3
import subprocess
import re
import sys
import json
from pathlib import Path

CASES = [
    (7, "original", None),
    (7, "blank", None),
    (7, "nonsense", None),
    (0, "original", None),
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

def get_task_name(tid, repo_root):
    gen_file = repo_root / "perturb" / "generated" / "libero_goal.jsonl"
    for line in gen_file.read_text().splitlines():
        r = json.loads(line)
        if int(r["task_id"]) == tid and r["condition"] == "original":
            return r["task_name"]
    return f"task_{tid}"

def is_case_completed(tid, cond, axis, repo_root, output_dir="output_videos", target_num=5):
    task_name = get_task_name(tid, repo_root)
    label = axis if axis else cond
    out_folder = repo_root / output_dir / f"task{tid}_{task_name}" / label
    if out_folder.exists():
        mp4_files = list(out_folder.glob("*.mp4"))
        if len(mp4_files) >= target_num:
            return True
    return False

def get_active_job_for_task(tid):
    _, out, _ = run_cmd("squeue -u $USER -o '%i %j' -h")
    if not out:
        return None
    for line in out.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            job_id, name = parts[0], parts[1]
            if name == "vla-video":
                if "_" in job_id:
                    array_id, task_idx = job_id.split("_")
                    if task_idx.isdigit() and int(task_idx) == tid:
                        return array_id
                else:
                    _, job_info, _ = run_cmd(f"scontrol show job {job_id}")
                    if f"ArrayTaskId={tid}" in job_info or f"ArrayTaskId={tid}-{tid}" in job_info:
                        return job_id
    return None

def get_total_queued_jobs():
    _, out, _ = run_cmd("squeue -u $USER -o '%j' -h")
    if not out:
        return 0
    count = 0
    for line in out.strip().splitlines():
        if line.strip() == "vla-video":
            count += 1
    return count

def main():
    repo_root = Path(__file__).resolve().parents[2]
    
    # 1. Check current queue size
    total_queued = get_total_queued_jobs()
    print(f"[Cron Scheduler] Total vla-video jobs in queue: {total_queued}", flush=True)
    
    # Limit of total active/pending jobs in Slurm to prevent QOS submit limits
    if total_queued >= 6:
        print("[Cron Scheduler] Queue has 6 or more jobs. Skipping submission to respect QOS limits.", flush=True)
        return

    # 2. Find the first eligible case to submit
    for i, (tid, cond, axis) in enumerate(CASES):
        label = axis if axis else cond
        
        # Check if already completed
        if is_case_completed(tid, cond, axis, repo_root):
            continue
            
        # Check if there is already a running/pending job for this task ID
        active_job = get_active_job_for_task(tid)
        if active_job:
            print(f"[Cron Scheduler] Case {tid} ({label}) is not completed, but job {active_job} is already active for Task {tid}. Waiting.", flush=True)
            return

        # Submit it!
        if axis:
            cmd = f"sbatch --array={tid}-{tid} run/slurm/run_video_array.sbatch --paraphrase_axis {axis}"
        else:
            cmd = f"sbatch --array={tid}-{tid} run/slurm/run_video_array.sbatch --condition {cond}"
            
        print(f"[Cron Scheduler] Submitting first eligible case: {cmd}", flush=True)
        ret, stdout, stderr = run_cmd(cmd, cwd=str(repo_root))
        if ret != 0:
            print(f"[Cron Scheduler] Error submitting job: {stderr}", flush=True)
        else:
            print(f"[Cron Scheduler] Successfully submitted: {stdout}", flush=True)
        return

    print("[Cron Scheduler] All cases completed or queued!", flush=True)

if __name__ == "__main__":
    main()
