#!/usr/bin/env python3
import subprocess
import time
import re
import os
import sys
import json
from pathlib import Path

# List of cases: (task_id, condition, paraphrase_axis)
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

def is_job_running(job_id):
    _, out, _ = run_cmd(f"squeue -j {job_id} -h")
    return len(out.strip().splitlines()) > 0 if out else False

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

def main():
    repo_root = Path(__file__).resolve().parents[2]
    print(f"Starting VLA Evidence Video Generation Scheduler...", flush=True)

    for i, (tid, cond, axis) in enumerate(CASES):
        label = axis if axis else cond
        print(f"\n--- [Case {i+1}/{len(CASES)}] Task {tid} ({label}) ---", flush=True)

        # 1. Check if already completed
        if is_case_completed(tid, cond, axis, repo_root):
            print(f"Case {tid} ({label}) is already completed. Skipping.", flush=True)
            continue

        # 2. Check if a job is already running on the cluster for this task
        active_job = get_active_job_for_task(tid)
        if active_job:
            print(f"Active Job {active_job} is already running for Task {tid} on the cluster. Waiting for completion...", flush=True)
            job_id = active_job
        else:
            # 3. Submit the job
            if axis:
                cmd = f"sbatch --array={tid}-{tid} run/slurm/run_video_array.sbatch --paraphrase_axis {axis}"
            else:
                cmd = f"sbatch --array={tid}-{tid} run/slurm/run_video_array.sbatch --condition {cond}"

            print(f"Submitting job: {cmd}", flush=True)
            ret, stdout, stderr = run_cmd(cmd, cwd=str(repo_root))
            if ret != 0:
                print(f"Error submitting job: {stderr}", flush=True)
                if "Limit" in stderr or "limit" in stderr:
                    print("Hit submit limit. Waiting 2 minutes to retry...", flush=True)
                    time.sleep(120)
                    ret, stdout, stderr = run_cmd(cmd, cwd=str(repo_root))
                    if ret != 0:
                        print(f"Retry failed: {stderr}. Skipping.", flush=True)
                        continue
                else:
                    continue

            m = re.search(r"Submitted batch job (\d+)", stdout)
            if not m:
                print(f"Could not parse job ID: {stdout}", flush=True)
                continue
            job_id = m.group(1)
            print(f"Job {job_id} submitted successfully. Waiting for completion...", flush=True)

        # 4. Wait for job to complete
        time.sleep(10)
        while True:
            running = is_job_running(job_id)
            if not running:
                print(f"Job {job_id} has finished.", flush=True)
                break
            print(f"[Scheduler] Job {job_id} is still running (polling squeue)...", flush=True)
            time.sleep(30)

    print("\n=== All Evidence Videos Generated successfully! ===", flush=True)

if __name__ == "__main__":
    main()
