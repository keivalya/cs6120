#!/usr/bin/env python3
"""core/analyze/update_video_readme.py — Scan data/videos and generate task-wise Markdown tables in data/videos/README.md
"""
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = REPO_ROOT / "data" / "videos"
README_PATH = VIDEO_DIR / "README.md"

def main():
    instructions_map = {}
    gen_file = REPO_ROOT / "data" / "instructions" / "libero_goal.jsonl"
    if gen_file.exists():
        for line in gen_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                tid = int(r["task_id"])
                cond = r["condition"]
                instr = r.get("instruction", "")
                instructions_map[(tid, cond)] = instr

    task_names = {
        0: "open_the_middle_drawer_of_the_cabinet",
        1: "put_the_bowl_on_the_stove",
        2: "put_the_wine_bottle_on_top_of_the_cabinet",
        3: "open_the_top_drawer_and_put_the_bowl_inside",
        4: "put_the_bowl_on_top_of_the_cabinet",
        5: "push_the_plate_to_the_front_of_the_stove",
        6: "put_the_cream_cheese_in_the_bowl",
        7: "turn_on_the_stove",
        8: "put_the_bowl_on_the_plate",
        9: "put_the_wine_bottle_on_the_rack",
    }

    all_videos = sorted(list(VIDEO_DIR.glob("**/*.mp4")))

    tasks_data = {}
    for vpath in all_videos:
        rel_path = os.path.relpath(vpath, REPO_ROOT)
        filename = vpath.name
        status = "Success" if "success" in filename.lower() else "Failure"
        
        parts = vpath.relative_to(VIDEO_DIR).parts
        if len(parts) >= 2:
            task_dir = parts[0]
            cond_dir = parts[1] if len(parts) > 2 else "original"
        elif len(parts) == 1:
            task_dir = parts[0]
            cond_dir = "original"
        else:
            continue
            
        tid = None
        if task_dir.startswith("task"):
            try:
                tid = int(task_dir.split("_")[0].replace("task", ""))
            except ValueError:
                pass
                
        if tid is None:
            continue
            
        default_instr = instructions_map.get((tid, "original"), task_names.get(tid, ""))
        instr = instructions_map.get((tid, cond_dir), default_instr)
        
        if tid not in tasks_data:
            tasks_data[tid] = []
            
        tasks_data[tid].append({
            "rel_path": rel_path,
            "status": status,
            "condition": cond_dir,
            "filename": filename,
            "instruction": instr
        })

    doc_lines = [
        "# VLA Simulation Rollout Videos Index",
        "",
        "This directory contains rollout videos generated during Vision-Language-Action (VLA) policy evaluations on LIBERO-Goal tasks.",
        "",
        "## Summary of Tasks",
        ""
    ]

    for tid in sorted(task_names.keys()):
        tname = task_names[tid]
        entries = tasks_data.get(tid, [])
        doc_lines.append(f"### Task {tid}: `{tname}`")
        doc_lines.append("")
        
        if not entries:
            doc_lines.append("_No rollout videos generated yet for this task._")
            doc_lines.append("")
            continue
            
        doc_lines.append("| Video File | Outcome Status | Condition | Text Instruction |")
        doc_lines.append("| :--- | :---: | :---: | :--- |")
        
        for e in entries:
            file_link = f"[{e['filename']}]({e['rel_path']})"
            status_badge = f"🟢 **{e['status']}**" if e['status'] == "Success" else f"🔴 **{e['status']}**"
            instr_str = f"`{e['instruction']}`" if e['instruction'] else "_N/A_"
            doc_lines.append(f"| {file_link} | {status_badge} | `{e['condition']}` | {instr_str} |")
            
        doc_lines.append("")

    README_PATH.write_text("\n".join(doc_lines))
    print(f"[update_video_readme] Updated {README_PATH} with {len(all_videos)} video entries across {len(tasks_data)} tasks.")

if __name__ == "__main__":
    main()
