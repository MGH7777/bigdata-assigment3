import csv
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
from motion_codec import (
    read_video_frames, process_video, motion_quiver_image, diff_heatmap
)

# ----------- helpers -----------
def make_synthetic_video(width=256, height=192, frames=60, square=32, velocity=(2, 1)) -> list:
    seq, x, y = [], 30, 40
    vx, vy = velocity
    for _ in range(frames):
        img = np.zeros((height, width), dtype=np.uint8)
        x0, y0 = int(x), int(y)
        x1, y1 = x0 + square, y0 + square
        cv2.rectangle(img, (x0, y0), (x1, y1), 200, -1)
        seq.append(img); x += vx; y += vy
    return seq

def safe_write_csv(path, header, rows):
    """Write CSV atomically and avoid PermissionError when file is open in Excel."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        w.writerows(rows)
    try:
        os.replace(tmp, path)  
        print(f"Saved: {path}")
    except PermissionError:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.replace(".csv", f"_{stamp}.csv")
        os.replace(tmp, alt)
        print(f"[WARN] '{path}' locked; wrote to '{alt}' instead.")

def prompt_for_video_frames():
    while True:
        try:
            path = input("Enter video path (or press Enter for synthetic): ").strip()
        except EOFError:
            path = ""
        if path == "":
            print("[INFO] Using synthetic demo.")
            return None
        path = path.strip('"').strip("'")
        if not os.path.exists(path):
            print(f"[ERROR] Path does not exist: {path}")
            continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[ERROR] OpenCV could not open: {path}")
            continue
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        if n < 2:
            print(f"[ERROR] Video has {n} frame(s); need at least 2.")
            continue
        try:
            frames = read_video_frames(path, to_grayscale=True)
            print(f"[INFO] Loaded {len(frames)} frames from {path}")
            return frames
        except Exception as e:
            print(f"[ERROR] Failed to read frames: {e}")

# ----------- main -----------
if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)

    frames = prompt_for_video_frames()
    if frames is None:
        frames = make_synthetic_video()

    # Compare Full vs Diamond (same decision logic; different ME on MEDIUM frames)
    # low_q/high_q (percentiles), low_diff_threshold (pixel)
    result_full = process_video(frames, block=16, search_range=8, method="full",
                                calibration_frames=20, low_q=40.0, high_q=80.0,
                                low_diff_threshold=10)
    result_dia  = process_video(frames, block=16, search_range=8, method="diamond",
                                calibration_frames=20, low_q=40.0, high_q=80.0,
                                low_diff_threshold=10)

    # Visualize one ME frame from diamond (if any) on the actual current frame
    me_frames = list(result_dia["outputs"]["ME"])
    if me_frames:
        sample = me_frames[len(me_frames)//2]
        frame_for_viz = frames[sample["index"]]
        quiv = motion_quiver_image(frame_for_viz, sample["mv_y"], sample["mv_x"], block=16, stride=2)
        cv2.imwrite("out/motion_vectors.png", quiv)

        heat = diff_heatmap(sample["diff"])
        cv2.imwrite("out/diff_heatmap.png", heat)

    # Per-frame decisions (both runs) — metrics for ALL classes
    rows_full = [[d["frame"], d["mad"], d["raw_mad"], d["class"], d["time_ms"],
                  d["mean_cost"], d["psnr"], d["p90_diff"], d.get("low_diff_ratio")]
                 for d in result_full["decisions"]]
    safe_write_csv(
        "out/decisions_full.csv",
        ["frame_index", "MAD_norm", "MAD_raw", "class", "time_ms", "mean_sad_cost", "psnr", "p90_diff", "low_diff_ratio"],
        rows_full
    )

    rows_dia = [[d["frame"], d["mad"], d["raw_mad"], d["class"], d["time_ms"],
                 d["mean_cost"], d["psnr"], d["p90_diff"], d.get("low_diff_ratio")]
                for d in result_dia["decisions"]]
    safe_write_csv(
        "out/decisions_diamond.csv",
        ["frame_index", "MAD_norm", "MAD_raw", "class", "time_ms", "mean_sad_cost", "psnr", "p90_diff", "low_diff_ratio"],
        rows_dia
    )

    # Method-level metrics table (only MEDIUM frames have timing/psnr)
    rows_metrics = []
    for e in result_full["outputs"]["ME"]:
        rows_metrics.append(["full", e["index"], e["time_ms"], e["mean_cost"], e["psnr"]])
    for e in result_dia["outputs"]["ME"]:
        rows_metrics.append(["diamond", e["index"], e["time_ms"], e["mean_cost"], e["psnr"]])
    safe_write_csv("out/method_metrics.csv",
                   ["method", "frame_index", "time_ms", "mean_sad_cost", "psnr_compensation"],
                   rows_metrics)

    def summarize(name, res):
        from collections import Counter
        c = Counter([d["class"] for d in res["decisions"]])
        meta = res["meta"]
        print(f"[{name}] LOW={c.get('LOW',0)} MEDIUM={c.get('MEDIUM',0)} HIGH={c.get('HIGH',0)} "
              f"| T_low={meta['T_low']:.2f} T_high={meta['T_high']:.2f} "
              f"| low_q={meta['low_q']} high_q={meta['high_q']} low_diff_thr={meta['low_diff_threshold']}")

    summarize("full", result_full)
    summarize("diamond", result_dia)

    print("Saved outputs to ./out")
