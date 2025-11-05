import cv2
import numpy as np
from typing import Tuple, List, Dict

# ---------- utils ----------
def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def read_video_frames(path: str, max_frames: int = None, to_grayscale: bool = True) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames, n = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if to_grayscale:
            f = to_gray(f)
        frames.append(f.astype(np.uint8))
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    if len(frames) < 2:
        raise ValueError("Need at least 2 frames for motion analysis.")
    return frames

def abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.absdiff(a, b)

def frame_mad(diff: np.ndarray) -> float:
    return float(np.mean(diff))

def clip_range(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def illum_invariant_mad(prev: np.ndarray, curr: np.ndarray) -> float:
    """
    MAD after normalizing global brightness/contrast between frames.
    Reduces false motion from light flicker.
    """
    prev_f = prev.astype(np.float32)
    curr_f = curr.astype(np.float32)

    m0, s0 = float(prev_f.mean()), float(prev_f.std() + 1e-6)
    m1, s1 = float(curr_f.mean()), float(curr_f.std() + 1e-6)
    curr_n = (curr_f - m1) * (s0 / s1) + m0  

    curr_n = np.clip(curr_n, 0.0, 255.0).astype(np.uint8)
    diff = cv2.absdiff(prev.astype(np.uint8), curr_n)
    return float(diff.mean())

# ---------- thresholds ----------
def calibrate_thresholds(frames: List[np.ndarray], calibration_frames: int = 30,
                         low_q: float = 40.0, high_q: float = 80.0) -> Tuple[float, float]:
    n = min(calibration_frames, len(frames) - 1)
    if n <= 0:
        raise ValueError("Not enough frames for calibration.")
    mads = []
    for i in range(1, n + 1):
        d = abs_diff(frames[i], frames[i - 1])
        mads.append(frame_mad(d))
    T_low = float(np.percentile(mads, low_q))
    T_high = float(np.percentile(mads, high_q))
    if T_high < T_low:
        T_high = T_low
    return T_low, T_high

# ---------- block matching ----------
def sad(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(np.abs(a.astype(np.int32) - b.astype(np.int32))))

def full_search_bma(curr: np.ndarray, ref: np.ndarray, by: int, bx: int,
                    block: int, search: int) -> Tuple[int, int, int]:
    h, w = curr.shape
    y0, x0 = by * block, bx * block
    cur_blk = curr[y0:y0 + block, x0:x0 + block]

    best_cost, best_dy, best_dx = 10**18, 0, 0
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            yy = clip_range(y0 + dy, 0, h - block)
            xx = clip_range(x0 + dx, 0, w - block)
            cost = sad(cur_blk, ref[yy:yy + block, xx:xx + block])
            if cost < best_cost:
                best_cost, best_dy, best_dx = cost, yy - y0, xx - x0
    return best_dy, best_dx, int(best_cost)

def diamond_search(curr: np.ndarray, ref: np.ndarray, by: int, bx: int,
                   block: int, search: int) -> Tuple[int, int, int]:
    h, w = curr.shape
    y0, x0 = by * block, bx * block
    cur_blk = curr[y0:y0 + block, x0:x0 + block]

    def cost_at(dy, dx):
        yy = clip_range(y0 + dy, 0, h - block)
        xx = clip_range(x0 + dx, 0, w - block)
        return sad(cur_blk, ref[yy:yy + block, xx:xx + block])

    LDSP = [(0, 0), (0, -2), (0, 2), (-2, 0), (2, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    SDSP = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

    best_dy, best_dx = 0, 0
    best_cost = cost_at(0, 0)

    step = 2
    improved = True
    while improved and max(abs(best_dy), abs(best_dx)) + step <= search:
        improved = False
        for dy, dx in LDSP:
            cy, cx = best_dy + dy, best_dx + dx
            if max(abs(cy), abs(cx)) > search:
                continue
            c = cost_at(cy, cx)
            if c < best_cost:
                best_cost, best_dy, best_dx, improved = c, cy, cx, True

    improved = True
    while improved:
        improved = False
        for dy, dx in SDSP:
            cy, cx = best_dy + dy, best_dx + dx
            if max(abs(cy), abs(cx)) > search:
                continue
            c = cost_at(cy, cx)
            if c < best_cost:
                best_cost, best_dy, best_dx, improved = c, cy, cx, True

    return best_dy, best_dx, int(best_cost)

def estimate_motion(curr: np.ndarray, ref: np.ndarray, block: int = 16,
                    search_range: int = 8, method: str = "full") -> Dict[str, np.ndarray]:
    H, W = curr.shape
    ny, nx = H // block, W // block
    mv_y = np.zeros((ny, nx), dtype=np.int16)
    mv_x = np.zeros((ny, nx), dtype=np.int16)
    cost = np.zeros((ny, nx), dtype=np.int32)

    for by in range(ny):
        for bx in range(nx):
            if method == "full":
                dy, dx, c = full_search_bma(curr, ref, by, bx, block, search_range)
            elif method == "diamond":
                dy, dx, c = diamond_search(curr, ref, by, bx, block, search_range)
            else:
                raise ValueError("Unknown method: " + method)
            mv_y[by, bx], mv_x[by, bx], cost[by, bx] = dy, dx, c
    return {"mv_y": mv_y, "mv_x": mv_x, "cost": cost}

def motion_compensate(ref: np.ndarray, mv_y: np.ndarray, mv_x: np.ndarray, block: int = 16) -> np.ndarray:
    H, W = ref.shape
    ny, nx = mv_y.shape
    out = np.zeros_like(ref)
    for by in range(ny):
        for bx in range(nx):
            y0, x0 = by * block, bx * block
            dy, dx = int(mv_y[by, bx]), int(mv_x[by, bx])
            yy = np.clip(y0 + dy, 0, H - block)
            xx = np.clip(x0 + dx, 0, W - block)
            out[y0:y0 + block, x0:x0 + block] = ref[yy:yy + block, xx:xx + block]
    return out

# ---------- pipeline ----------
def classify_motion(mad_value: float, T_low: float, T_high: float) -> str:
    if mad_value < T_low:
        return "LOW"
    elif mad_value > T_high:
        return "HIGH"
    else:
        return "MEDIUM"

def process_video(frames: List[np.ndarray],
                  block: int = 16,
                  search_range: int = 8,
                  method: str = "full",
                  calibration_frames: int = 30,
                  low_q: float = 40.0,
                  high_q: float = 80.0,
                  low_diff_threshold: int = 10) -> Dict:
    """
    low_q/high_q: percentile thresholds for T_low/T_high calibration
    low_diff_threshold: pixel threshold for 'low_diff_ratio' (|Δ| < threshold)
    """
    import time

    T_low, T_high = calibrate_thresholds(frames,
                                         calibration_frames=calibration_frames,
                                         low_q=low_q, high_q=high_q)
    decisions = []   
    outputs = {"I_frames": [], "diff_frames": [], "ME": []}

    for i in range(1, len(frames)):
        prev_f, curr_f = frames[i - 1], frames[i]

        # raw diff + illumination-invariant MAD
        diff = abs_diff(curr_f, prev_f)
        raw_mad = frame_mad(diff)                      
        mad = illum_invariant_mad(prev_f, curr_f)     
        p90 = float(np.percentile(diff, 90))
        low_diff_ratio = float((diff < low_diff_threshold).mean())  
        cls = classify_motion(mad, T_low, T_high)

        if cls == "LOW":
            decisions.append({
                "frame": i, "mad": mad, "raw_mad": raw_mad, "class": "LOW",
                "mean_cost": np.nan, "psnr": np.nan, "time_ms": 0.0,
                "p90_diff": p90, "low_diff_ratio": low_diff_ratio
            })
            outputs["diff_frames"].append({"index": i, "diff": diff})

        elif cls == "MEDIUM":
            t0 = time.perf_counter()
            me = estimate_motion(curr_f, prev_f, block=block, search_range=search_range, method=method)
            comp = motion_compensate(prev_f, me["mv_y"], me["mv_x"], block=block)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            mean_cost = float(np.mean(me["cost"]))
            mse = float(np.mean((curr_f.astype(np.float32) - comp.astype(np.float32)) ** 2))
            psnr = float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 1e-9 else float("inf")

            decisions.append({
                "frame": i, "mad": mad, "raw_mad": raw_mad, "class": "MEDIUM",
                "mean_cost": mean_cost, "psnr": psnr, "time_ms": dt_ms,
                "p90_diff": p90, "low_diff_ratio": low_diff_ratio
            })
            outputs["ME"].append({
                "index": i, "mv_x": me["mv_x"], "mv_y": me["mv_y"],
                "cost": me["cost"], "compensated": comp, "diff": diff,
                "time_ms": dt_ms, "mean_cost": mean_cost, "psnr": psnr
            })

        else:  
            decisions.append({
                "frame": i, "mad": mad, "raw_mad": raw_mad, "class": "HIGH",
                "mean_cost": np.nan, "psnr": np.nan, "time_ms": 0.0,
                "p90_diff": p90, "low_diff_ratio": low_diff_ratio
            })
            outputs["I_frames"].append({"index": i, "frame": curr_f})

    meta = {
        "T_low": T_low, "T_high": T_high,
        "block": block, "search_range": search_range,
        "method": method, "calibration_frames": calibration_frames,
        "low_q": low_q, "high_q": high_q, "low_diff_threshold": low_diff_threshold
    }
    return {"decisions": decisions, "outputs": outputs, "meta": meta}

# ---------- visualization ----------
def motion_quiver_image(frame: np.ndarray, mv_y: np.ndarray, mv_x: np.ndarray,
                        block: int = 16, stride: int = 1) -> np.ndarray:
    """
    Draw motion vectors on top of a grayscale frame.
    Set `stride` > 1 to skip arrows and reduce clutter (e.g., stride=2).
    """
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    ny, nx = mv_y.shape
    for by in range(0, ny, stride):
        for bx in range(0, nx, stride):
            y0 = by * block + block // 2
            x0 = bx * block + block // 2
            dy, dx = int(mv_y[by, bx]), int(mv_x[by, bx])
            if dy == 0 and dx == 0:
                continue
            cv2.arrowedLine(vis, (x0, y0), (x0 + dx, y0 + dy), (0, 255, 0), 1, tipLength=0.3)
    return vis

def diff_heatmap(diff: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(diff.astype(np.uint8), cv2.COLORMAP_JET)
