#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# --------- deps ----------
try:
    import cv2
except Exception:
    raise SystemExit("OpenCV (cv2) is required. Install with: pip install opencv-python")

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# --------- file handling ----------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}

def looks_like_dicom(path: Path) -> bool:
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
        return len(pre) >= 132 and pre[128:132] == b"DICM"
    except Exception:
        return False

def imread_any(path: Path) -> np.ndarray:
    # Raster formats
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        raise IOError(f"Failed to read raster image: {path}")

    # DICOM (when it really is one)
    if looks_like_dicom(path):
        try:
            import pydicom
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array.astype(np.float32)

            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

            def _first(x):
                if isinstance(x, (list, tuple)) and len(x):
                    return float(x[0])
                try:
                    return float(x)
                except Exception:
                    return None

            wc = _first(getattr(ds, "WindowCenter", None))
            ww = _first(getattr(ds, "WindowWidth", None))
            if wc is not None and ww is not None and ww > 0:
                lo, hi = wc - ww/2.0, wc + ww/2.0
                arr = np.clip(arr, lo, hi)
            else:
                lo, hi = np.percentile(arr, (1, 99))
                if hi > lo:
                    arr = np.clip(arr, lo, hi)

            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn) * 255.0
            arr8 = arr.astype(np.uint8)

            if arr8.ndim == 2:
                return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)
            if arr8.ndim == 3 and arr8.shape[2] == 3:
                return cv2.cvtColor(arr8, cv2.COLOR_RGB2BGR)
            return cv2.cvtColor(arr8.squeeze(), cv2.COLOR_GRAY2BGR)
        except Exception as e:
            raise IOError(f"Failed to read DICOM: {path} ({e})")

    raise IOError(f"Unsupported file type (not image/DICOM): {path}")

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------- preprocessing ----------
def clahe_gray(img_bgr: np.ndarray, clip=2.0, tile_grid=(8, 8)) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid)
    return clahe.apply(gray)

def denoise_blur(gray: np.ndarray) -> np.ndarray:
    try:
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    except Exception:
        return cv2.GaussianBlur(gray, (5, 5), 1.2)

# --------- teacher's fixed-radius Hough ----------
def houghcircle_py(Imbinary: np.ndarray, r: int, thresh: int) -> Tuple[Optional[Tuple[int,int]], np.ndarray]:
    """
    Python port of MATLAB houghcircle.m for a single fixed radius.
    Imbinary: uint8 binary edges (0/255). r: radius in px. thresh: min votes (>=4 typical).
    Returns (y0, x0) for the best center (or None), and the accumulator array.
    """
    if r < 3:
        return None, np.zeros_like(Imbinary, dtype=np.int32)

    # Ensure 0/1 for voting
    edges = (Imbinary > 0).astype(np.uint8)
    ys, xs = np.where(edges)
    H, W = Imbinary.shape[:2]

    acc = np.zeros((H, W), dtype=np.int32)
    r2 = r * r

    # Vote like the MATLAB loop (vectorized per edge-pixel over x0 range)
    for (y, x) in zip(ys, xs):
        x_low = max(0, x - r)
        x_high = min(W - 1, x + r)
        x0s = np.arange(x_low, x_high + 1)
        dx = x - x0s
        dy_sq = r2 - dx*dx
        mask = dy_sq >= 0
        if not np.any(mask):
            continue
        y_off = np.sqrt(dy_sq[mask])
        x0s_valid = x0s[mask]

        y01 = np.round(y - y_off).astype(int)
        y02 = np.round(y + y_off).astype(int)

        # In-bounds increments
        valid1 = (y01 >= 0) & (y01 < H)
        acc[y01[valid1], x0s_valid[valid1]] += 1

        valid2 = (y02 >= 0) & (y02 < H)
        acc[y02[valid2], x0s_valid[valid2]] += 1

    # Local maxima (3x3 nms) and threshold
    acc_u16 = acc.astype(np.uint16)  # for cv2.dilate
    kernel = np.ones((3, 3), np.uint8)
    acc_dil = cv2.dilate(acc_u16, kernel)
    is_max = (acc_u16 == acc_dil)
    keep = (acc >= int(thresh)) & is_max

    ys_max, xs_max = np.where(keep)
    if ys_max.size == 0:
        return None, acc

    # pick strongest
    votes = acc[ys_max, xs_max]
    idx = int(np.argmax(votes))
    y0, x0 = int(ys_max[idx]), int(xs_max[idx])
    return (y0, x0), acc

def detect_best_circle_teacher(
    img_bgr: np.ndarray,
    r_fixed_px: int,
    acc_thresh: int = 12,
    canny_low: int = 80,
    canny_high: int = 180
) -> Tuple[Optional[Tuple[float,float,float]], dict]:
    """
    End-to-end teacher-style detection:
      gray -> denoise -> edges -> fixed-radius Hough accumulator -> best (y,x)
    Returns (cx, cy, r) in float like OpenCV, plus diagnostics.
    """
    gray0 = clahe_gray(img_bgr)
    gray = denoise_blur(gray0)

    # Edges like typical MATLAB workflows
    edges = cv2.Canny(gray, canny_low, canny_high)

    center_yx, acc = houghcircle_py(edges, int(r_fixed_px), int(acc_thresh))
    if center_yx is None:
        return None, {"mode": "teacher", "reason": "no_maxima", "r_fixed_px": int(r_fixed_px)}

    cy, cx = center_yx
    r = float(r_fixed_px)
    return (float(cx), float(cy), r), {
        "mode": "teacher",
        "r_fixed_px": int(r_fixed_px),
        "votes_at_peak": int(acc[cy, cx]),
    }

# --------- viz ----------
def overlay_result(img: np.ndarray, circle: Optional[tuple]) -> np.ndarray:
    vis = img.copy()
    if circle is None:
        cv2.putText(vis, "NO CIRCLE FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        return vis
    cx, cy, r = circle
    cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 0), 3)
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)
    return vis

def save_panel(out_path: Path, original: np.ndarray, pre_gray: np.ndarray, edges: np.ndarray, vis: np.ndarray):
    if not HAS_PLT:
        cv2.imwrite(str(out_path), vis)
        return
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1); ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); ax1.set_title("Original"); ax1.axis("off")
    ax2 = plt.subplot(2, 2, 2); ax2.imshow(pre_gray, cmap="gray"); ax2.set_title("Preprocessed (CLAHE+denoise)"); ax2.axis("off")
    ax3 = plt.subplot(2, 2, 3); ax3.imshow(edges, cmap="gray"); ax3.set_title("Edges (Canny)"); ax3.axis("off")
    ax4 = plt.subplot(2, 2, 4); ax4.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax4.set_title("Fixed-radius Hough result"); ax4.axis("off")
    plt.tight_layout(); plt.savefig(str(out_path), dpi=150); plt.close()

# --------- batch ----------
def process_folder(
    in_dir: Path,
    out_dir: Path,
    recursive: bool,
    max_side: Optional[int],
    fixed_radius_px: Optional[int],
    fixed_radius_mm: Optional[float],
    fixed_radius_ratio: Optional[float],
    use_teacher: bool,
    acc_thresh: int,
    canny_low: int,
    canny_high: int,
):
    ensure_dir(out_dir)
    cand_paths = (in_dir.rglob("*") if recursive else in_dir.iterdir())
    cand_paths = [p for p in cand_paths if p.is_file()]

    skip_tokens = ("_panel", "bestsofar", "result_", "findthecentreoftheheart", "radius", "sensitivity")
    files = [
        p for p in cand_paths
        if all(tok not in p.stem.lower() for tok in skip_tokens)
        and (p.suffix.lower() in VALID_EXTS or (p.suffix == "" and looks_like_dicom(p)))
    ]
    files = sorted(files)

    print(f"Found {len(files)} candidate file(s) in {in_dir} (panels/results skipped)")
    if not files:
        return

    ok = 0; no = 0; err = 0
    for i, img_path in enumerate(files):
        try:
            print(f"[{i+1}/{len(files)}] {img_path.name}")
            img = imread_any(img_path)

            # optional downscale
            if max_side is not None and max_side > 0:
                H, W = img.shape[:2]
                scale = max(H, W) / float(max_side)
                if scale > 1:
                    img = cv2.resize(img, (int(W/scale), int(H/scale)), interpolation=cv2.INTER_AREA)

            # derive fixed radius in pixels
            r_px = None
            if fixed_radius_px is not None:
                r_px = int(max(3, fixed_radius_px))
            elif fixed_radius_mm is not None:
                try:
                    import pydicom
                    ds = pydicom.dcmread(str(img_path), force=True)
                    sp = getattr(ds, "PixelSpacing", None)
                    if sp and len(sp) >= 2:
                        mm_per_px = float(sp[0] + sp[1]) / 2.0
                        if mm_per_px > 0:
                            r_px = int(round(float(fixed_radius_mm) / mm_per_px))
                except Exception:
                    r_px = None
            elif fixed_radius_ratio is not None:
                H, W = img.shape[:2]
                r_px = int(round(float(fixed_radius_ratio) * min(H, W)))

            # Preprocess + edges for the panel
            gray = denoise_blur(clahe_gray(img))
            edges = cv2.Canny(gray, canny_low, canny_high)

            circle = None; diag = {}
            # Teacher path if radius known OR user forced it
            if r_px is not None or use_teacher:
                if r_px is None:
                    # fallback: estimate radius from image size (conservative)
                    H, W = img.shape[:2]
                    r_px = int(round(0.22 * min(H, W)))
                circle, diag = detect_best_circle_teacher(
                    img, r_fixed_px=r_px, acc_thresh=acc_thresh,
                    canny_low=canny_low, canny_high=canny_high
                )
            else:
                # If you ever want to keep an OpenCV fallback, you could add it here.
                circle, diag = detect_best_circle_teacher(
                    img, r_fixed_px=int(round(0.22 * min(img.shape[:2]))),
                    acc_thresh=acc_thresh, canny_low=canny_low, canny_high=canny_high
                )

            vis = overlay_result(img, circle)
            panel_path = out_dir / f"{img_path.stem}_panel.png"
            save_panel(panel_path, img, gray, edges, vis)

            if circle is None:
                no += 1
            else:
                ok += 1
        except Exception as e:
            print("ERROR:", e)
            err += 1

    total = ok + no + err
    print(f"\nSummary: {ok}/{total} OK ({100.0*ok/max(total,1):.1f}%) | NO_CIRCLE: {no} | ERROR: {err}")

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Fixed-radius Hough (teacher version) for heart center on 'herzinfarkt' images.")
    ap.add_argument("--images", type=str, help="Folder with input images.")
    ap.add_argument("--out", type=str, help="Output folder for panels.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--max-side", type=int, default=1024, help="Downscale longest side to this many pixels.")

    # Teacher Hough controls
    ap.add_argument("--use-teacher", action="store_true",
                    help="Force teacher fixed-radius Hough even if no fixed radius provided.")
    ap.add_argument("--fixed-radius-px", type=int, default=None,
                    help="Fixed circle radius in pixels.")
    ap.add_argument("--fixed-radius-mm", type=float, default=None,
                    help="Fixed circle radius in millimeters; needs DICOM PixelSpacing to convert.")
    ap.add_argument("--fixed-radius-ratio", type=float, default=None,
                    help="Fixed radius as a fraction of min(H,W), e.g., 0.22.")

    ap.add_argument("--acc-thresh", type=int, default=12,
                    help="Accumulator vote threshold for a valid center (≥4 is legal; 8–20 typical).")
    ap.add_argument("--canny-low", type=int, default=80, help="Canny lower threshold.")
    ap.add_argument("--canny-high", type=int, default=180, help="Canny upper threshold.")

    args = ap.parse_args()

    # Prompt for paths if missing (same UX as your earlier scripts)
    if not args.images:
        try:
            args.images = input("Enter path to your images: ").strip('"').strip()
        except EOFError:
            args.images = None
    if not args.out:
        try:
            default_out = "out_herzinfarkt"
            user_out = input(f"Enter output folder (default: {default_out}): ").strip('"').strip()
            args.out = user_out or default_out
        except EOFError:
            args.out = "out_herzinfarkt"

    if not args.images:
        raise SystemExit("No images folder provided.")
    in_dir = Path(args.images)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (in_dir.parent / out_dir).resolve()

    if not in_dir.exists():
        raise SystemExit(f"Input dir does not exist: {in_dir}")
    ensure_dir(out_dir)

    process_folder(
        in_dir=in_dir,
        out_dir=out_dir,
        recursive=args.recursive,
        max_side=args.max_side,
        fixed_radius_px=args.fixed_radius_px,
        fixed_radius_mm=args.fixed_radius_mm,
        fixed_radius_ratio=args.fixed_radius_ratio,
        use_teacher=args.use_teacher,
        acc_thresh=args.acc_thresh,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )

    print(f"\nDone. Results saved in:\n  {out_dir}\n- Panels: {out_dir}/*.png")

if __name__ == "__main__":
    main()
