#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Tuple, Optional, Literal
import numpy as np

# ---------- Optional dependencies ----------
try:
    import cv2
except Exception:
    raise SystemExit("OpenCV (cv2) is required. Install with: pip install opencv-python")

try:
    # Stronger ellipse Hough (optional)
    from skimage.transform import hough_ellipse
    HAS_SKIMG = True
except Exception:
    HAS_SKIMG = False

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


# ---------- File filtering ----------
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm"}  # allow extensionless DICOM via magic

def looks_like_dicom(path: Path) -> bool:
    """Fast heuristic: .dcm extension OR DICM magic after 128-byte preamble."""
    if path.suffix.lower() == ".dcm":
        return True
    try:
        with open(path, "rb") as f:
            pre = f.read(132)
        return len(pre) >= 132 and pre[128:132] == b"DICM"
    except Exception:
        return False


# ---------- IO ----------
def imread_any(path: Path) -> np.ndarray:
    """
    Read a raster image (png/jpg/...) or DICOM. Skip everything else.
    """
    import numpy as np, cv2

    # 1) Raster formats first
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        raise IOError(f"Failed to read raster image: {path}")

    # 2) DICOM only if it actually looks like DICOM (or extension is .dcm)
    if looks_like_dicom(path):
        try:
            import pydicom
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array.astype(np.float32)

            # rescale if present
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
                lo, hi = wc - ww / 2.0, wc + ww / 2.0
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

    # 3) Everything else → not supported (e.g., .m files)
    raise IOError(f"Unsupported file type (not image/DICOM): {path}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ---------- Preprocessing ----------
def clahe_gray(img_bgr: np.ndarray, clip=2.0, tile_grid=(8, 8)) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid)
    return clahe.apply(gray)

def denoise_blur(gray: np.ndarray) -> np.ndarray:
    try:
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    except Exception:
        return cv2.GaussianBlur(gray, (5, 5), 1.2)

def make_body_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    th = cv2.erode(th, np.ones((5, 5), np.uint8), iterations=1)
    return th

def edge_gradients(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    ang = np.arctan2(gy, gx)
    return gx, gy, mag, ang


# ---------- Sampling & scoring ----------
def bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xs0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    ys0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    xs1 = np.clip(xs0 + 1, 0, w - 1)
    ys1 = np.clip(ys0 + 1, 0, h - 1)
    wa = (xs1 - xs) * (ys1 - ys)
    wb = (xs1 - xs) * (ys - ys0)
    wc = (xs - xs0) * (ys1 - ys)
    wd = (xs - xs0) * (ys - ys0)
    Ia = img[ys0, xs0]; Ib = img[ys1, xs0]; Ic = img[ys0, xs1]; Id = img[ys1, xs1]
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def sample_circle_points(cx: float, cy: float, r: float, n: int = 360):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = cx + r*np.cos(t); ys = cy + r*np.sin(t)
    return xs, ys, np.cos(t), np.sin(t)

def sample_ellipse_points(cx: float, cy: float, a: float, b: float, theta: float, n: int = 360):
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    cos_t, sin_t = np.cos(t), np.sin(t)
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    x_prime = a*cos_t; y_prime = b*sin_t
    xs = cx + x_prime*cos_th - y_prime*sin_th
    ys = cy + x_prime*sin_th + y_prime*cos_th
    nx = (cos_t/a); ny = (sin_t/b)  # normals in local coords
    nx_rot = nx*cos_th - ny*sin_th
    ny_rot = nx*sin_th + ny*cos_th
    norm = np.sqrt(nx_rot**2 + ny_rot**2) + 1e-6
    nx_rot /= norm; ny_rot /= norm
    return xs, ys, nx_rot, ny_rot

def interior_score(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    rr = max(int(r // 2), 3)
    x0 = max(int(cx)-rr, 0); x1 = min(int(cx)+rr, gray.shape[1]-1)
    y0 = max(int(cy)-rr, 0); y1 = min(int(cy)+rr, gray.shape[0]-1)
    patch = gray[y0:y1+1, x0:x1+1]
    if patch.size == 0: return 0.0
    yy, xx = np.ogrid[y0:y1+1, x0:x1+1]
    cmask = (xx-cx)**2 + (yy-cy)**2 <= (0.7*r)**2
    vals = patch[cmask]
    if vals.size == 0: return 0.0
    std = float(np.std(vals))
    return 1.0 / (std + 1e-3)

def circle_support_score(gray, gx, gy, mag, cx, cy, r, pts=240):
    h, w = gray.shape[:2]
    if r < 3 or cx < 0 or cy < 0 or cx >= w or cy >= h:
        return (0.0, 0.0, 0.0)
    xs, ys, nx, ny = sample_circle_points(cx, cy, r, pts)
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, nx, ny = xs[mask], ys[mask], nx[mask], ny[mask]
    mag_s = bilinear_sample(mag, xs, ys)
    gx_s = bilinear_sample(gx, xs, ys)
    gy_s = bilinear_sample(gy, xs, ys)
    grad_norm = np.maximum(gx_s*nx + gy_s*ny, 0)
    edge_support = float(np.mean(mag_s)) if mag_s.size else 0.0
    grad_align   = float(np.mean(grad_norm) / (np.mean(mag_s)+1e-6)) if mag_s.size else 0.0
    hom = interior_score(gray, cx, cy, r)
    return edge_support, grad_align, hom

def ellipse_support_score(gray, gx, gy, mag, cx, cy, a, b, theta, pts=120):
    h, w = gray.shape[:2]
    if min(a, b) < 3 or cx < 0 or cy < 0 or cx >= w or cy >= h:
        return (0.0, 0.0, 0.0)
    xs, ys, nx, ny = sample_ellipse_points(cx, cy, a, b, theta, pts)
    mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs, ys, nx, ny = xs[mask], ys[mask], nx[mask], ny[mask]
    mag_s = bilinear_sample(mag, xs, ys)
    gx_s = bilinear_sample(gx, xs, ys)
    gy_s = bilinear_sample(gy, xs, ys)
    grad_norm = np.maximum(gx_s*nx + gy_s*ny, 0)
    edge_support = float(np.mean(mag_s)) if mag_s.size else 0.0
    grad_align   = float(np.mean(grad_norm) / (np.mean(mag_s)+1e-6)) if mag_s.size else 0.0
    r_eq = 0.7*float(np.sqrt(a*b))  # equivalent radius for interior score
    hom = interior_score(gray, cx, cy, r_eq)
    return edge_support, grad_align, hom


# ---------- Circle detection ----------
def detect_best_circle(
    img_bgr: np.ndarray,
    dp_list=(1.2,),
    min_dist_factor=0.40,
    canny_highs=(120,),
    acc_thresholds=(20,),
    min_radius_ratio=0.12,
    max_radius_ratio=0.35,
) -> Tuple[Optional[Tuple[float, float, float]], dict]:
    H, W = img_bgr.shape[:2]
    gray0 = clahe_gray(img_bgr)
    gray = denoise_blur(gray0)
    body = make_body_mask(gray)
    gx, gy, mag, _ = edge_gradients(gray)
    min_radius = max(5, int(min(H, W) * min_radius_ratio))
    max_radius = max(min_radius + 2, int(min(H, W) * max_radius_ratio))
    min_dist   = int(min(H, W) * min_dist_factor)

    all_candidates = []
    for dp in dp_list:
        for p1 in canny_highs:
            for p2 in acc_thresholds:
                try:
                    circles = cv2.HoughCircles(
                        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                        param1=p1, param2=p2, minRadius=min_radius, maxRadius=max_radius
                    )
                except Exception:
                    circles = None
                if circles is not None and len(circles) > 0:
                    all_candidates.append(circles.reshape(-1, 3))

    di = {"had_candidates": len(all_candidates) > 0}
    if not all_candidates:
        return None, {**di, "reason": "hough_no_circle_candidates"}

    cand = np.vstack(all_candidates)
    best, best_score, parts = None, -1.0, {}
    w_edge, w_align, w_hom = 0.50, 0.20, 0.30
    penalty_outside = 0.60

    h, w = gray.shape[:2]
    for (cx, cy, r) in cand:
        if cx - r < 2 or cy - r < 2 or cx + r > w - 3 or cy + r > h - 3:
            continue
        e, a, hscore = circle_support_score(gray, gx, gy, mag, cx, cy, r, pts=240)
        score = w_edge*e + w_align*a + w_hom*hscore
        if body[int(round(cy)), int(round(cx))] == 0:
            score *= (1.0 - penalty_outside)
        if score > best_score:
            best_score = score
            best = (float(cx), float(cy), float(r))
            parts = {"edge": e, "align": a, "hom": hscore}

    return best, {"best_score": float(best_score), "best_parts": parts}


def detect_best_ellipse(img_bgr: np.ndarray) -> Tuple[Optional[Tuple[float, float, float, float, float]], dict]:
    """
    Fast + stricter ellipse proposal using OpenCV contours only.
    - Body-mask gate for centers
    - Strong size bounds (reject tiny specks / ultra big)
    - Border margin guard
    - Penalize ultra-skinny shapes
    - Fewer scoring samples for speed
    Returns (cx, cy, a, b, theta) with a>=b, theta in radians.
    """
    gray = denoise_blur(clahe_gray(img_bgr))
    gx, gy, mag, _ = edge_gradients(gray)
    body = make_body_mask(gray)

    H, W = gray.shape[:2]
    min_dim = float(min(H, W))

    # ---- constraints (tune here) ----
    MIN_A = 0.18 * min_dim     # minimum semi-major axis
    MIN_B = 0.12 * min_dim     # minimum semi-minor axis
    MAX_A = 0.60 * min_dim     # maximum semi-major axis
    MARGIN = int(0.05 * min_dim)
    MAX_ECC_PENALTY = 0.95     # if ecc > this, penalty applied
    # ---------------------------------

    # Keep edges sparse so we get fewer junk contours
    edges = cv2.Canny(gray, 90, 180)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, {"reason": "no_edges"}

    # Work only on top-K by area (speed)
    K = 40
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:K]

    candidates: list[Tuple[float, float, float, float, float]] = []
    for c in cnts:
        if len(c) < 20:
            continue
        try:
            (cx, cy), (MA, ma), ang_deg = cv2.fitEllipse(c)
        except Exception:
            continue

        a = float(max(MA, ma) / 2.0)  # semi-major
        b = float(min(MA, ma) / 2.0)  # semi-minor
        if a < b:
            a, b = b, a
        if a < MIN_A or b < MIN_B or a > MAX_A:
            continue

        # keep ellipse away from borders (rough bbox)
        if not (MARGIN <= cx <= W - MARGIN and MARGIN <= cy <= H - MARGIN):
            continue
        if (cx - a < MARGIN) or (cx + a > W - MARGIN) or (cy - a < MARGIN) or (cy + a > H - MARGIN):
            continue

        # center must sit inside body mask
        if body[int(round(cy)), int(round(cx))] == 0:
            continue

        th = np.deg2rad(ang_deg)
        candidates.append((float(cx), float(cy), float(a), float(b), float(th)))

    if not candidates:
        return None, {"reason": "no_ellipse_candidates_after_filters"}

    # ---- score (lower sampling for speed) ----
    best, best_score, parts = None, -1.0, {}
    w_edge, w_align, w_hom = 0.50, 0.20, 0.30

    for (cx, cy, a, b, th) in candidates:
        e, al, hm = ellipse_support_score(gray, gx, gy, mag, cx, cy, a, b, th, pts=96)
        score = w_edge * e + w_align * al + w_hom * hm

        # penalize ultra-skinny shapes (speckles/vessels)
        ecc = np.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
        if ecc > MAX_ECC_PENALTY:
            score *= 0.85

        if score > best_score:
            best_score = score
            best = (cx, cy, a, b, th)
            parts = {"edge": e, "align": al, "hom": hm}

    return best, {"best_score": float(best_score), "best_parts": parts, "engine": "opencv", "tested": len(candidates)}




# ---------- Shape selection + viz ----------
def choose_best_shape(img_bgr: np.ndarray, min_score: float = 0.16):
    # First: circle (fast)
    circ, dc = detect_best_circle(img_bgr)
    circ_score = dc.get("best_score", -1.0) if circ is not None else -1.0

    # EARLY EXIT: if circle already good, don’t try ellipse
    # (tune the margin; 0.08 works well)
    if circ is not None and circ_score >= (min_score + 0.08):
        shape = ("circle", circ)
        return shape, {"best_score": float(circ_score), "circle_score": float(circ_score), "ellipse_score": -1.0, "skipped_ellipse": True}

    # Otherwise: try ellipse (slower)
    ell, de = detect_best_ellipse(img_bgr)
    ell_score = de.get("best_score", -1.0) if ell is not None else -1.0

    if circ is None and ell is None:
        return None, {"reason": "no_shape"}

    # Pick best
    if ell_score >= circ_score:
        best = ("ellipse", ell); score = ell_score
    else:
        best = ("circle", circ); score = circ_score

    if score < float(min_score):
        return None, {"reason": "low_score", "circle_score": float(circ_score), "ellipse_score": float(ell_score)}

    return best, {"best_score": float(score), "circle_score": float(circ_score), "ellipse_score": float(ell_score)}

def overlay_shape(img: np.ndarray, shape: Optional[Tuple[Literal["circle","ellipse"], tuple]]) -> np.ndarray:
    vis = img.copy()
    if shape is None:
        cv2.putText(vis, "NO SHAPE FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3, cv2.LINE_AA)
        return vis
    kind, params = shape
    if kind == "circle":
        cx, cy, r = params
        cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (0,255,0), 3)
        cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (0,0,255), -1)
    else:
        cx, cy, a, b, th = params
        angle = np.rad2deg(th)
        axes  = (int(round(a)), int(round(b)))
        center= (int(round(cx)), int(round(cy)))
        cv2.ellipse(vis, center, axes, angle, 0, 360, (0,255,0), 3)
        cv2.circle(vis, center, 3, (0,0,255), -1)
    return vis

def save_panel(out_path: Path, original: np.ndarray, pre_gray: np.ndarray, mask: np.ndarray, vis: np.ndarray):
    if not HAS_PLT:
        cv2.imwrite(str(out_path), vis); return
    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 2, 1); ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)); ax1.set_title("Original"); ax1.axis("off")
    ax2 = plt.subplot(2, 2, 2); ax2.imshow(pre_gray, cmap="gray"); ax2.set_title("Preprocessed (CLAHE+denoise)"); ax2.axis("off")
    ax3 = plt.subplot(2, 2, 3); ax3.imshow(mask, cmap="gray"); ax3.set_title("Body mask (Otsu)"); ax3.axis("off")
    ax4 = plt.subplot(2, 2, 4); ax4.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax4.set_title("Best shape overlay"); ax4.axis("off")
    plt.tight_layout(); plt.savefig(str(out_path), dpi=150); plt.close()


# ---------- Batch ----------
def process_folder(in_dir: Path, out_dir: Path, recursive: bool = False, max_side: Optional[int] = None, min_score: float = 0.16):
    ensure_dir(out_dir)
    cand_paths = (in_dir.rglob("*") if recursive else in_dir.iterdir())
    cand_paths = [p for p in cand_paths if p.is_file()]

    # Skip our own outputs / teacher demo overlays and filter to image/DICOM only
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
            if max_side is not None and max_side > 0:
                H, W = img.shape[:2]
                scale = max(H, W) / float(max_side)
                if scale > 1:
                    img = cv2.resize(img, (int(W/scale), int(H/scale)), interpolation=cv2.INTER_AREA)

            gray = denoise_blur(clahe_gray(img))
            mask = make_body_mask(gray)
            shape, diag = choose_best_shape(img, min_score=min_score)
            vis = overlay_shape(img, shape)

            panel_path = out_dir / f"{img_path.stem}_panel.png"
            save_panel(panel_path, img, gray, mask, vis)

            if shape is None:
                no += 1
            else:
                ok += 1
        except Exception as e:
            print("ERROR:", e)
            err += 1

    total = ok + no + err
    print(f"\nSummary: {ok}/{total} OK ({100.0*ok/max(total,1):.1f}%) | NO_SHAPE: {no} | ERROR: {err}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Circle + Ellipse detector for heart center (herzinfarkt images).", add_help=True)
    ap.add_argument("--images", type=str, help="Folder with input images.")
    ap.add_argument("--out", type=str, help="Output folder for panels.")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    ap.add_argument("--max-side", type=int, default=1024, help="Downscale so the longest side is at most this many pixels.")
    ap.add_argument("--min-score", type=float, default=0.16, help="Minimum score to accept a shape; otherwise NO_SHAPE.")
    args = ap.parse_args()

    if not args.images:
        try: args.images = input("Enter path to your images: ").strip('"').strip()
        except EOFError: args.images = None
    if not args.out:
        try:
            default_out = "out_herzinfarkt"
            user_out = input(f"Enter output folder (default: {default_out}): ").strip('"').strip()
            args.out = user_out or default_out
        except EOFError:
            args.out = "out_herzinfarkt"

    if not args.images: raise SystemExit("No images folder provided.")
    in_dir = Path(args.images)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (in_dir.parent / out_dir).resolve()

    if not in_dir.exists(): raise SystemExit(f"Input dir does not exist: {in_dir}")
    ensure_dir(out_dir)

    process_folder(in_dir, out_dir, recursive=args.recursive, max_side=args.max_side, min_score=args.min_score)
    print(f"\nDone. Results saved in:\n  {out_dir}\n- Panels: {out_dir}/*.png")


if __name__ == "__main__":
    main()
