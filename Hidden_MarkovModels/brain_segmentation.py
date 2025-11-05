import argparse
import os
import glob
import numpy as np
import pydicom
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes, label as cc_label
from sklearn.cluster import KMeans
from skimage import morphology, filters
import matplotlib.pyplot as plt

try:
    import nibabel as nib
    HAVE_NIB = True
except Exception:
    HAVE_NIB = False

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)  
except Exception:
    pass


# IO + Preprocessing
def load_dicom_series(dicom_dir):
    files = []
    for ext in ("*.dcm", "*"):
        files.extend(glob.glob(os.path.join(dicom_dir, ext)))
    ds_list = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=False)
            if hasattr(ds, "PixelData"):
                ds_list.append(ds)
        except Exception:
            pass
    if not ds_list:
        raise RuntimeError(f"No readable DICOM slices in {dicom_dir}")

    # Sort slices by ImagePositionPatient (z) or InstanceNumber
    def slice_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        inst = getattr(ds, "InstanceNumber", None)
        return float(inst) if inst is not None else 0.0

    ds_list.sort(key=slice_key)

    # Stack to 3D volume (Z, Y, X)
    vol = np.stack([ds.pixel_array.astype(np.float32) for ds in ds_list], axis=0)
    # Rescale if needed
    for i, ds in enumerate(ds_list):
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        vol[i] = vol[i] * slope + inter
    # Spacing (z, y, x)
    try:
        px = tuple(map(float, ds_list[0].PixelSpacing))
        sx = float(getattr(ds_list[0], "SliceThickness", 1.0))
        spacing = (sx, px[1], px[0])
    except Exception:
        spacing = (1.0, 1.0, 1.0)
    return vol, spacing


def robust_normalize(vol, p_low=2.0, p_high=98.0, eps=1e-6):
    lo = np.percentile(vol, p_low)
    hi = np.percentile(vol, p_high)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / max(hi - lo, eps)
    return vol


def bias_field_correction(vol, sigma=20.0, eps=1e-6):
    """Simple homomorphic correction: divide out a large-scale Gaussian blur."""
    smooth = gaussian_filter(vol, sigma=sigma, mode="nearest")
    smooth = np.clip(smooth, eps, None)
    corrected = vol / smooth
    corrected = corrected / (np.percentile(corrected, 99.0) + eps)
    return corrected


def skull_strip(vol_corr):
    """Create a brain mask from bias-corrected volume."""
    thresh = filters.threshold_otsu(vol_corr[vol_corr > 0])
    mask = vol_corr > max(thresh, 0.05)
    lbl, n = cc_label(mask)
    if n > 0:
        sizes = [(lbl == i).sum() for i in range(1, n + 1)]
        keep = 1 + int(np.argmax(sizes))
        mask = lbl == keep
    mask = binary_opening(mask, structure=morphology.ball(2))
    mask = binary_closing(mask, structure=morphology.ball(2))
    mask = binary_fill_holes(mask)
    return mask


# HMRF / EM helpers
def init_gmm_params(vol, mask, K):
    data = vol[mask].reshape(-1, 1)
    if data.shape[0] < K:
        raise RuntimeError("Mask too small for initialization")
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(data)
    labels = np.zeros_like(vol, dtype=np.int32)
    labels[mask] = km.labels_
    means = np.array([data[km.labels_ == k].mean() for k in range(K)], dtype=np.float64)
    vars_ = np.array([data[km.labels_ == k].var() + 1e-6 for k in range(K)], dtype=np.float64)
    order = np.argsort(means)
    means = means[order]
    vars_ = vars_[order]
    remap = np.zeros(K, dtype=np.int32)
    for new, old in enumerate(order):
        remap[old] = new
    labels[mask] = remap[km.labels_]
    return labels, means, vars_


def gaussian_nll(y, mean, var, eps=1e-6):
    """Negative log-likelihood for a Gaussian (ignoring constants)."""
    return 0.5 * np.log(var + eps) + 0.5 * ((y - mean) ** 2) / (var + eps)


def icm_update_labels(
    vol, mask, means, vars_, beta,
    inplane_only=False, sweeps=1,
    add_diagonals=False, diag_weight=0.5
):
   
    K = len(means)
    Z, Y, X = vol.shape
    data_terms = np.stack([gaussian_nll(vol, m, v) for m, v in zip(means, vars_)], axis=0)  # (K,Z,Y,X)
    labels = np.argmin(data_terms, axis=0).astype(np.int32)

    if inplane_only:
        neigh = [(0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]  # 4-neigh in plane
        diag  = [(0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)] if add_diagonals else []
    else:
        neigh = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]  # 6-neigh in 3D
        diag = []

    for _ in range(max(1, sweeps)):
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    if not mask[z, y, x]:
                        continue
                    # Count neighbor labels
                    counts = np.zeros(K, dtype=np.int32)
                    for dz, dy, dx in neigh:
                        zz, yy, xx = z + dz, y + dy, x + dx
                        if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X and mask[zz, yy, xx]:
                            counts[labels[zz, yy, xx]] += 1
                    total_n = counts.sum()

                    # Optional diagonals (smaller weight)
                    if diag:
                        dcounts = np.zeros(K, dtype=np.int32)
                        for dz, dy, dx in diag:
                            zz, yy, xx = z + dz, y + dy, x + dx
                            if 0 <= zz < Z and 0 <= yy < Y and 0 <= xx < X and mask[zz, yy, xx]:
                                dcounts[labels[zz, yy, xx]] += 1
                        dtotal = dcounts.sum()
                        E = data_terms[:, z, y, x] \
                            + beta * (total_n - counts.astype(np.float64)) \
                            + (diag_weight * beta) * (dtotal - dcounts.astype(np.float64))
                    else:
                        E = data_terms[:, z, y, x] + beta * (total_n - counts.astype(np.float64))

                    labels[z, y, x] = int(np.argmin(E))
    return labels


def m_step_update(vol, labels, mask, K):
    """Re-estimate Gaussian means/vars from current labels, keep class order stable."""
    means = np.zeros(K, dtype=np.float64)
    vars_ = np.zeros(K, dtype=np.float64)
    for k in range(K):
        vox = vol[(labels == k) & mask]
        if vox.size < 10:
            means[k] = float(np.median(vol[mask]))
            vars_[k] = float(np.var(vol[mask]) + 1e-6)
        else:
            means[k] = float(vox.mean())
            vars_[k] = float(vox.var() + 1e-6)
    order = np.argsort(means)
    means = means[order]
    vars_ = vars_[order]
    remap = np.zeros(K, dtype=np.int32)
    for new, old in enumerate(order):
        remap[old] = new
    labels_remapped = remap[labels]
    return labels_remapped, means, vars_


# Visualization utilities
def plot_hmrf_emissions(means, vars_, out_png):
    """Plot the Gaussian emission shapes for documentation."""
    x = np.linspace(0, 1, 400)
    plt.figure(figsize=(6, 4))
    for k, (m, v) in enumerate(zip(means, vars_)):
        y = np.exp(-0.5 * ((x - m) ** 2) / (v + 1e-9))  
        plt.plot(x, y, label=f"class {k} (μ={m:.3f}, σ²={v:.4g})")
    plt.xlabel("intensity (normalized)")
    plt.ylabel("emission (unnormalized)")
    plt.title("Gaussian emissions p(y|z=k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def posterior_slice_2d(vol_slice, labels_slice, means, vars_, mask_slice, beta=1.2):

    H, W = vol_slice.shape
    K = len(means)
    emis = np.stack([
        -0.5 * np.log(vars_[k] + 1e-9) - 0.5 * ((vol_slice - means[k]) ** 2) / (vars_[k] + 1e-9)
        for k in range(K)
    ], axis=0)  

    prior = np.zeros_like(emis)
    neigh2d = [(1,0), (-1,0), (0,1), (0,-1)]
    for y in range(H):
        for x in range(W):
            if not mask_slice[y, x]:
                continue
            neigh = []
            for dy, dx in neigh2d:
                yy, xx = y + dy, x + dx
                if 0 <= yy < H and 0 <= xx < W and mask_slice[yy, xx]:
                    neigh.append(labels_slice[yy, xx])
            if not neigh:
                continue
            neigh = np.array(neigh)
            for k in range(K):
                disagree = int((neigh != k).sum())
                prior[k, y, x] = -beta * disagree

    log_post = emis + prior
    m = log_post.max(axis=0, keepdims=True)
    post = np.exp(log_post - m)
    post /= (post.sum(axis=0, keepdims=True) + 1e-12)
    return post  


def save_qa_figures(vol, labels, mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mid = vol.shape[0] // 2

    plt.figure()
    plt.imshow(vol[mid], cmap="gray")
    plt.title("Bias-corrected, normalized - mid slice")
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "vol_mid.png"), bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(labels[mid], cmap="tab20", interpolation="nearest")
    plt.title("Segmentation labels - mid slice")
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "labels_mid.png"), bbox_inches="tight", dpi=150)
    plt.close()

    plt.figure()
    plt.imshow(mask[mid].astype(np.uint8), cmap="gray")
    plt.title("Brain mask - mid slice")
    plt.axis("off")
    plt.savefig(os.path.join(out_dir, "mask_mid.png"), bbox_inches="tight", dpi=150)
    plt.close()


# HMRF-EM (wrapped for clarity)
def hmrf_em_segmentation(
    vol, mask, K=3, max_iters=12, beta=1.2,
    inplane_only=False, icm_sweeps=1, stop_ratio=0.01,
    add_diagonals=False, diag_weight=0.5,
    qa_dir=None, viz_emissions=False
):
    """
    HMRF-EM segmentation with explicit components:
      - Hidden states: tissue classes {0..K-1}
      - Observations: voxel intensities (preprocessed)
      - Emissions p(y|z=k): Gaussian N(μ_k, σ_k^2), learned in M-step
      - Spatial prior p(z): Potts model; penalty β per neighbor disagreement
    Optimization: K-means init → [ICM (MAP) + M-step] × max_iters
    """
    labels, means, vars_ = init_gmm_params(vol, mask, K)
    last_labels = labels.copy()
    for it in range(max_iters):
        print(f"  Iteration {it+1}/{max_iters}")
        labels = icm_update_labels(
            vol, mask, means, vars_,
            beta=beta,
            inplane_only=inplane_only,
            sweeps=icm_sweeps,
            add_diagonals=add_diagonals,
            diag_weight=diag_weight
        )
        labels, means, vars_ = m_step_update(vol, labels, mask, K)
        print(f"    means = {np.round(means, 4)}, vars = {np.round(vars_, 6)}")

        if viz_emissions and qa_dir:
            os.makedirs(qa_dir, exist_ok=True)
            plot_hmrf_emissions(means, vars_, os.path.join(qa_dir, f"emissions_iter_{it+1}.png"))

        changed = (labels != last_labels).sum()
        print(f"    voxels changed = {changed}")
        total = int(mask.sum())
        if total > 0 and (changed / total) < stop_ratio:
            print(f"    Early stop: change ratio {changed/total:.4f} < {stop_ratio}")
            break
        if changed == 0:
            print("    Converged (no label change).")
            break
        last_labels = labels.copy()

    return labels, means, vars_


def main():
    import sys
    ap = argparse.ArgumentParser(description="HMRF-EM brain MRI segmentation (CSF/GM/WM) on a DICOM series.")
    ap.add_argument("--dicom_dir", default=None, help="Path to folder with DICOM slices (one 3D series).")
    ap.add_argument("--out_dir", default="./outputs", help="Output directory.")
    ap.add_argument("--classes", type=int, default=3, help="Number of tissue classes (default 3).")
    ap.add_argument("--iters", type=int, default=12, help="Number of EM/ICM iterations.")
    ap.add_argument("--beta", type=float, default=1.2, help="Potts prior strength (higher = smoother).")
    ap.add_argument("--nlm", action="store_true", help="Use Non-Local Means denoising (slower).")
    ap.add_argument("--gaussian_sigma", type=float, default=0.8, help="Gaussian sigma if not using NLM.")
    ap.add_argument("--save_qa", action="store_true", help="Save QA PNGs for quick visual checks.")
    ap.add_argument("--mask_npy", type=str, default=None, help="Optional path to a precomputed brain mask .npy")
    ap.add_argument("--export_nifti", action="store_true", help="Export NIfTI if nibabel is installed.")
    ap.add_argument("--gui", action="store_true", help="Open a folder picker to select the DICOM directory.")
    ap.add_argument("--inplane_only", action="store_true", help="Ignore z-neighbors (faster).")
    ap.add_argument("--icm_sweeps", type=int, default=1, help="ICM Gauss-Seidel passes (default 1).")
    ap.add_argument("--stop_ratio", type=float, default=0.01, help="Early stop if label-change ratio < this (default 1%).")
    ap.add_argument("--add_diagonals", action="store_true", help="Include in-plane diagonal neighbors (slight weight).")
    ap.add_argument("--diag_weight", type=float, default=0.5, help="Weight for diagonal neighbors relative to beta.")
    ap.add_argument("--viz_emissions", action="store_true", help="Save Gaussian emission plots per iteration.")
    ap.add_argument("--posterior_png", action="store_true", help="Save posterior heatmaps (mid slice).")
    ap.add_argument("--baseline_no_mrf", action="store_true", help="Also save a β=0 baseline result for comparison.")
    args = ap.parse_args()

    if not args.dicom_dir:
        if args.gui:
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                chosen = filedialog.askdirectory(title="Select DICOM series folder")
                if chosen:
                    args.dicom_dir = chosen
                else:
                    print("No folder selected. Exiting.")
                    sys.exit(1)
            except Exception as e:
                print("GUI folder picker unavailable:", e)
                print("Please run again with --dicom_dir <path> or without --gui to use text prompt.")
                sys.exit(1)
        else:
            try:
                args.dicom_dir = input("Enter path to DICOM series folder (e.g., ./MR_brain_1): ").strip('"').strip()
            except EOFError:
                print("No input received. Exiting.")
                sys.exit(1)

    if not os.path.isdir(args.dicom_dir):
        print(f"Provided path does not exist or is not a directory: {args.dicom_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    qa_dir = os.path.join(args.out_dir, "qa")

    print("[1/6] Loading DICOM series...")
    vol, spacing = load_dicom_series(args.dicom_dir)
    print(f"  Volume shape: {vol.shape}, spacing (z,y,x): {spacing}")

    print("[2/6] Intensity normalization...")
    vol = robust_normalize(vol)

    print("[3/6] Denoising...")
    if args.nlm:
        from skimage.restoration import denoise_nl_means, estimate_sigma
        denoised = np.zeros_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            sigma_est = np.mean(estimate_sigma(vol[z], channel_axis=None))
            denoised[z] = denoise_nl_means(
                vol[z], h=0.8 * sigma_est, fast_mode=True,
                patch_size=5, patch_distance=6, channel_axis=None
            ).astype(np.float32)
        vol = denoised
    else:
        vol = gaussian_filter(vol, sigma=args.gaussian_sigma)

    vol = vol[:, ::2, ::2]
    spacing = (spacing[0], spacing[1] * 2, spacing[2] * 2)

    print("[4/6] Bias-field correction and skull stripping...")
    vol_corr = bias_field_correction(vol, sigma=20.0)
    if args.mask_npy and os.path.exists(args.mask_npy):
        mask = np.load(args.mask_npy).astype(bool)
        auto_mask = skull_strip(vol_corr)
        mask = mask & auto_mask
    else:
        mask = skull_strip(vol_corr)

    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = morphology.remove_small_holes(mask, area_threshold=500)

    print("[5/6] Initialize GMM parameters...")
    K = int(args.classes)
    labels, means, vars_ = init_gmm_params(vol_corr, mask, K)
    print(f"  Initial means: {means}")

    print("[6/6] HMRF-EM iterations (β = {args.beta})...")
    labels, means, vars_ = hmrf_em_segmentation(
        vol_corr, mask, K=K, max_iters=args.iters, beta=args.beta,
        inplane_only=args.inplane_only, icm_sweeps=args.icm_sweeps,
        stop_ratio=args.stop_ratio, add_diagonals=args.add_diagonals,
        diag_weight=args.diag_weight, qa_dir=qa_dir, viz_emissions=args.viz_emissions
    )

    np.save(os.path.join(args.out_dir, "segmentation_labels.npy"), labels.astype(np.int32))
    np.save(os.path.join(args.out_dir, "class_means.npy"), means.astype(np.float64))
    np.save(os.path.join(args.out_dir, "class_vars.npy"), vars_.astype(np.float64))
    np.save(os.path.join(args.out_dir, "brain_mask.npy"), mask.astype(np.uint8))

    if args.save_qa:
        save_qa_figures(vol_corr, labels, mask, qa_dir)

    if args.export_nifti and HAVE_NIB:
        affine = np.diag([spacing[2], spacing[1], spacing[0], 1.0])
        img = nib.Nifti1Image(labels.astype(np.int16), affine=affine)
        nib.save(img, os.path.join(args.out_dir, "segmentation_labels.nii.gz"))
        msk = nib.Nifti1Image(mask.astype(np.uint8), affine=affine)
        nib.save(msk, os.path.join(args.out_dir, "brain_mask.nii.gz"))

    if args.posterior_png:
        mid = vol_corr.shape[0] // 2
        post = posterior_slice_2d(vol_corr[mid], labels[mid], means, vars_, mask[mid], beta=args.beta)
        os.makedirs(qa_dir, exist_ok=True)
        for k in range(K):
            plt.figure(figsize=(4, 4))
            plt.imshow(post[k], cmap="magma")
            plt.axis("off")
            plt.title(f"Posterior p(z={k}|y), slice mid")
            plt.tight_layout()
            plt.savefig(os.path.join(qa_dir, f"posterior_k{k}_mid.png"), dpi=150)
            plt.close()

    if args.baseline_no_mrf:
        print("[Baseline] Running β = 0 (no spatial prior) for comparison...")
        labels0, means0, vars0 = hmrf_em_segmentation(
            vol_corr, mask, K=K, max_iters=min(4, args.iters), beta=0.0,
            inplane_only=True, icm_sweeps=1, stop_ratio=0.01,
            add_diagonals=False, qa_dir=None, viz_emissions=False
        )
        np.save(os.path.join(args.out_dir, "segmentation_labels_beta0.npy"), labels0.astype(np.int32))
        if args.save_qa:
            os.makedirs(qa_dir, exist_ok=True)
            plt.figure()
            plt.imshow(labels0[mid], cmap="tab20", interpolation="nearest")
            plt.axis("off")
            plt.title("Labels mid (β=0 baseline)")
            plt.savefig(os.path.join(qa_dir, "labels_mid_beta0.png"), dpi=150, bbox_inches="tight")
            plt.close()

    print("Done. Outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()
