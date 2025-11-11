import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---- config ----
OUT_DIR = "./outputs_best"  

# ---- load outputs ----
labels_path = os.path.join(OUT_DIR, "segmentation_labels.npy")
mask_path   = os.path.join(OUT_DIR, "brain_mask.npy")
means_path  = os.path.join(OUT_DIR, "class_means.npy")

L = np.load(labels_path)                     # (Z, Y, X) int labels
M = np.load(mask_path).astype(bool)          # (Z, Y, X) brain mask
Z = L.shape[0]

if os.path.exists(means_path):
    means = np.load(means_path)             
    order = np.argsort(means)                
    inv = np.zeros_like(order)
    inv[order] = np.arange(order.size)
    L_disp = inv[L]                          
else:
    L_disp = L.copy()
    means = None

colors = np.array([
    [0.90, 0.20, 0.20, 1.0],  # WM
    [0.20, 0.70, 0.25, 1.0],  # GM
    [0.20, 0.40, 0.90, 1.0],  # CSF
    [0.00, 0.00, 0.00, 1.0],  # background
], dtype=float)
cmap_labels = ListedColormap(np.vstack([colors[:3], colors[3]]))  

# Start on the mid slice
z = Z // 2

# Prepare first frame (mask background to index 3)
def compose_slice(z_idx):
    s = L_disp[z_idx].copy()
    s[~M[z_idx]] = 3  # background index
    return s

# Set up figure
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(compose_slice(z), cmap=cmap_labels, vmin=0, vmax=3, interpolation="nearest")
ax.axis("off")

def title_text():
    base = f"Segmentation (WM=red, GM=green, CSF=blue) — slice {z+1}/{Z}"
    if means is not None:
        sorted_means = np.sort(means)
        base += f"\nclass means (sorted): {np.round(sorted_means, 4)}"
    return base

ax.set_title(title_text(), fontsize=10)

# Keyboard navigation
def redraw():
    im.set_data(compose_slice(z))
    ax.set_title(title_text(), fontsize=10)
    fig.canvas.draw_idle()

def on_key(e):
    global z
    if e.key in ("right", "down"):
        z = min(z + 1, Z - 1)
        redraw()
    elif e.key in ("left", "up"):
        z = max(z - 1, 0)
        redraw()
    elif e.key == "home":
        z = 0
        redraw()
    elif e.key == "end":
        z = Z - 1
        redraw()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
