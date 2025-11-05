import numpy as np, matplotlib.pyplot as plt

L = np.load("./outputs/segmentation_labels.npy")
M = np.load("./outputs/brain_mask.npy").astype(bool)
z = L.shape[0] // 2

fig, ax = plt.subplots()
im = ax.imshow(np.where(M[z], L[z], -1), cmap="tab20", interpolation="nearest")
ax.set_title(f"slice {z+1}/{L.shape[0]}")
ax.axis("off")

def redraw():
    im.set_data(np.where(M[z], L[z], -1))
    ax.set_title(f"slice {z+1}/{L.shape[0]}")
    fig.canvas.draw_idle()

def on_key(e):
    global z
    if e.key in ("right","down"):
        z = min(z+1, L.shape[0]-1); redraw()
    elif e.key in ("left","up"):
        z = max(z-1, 0); redraw()

fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
