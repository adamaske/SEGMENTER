"""
visualize_5tt.py
Loads a FreeSurfer 5-tissue-type label volume (5tt.mgz) and displays
interactive orthogonal slice views + a 3D voxel overview.

Usage:
    python visualize_5tt.py <path_to_5tt.mgz>

Example:
    python visualize_5tt.py output/sub-116/mri/5tt.mgz

Dependencies:
    pip install nibabel numpy matplotlib scipy

Controls:
    - Slice viewers: click anywhere on a slice to move the crosshair
    - Sliders: drag to scroll through slices
    - 3D view: rotate with mouse, scroll to zoom
    - Opacity slider: adjust tissue transparency in 3D view
    - Checkboxes: toggle individual tissues on/off
"""

import sys
import os
import argparse
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] nibabel not installed. Run: pip install nibabel")
    sys.exit(1)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.widgets import Slider, CheckButtons, Button
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    print("[ERROR] matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.ndimage import zoom, binary_erosion
    HAS_SCIPY = True
except ImportError:
    print("[WARN] scipy not installed. 3D downsampling will use numpy fallback.")
    HAS_SCIPY = False

# ── Tissue definitions ────────────────────────────────────────────────────────

TISSUES = {
    0: {"name": "Background", "color": (0.05, 0.05, 0.05), "alpha": 0.0,  "show": False},
    1: {"name": "White Matter","color": (0.95, 0.95, 0.85), "alpha": 0.6,  "show": True},
    2: {"name": "Grey Matter", "color": (0.70, 0.50, 0.50), "alpha": 0.6,  "show": True},
    3: {"name": "CSF",         "color": (0.30, 0.55, 0.90), "alpha": 0.4,  "show": True},
    4: {"name": "Skull",       "color": (0.90, 0.82, 0.65), "alpha": 0.3,  "show": True},
    5: {"name": "Scalp",       "color": (0.92, 0.72, 0.60), "alpha": 0.2,  "show": True},
}

# Colourmap for 2D slice views: one distinct colour per label
SLICE_COLORS = np.array([
    [0.05, 0.05, 0.05, 1.0],   # 0 Background — near black
    [0.95, 0.95, 0.85, 1.0],   # 1 White Matter — off white
    [0.70, 0.50, 0.50, 1.0],   # 2 Grey Matter — dusty rose
    [0.30, 0.55, 0.90, 1.0],   # 3 CSF — blue
    [0.90, 0.82, 0.65, 1.0],   # 4 Skull — bone
    [0.92, 0.72, 0.60, 1.0],   # 5 Scalp — skin
], dtype=np.float32)

from matplotlib.colors import ListedColormap
SLICE_CMAP = ListedColormap(SLICE_COLORS)


# ── Marching cubes (pure numpy fallback) ─────────────────────────────────────

def extract_surface_marching_cubes(mask, step=2):
    """
    Use scipy's marching cubes if available, otherwise return a point cloud
    of surface voxels for 3D scatter plotting.
    """
    try:
        from scipy.ndimage import binary_erosion
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(mask.astype(np.float32), level=0.5, step_size=step)
        return verts, faces
    except ImportError:
        pass

    # Fallback: erode and find surface voxels
    eroded = binary_erosion(mask) if HAS_SCIPY else mask
    surface = mask & ~eroded
    coords = np.argwhere(surface)
    return coords, None


# ── Downsampling ──────────────────────────────────────────────────────────────

def downsample_vol(vol, target_size=64):
    """Downsample label volume to target_size^3 for 3D display using nearest-neighbour."""
    current = np.array(vol.shape)
    factor = target_size / current.max()
    if factor >= 1.0:
        return vol
    if HAS_SCIPY:
        return zoom(vol, factor, order=0).astype(np.uint8)
    # Numpy fallback: simple stride sampling
    s = max(1, int(1 / factor))
    return vol[::s, ::s, ::s].astype(np.uint8)


# ── Slice colouring ───────────────────────────────────────────────────────────

def label_slice_to_rgb(slice_2d):
    """Convert a 2D label slice to an RGB image using SLICE_COLORS."""
    clipped = np.clip(slice_2d, 0, len(SLICE_COLORS) - 1)
    return SLICE_COLORS[clipped, :3]


# ── Main viewer ───────────────────────────────────────────────────────────────

class Viewer5TT:
    def __init__(self, vol, affine, subject_name):
        self.vol = vol.astype(np.uint8)
        self.affine = affine
        self.subject = subject_name
        self.shape = vol.shape

        # Current crosshair position
        self.cx = self.shape[0] // 2
        self.cy = self.shape[1] // 2
        self.cz = self.shape[2] // 2

        # Tissue visibility (mutable copy)
        self.visible = {k: v["show"] for k, v in TISSUES.items()}

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(18, 10), facecolor="#1a1a2e")
        self.fig.canvas.manager.set_window_title(f"5TT Viewer — {self.subject}")

        # Grid: 3 slice panels | 1 3D panel | 1 controls panel
        gs = self.fig.add_gridspec(
            3, 5,
            left=0.04, right=0.98, top=0.93, bottom=0.08,
            wspace=0.35, hspace=0.4,
            width_ratios=[1, 1, 1, 1.4, 0.55]
        )

        ax_style = dict(facecolor="#0d0d1a")

        self.ax_ax  = self.fig.add_subplot(gs[0, 0], **ax_style)   # axial
        self.ax_cor = self.fig.add_subplot(gs[0, 1], **ax_style)   # coronal
        self.ax_sag = self.fig.add_subplot(gs[0, 2], **ax_style)   # sagittal
        self.ax_3d  = self.fig.add_subplot(gs[:, 3], projection="3d", facecolor="#0d0d1a")
        self.ax_ctrl= self.fig.add_subplot(gs[1:, 4])
        self.ax_ctrl.set_visible(False)

        # Slice sliders
        slider_kw = dict(color="#3a3a5c", track_color="#1a1a2e")
        self.sl_ax  = Slider(self.fig.add_axes([0.04, 0.04, 0.17, 0.018]),
                             "Axial Z",    0, self.shape[2]-1, valinit=self.cz,  valstep=1, **slider_kw)
        self.sl_cor = Slider(self.fig.add_axes([0.25, 0.04, 0.17, 0.018]),
                             "Coronal Y",  0, self.shape[1]-1, valinit=self.cy,  valstep=1, **slider_kw)
        self.sl_sag = Slider(self.fig.add_axes([0.46, 0.04, 0.17, 0.018]),
                             "Sagittal X", 0, self.shape[0]-1, valinit=self.cx,  valstep=1, **slider_kw)

        self.sl_ax.on_changed(self._on_slider)
        self.sl_cor.on_changed(self._on_slider)
        self.sl_sag.on_changed(self._on_slider)

        # Tissue checkboxes
        labels_vis = [f"  {TISSUES[i]['name']}" for i in range(1, 6)]
        check_ax = self.fig.add_axes([0.875, 0.25, 0.11, 0.35], facecolor="#0d0d1a")
        self.checks = CheckButtons(
            check_ax, labels_vis,
            actives=[self.visible[i] for i in range(1, 6)]
        )
        # Support both old (.rectangles) and new (.patches) matplotlib API
        _check_boxes = getattr(self.checks, "patches", None) or getattr(self.checks, "rectangles", [])
        for i, rect in enumerate(_check_boxes):
            rect.set_facecolor(TISSUES[i+1]["color"])
            rect.set_edgecolor("#ffffff")
        for txt in self.checks.labels:
            txt.set_color("#e0e0e0")
            txt.set_fontsize(8)
        self.checks.on_clicked(self._on_check)

        # Legend
        legend_ax = self.fig.add_axes([0.875, 0.62, 0.11, 0.28], facecolor="#0d0d1a")
        legend_ax.set_axis_off()
        patches = [
            mpatches.Patch(facecolor=TISSUES[i]["color"], edgecolor="#555", label=TISSUES[i]["name"])
            for i in range(1, 6)
        ]
        legend_ax.legend(handles=patches, loc="upper left", fontsize=7,
                         facecolor="#1a1a2e", edgecolor="#555",
                         labelcolor="#e0e0e0", framealpha=0.8)

        # Title
        self.fig.text(0.5, 0.97, f"5-Tissue Segmentation — {self.subject}",
                      ha="center", va="top", color="#b4befe", fontsize=13, fontweight="bold")

        # Stats text
        self._stats_ax = self.fig.add_axes([0.875, 0.06, 0.11, 0.17], facecolor="#0d0d1a")
        self._stats_ax.set_axis_off()
        self._draw_stats()

        # Connect click events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        # Initial draw
        self._draw_slices()
        self._draw_3d()

    # ── Stats panel ───────────────────────────────────────────────────────────

    def _draw_stats(self):
        self._stats_ax.cla()
        self._stats_ax.set_axis_off()
        total = self.vol.size
        lines = ["Voxel counts\n"]
        for i in range(1, 6):
            n = int((self.vol == i).sum())
            pct = 100 * n / total
            lines.append(f"{TISSUES[i]['name'][:10]}\n  {n:,} ({pct:.1f}%)")
        self._stats_ax.text(0.05, 0.98, "\n".join(lines),
                            transform=self._stats_ax.transAxes,
                            va="top", ha="left", fontsize=6.5, color="#a0a0c0",
                            fontfamily="monospace")

    # ── Slice drawing ─────────────────────────────────────────────────────────

    def _draw_slices(self):
        # Mask invisible labels
        vol = self.vol.copy()
        for label, vis in self.visible.items():
            if not vis:
                vol[vol == label] = 0

        axial   = label_slice_to_rgb(vol[:, :, self.cz].T)
        coronal = label_slice_to_rgb(vol[:, self.cy, :].T)
        sagittal= label_slice_to_rgb(vol[self.cx, :, :].T)

        for ax, img, title in [
            (self.ax_ax,  axial,    f"Axial  z={self.cz}"),
            (self.ax_cor, coronal,  f"Coronal  y={self.cy}"),
            (self.ax_sag, sagittal, f"Sagittal  x={self.cx}"),
        ]:
            ax.cla()
            ax.imshow(img, origin="lower", aspect="equal", interpolation="nearest")
            ax.set_title(title, color="#b4befe", fontsize=8, pad=3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5c")

        # Crosshairs
        def _cross(ax, h, v, hlim, vlim):
            ax.axhline(h, color="#f38ba8", linewidth=0.6, alpha=0.7)
            ax.axvline(v, color="#f38ba8", linewidth=0.6, alpha=0.7)

        _cross(self.ax_ax,  self.cy, self.cx, self.shape[1], self.shape[0])
        _cross(self.ax_cor, self.cz, self.cx, self.shape[2], self.shape[0])
        _cross(self.ax_sag, self.cz, self.cy, self.shape[2], self.shape[1])

        self.fig.canvas.draw_idle()

    # ── 3D drawing ────────────────────────────────────────────────────────────

    def _draw_3d(self):
        ax = self.ax_3d
        ax.cla()
        ax.set_facecolor("#0d0d1a")
        ax.set_xlabel("X", color="#666", fontsize=7)
        ax.set_ylabel("Y", color="#666", fontsize=7)
        ax.set_zlabel("Z", color="#666", fontsize=7)
        ax.tick_params(colors="#444", labelsize=6)
        ax.set_title("3D Overview", color="#b4befe", fontsize=9, pad=4)

        # Downsample for performance
        small = downsample_vol(self.vol, target_size=60)
        scale = np.array(self.shape) / np.array(small.shape)

        # Draw tissues outer → inner so transparency looks right
        for label in [5, 4, 3, 2, 1]:
            if not self.visible.get(label, True):
                continue
            t = TISSUES[label]
            mask = small == label
            if not mask.any():
                continue

            verts, faces = extract_surface_marching_cubes(mask, step=1)

            if faces is not None:
                # Full marching cubes mesh
                verts_scaled = verts * scale
                mesh = Poly3DCollection(
                    verts_scaled[faces],
                    alpha=t["alpha"],
                    linewidth=0,
                )
                mesh.set_facecolor(t["color"])
                mesh.set_edgecolor("none")
                ax.add_collection3d(mesh)
            else:
                # Scatter fallback
                coords = verts * scale
                # Subsample points for speed
                idx = np.random.choice(len(coords), min(len(coords), 2000), replace=False)
                ax.scatter(
                    coords[idx, 0], coords[idx, 1], coords[idx, 2],
                    c=[t["color"]], s=1, alpha=t["alpha"] * 2,
                    depthshade=True
                )

        sh = np.array(self.shape)
        ax.set_xlim(0, sh[0])
        ax.set_ylim(0, sh[1])
        ax.set_zlim(0, sh[2])
        ax.set_box_aspect([sh[0], sh[1], sh[2]])

        # Draw crosshair planes
        for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
            spine.pane.fill = False
            spine.pane.set_edgecolor("#222244")

        self.fig.canvas.draw_idle()

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_slider(self, val):
        self.cz = int(self.sl_ax.val)
        self.cy = int(self.sl_cor.val)
        self.cx = int(self.sl_sag.val)
        self._draw_slices()

    def _on_check(self, label):
        name = label.strip()
        for i, t in TISSUES.items():
            if t["name"] == name:
                self.visible[i] = not self.visible[i]
                break
        self._draw_slices()
        self._draw_3d()

    def _on_click(self, event):
        if event.inaxes == self.ax_ax:
            self.cx = int(np.clip(event.xdata, 0, self.shape[0]-1))
            self.cy = int(np.clip(event.ydata, 0, self.shape[1]-1))
            self.sl_sag.set_val(self.cx)
            self.sl_cor.set_val(self.cy)
        elif event.inaxes == self.ax_cor:
            self.cx = int(np.clip(event.xdata, 0, self.shape[0]-1))
            self.cz = int(np.clip(event.ydata, 0, self.shape[2]-1))
            self.sl_sag.set_val(self.cx)
            self.sl_ax.set_val(self.cz)
        elif event.inaxes == self.ax_sag:
            self.cy = int(np.clip(event.xdata, 0, self.shape[1]-1))
            self.cz = int(np.clip(event.ydata, 0, self.shape[2]-1))
            self.sl_cor.set_val(self.cy)
            self.sl_ax.set_val(self.cz)
        self._draw_slices()

    def show(self):
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise a FreeSurfer 5-tissue-type label volume (5tt.mgz)"
    )
    parser.add_argument(
        "path", nargs="?",
        help="Path to 5tt.mgz. If omitted, searches output/ for any 5tt.mgz."
    )
    args = parser.parse_args()

    path = args.path

    # Auto-discover if no path given
    if not path:
        for root, dirs, files in os.walk("output"):
            for f in files:
                if f == "5tt.mgz":
                    path = os.path.join(root, f)
                    print(f"[INFO] Found: {path}")
                    break
            if path:
                break

    if not path or not os.path.exists(path):
        print("[ERROR] 5tt.mgz not found.")
        print("Usage: python visualize_5tt.py <path/to/5tt.mgz>")
        sys.exit(1)

    print(f"[INFO] Loading {path} ...")
    img = nib.load(path)
    vol = np.asarray(img.dataobj, dtype=np.uint8)
    affine = img.affine

    print(f"[INFO] Volume shape: {vol.shape}")
    print(f"[INFO] Label distribution:")
    for i, t in TISSUES.items():
        n = int((vol == i).sum())
        print(f"        {i} = {t['name']:15s}: {n:>12,} voxels")

    # Try to get subject name from path
    parts = os.path.normpath(path).split(os.sep)
    subject = parts[-3] if len(parts) >= 3 else os.path.basename(os.path.dirname(path))

    print(f"\n[INFO] Opening viewer for subject: {subject}")
    print("[INFO] 3D view may take 10-30 seconds to render...")

    matplotlib.rcParams["figure.facecolor"] = "#1a1a2e"
    matplotlib.rcParams["text.color"] = "#e0e0e0"

    viewer = Viewer5TT(vol, affine, subject)
    viewer.show()


if __name__ == "__main__":
    main()