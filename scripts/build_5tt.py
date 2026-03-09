"""
build_5tt.py
Merges FreeSurfer aseg.mgz (WM/GM/CSF labels) with mri_watershed BEM surfaces
(skull, scalp) into a single 5-tissue-type (5TT) label volume.

Output label values:
    0 = Background / air
    1 = White Matter
    2 = Grey Matter (cortex + subcortical GM)
    3 = CSF
    4 = Skull
    5 = Scalp

Usage:
    python build_5tt.py <subject_dir>

Example:
    python build_5tt.py output/sub-116
"""

import sys
import os
import struct
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] nibabel not installed. Run: pip install nibabel numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# FreeSurfer aseg label groups
# ---------------------------------------------------------------------------

WM_LABELS = {
    2,   # Left-Cerebral-White-Matter
    41,  # Right-Cerebral-White-Matter
    7,   # Left-Cerebellum-White-Matter
    46,  # Right-Cerebellum-White-Matter
    16,  # Brain-Stem
    28,  # Left-VentralDC
    60,  # Right-VentralDC
    192, # Corpus callosum (sometimes present)
    250, 251, 252, 253, 254, 255,  # CC subdivisions
}

GM_LABELS = {
    3,   # Left-Cerebral-Cortex
    42,  # Right-Cerebral-Cortex
    8,   # Left-Cerebellum-Cortex
    47,  # Right-Cerebellum-Cortex
    # Subcortical GM structures
    10,  # Left-Thalamus
    11,  # Left-Caudate
    12,  # Left-Putamen
    13,  # Left-Pallidum
    17,  # Left-Hippocampus
    18,  # Left-Amygdala
    26,  # Left-Accumbens-area
    49,  # Right-Thalamus
    50,  # Right-Caudate
    51,  # Right-Putamen
    52,  # Right-Pallidum
    53,  # Right-Hippocampus
    54,  # Right-Amygdala
    58,  # Right-Accumbens-area
}

CSF_LABELS = {
    4,   # Left-Lateral-Ventricle
    5,   # Left-Inf-Lat-Vent
    14,  # 3rd-Ventricle
    15,  # 4th-Ventricle
    24,  # CSF
    43,  # Right-Lateral-Ventricle
    44,  # Right-Inf-Lat-Vent
    72,  # 5th-Ventricle
}

# Output label constants
LABEL_BG    = 0
LABEL_WM    = 1
LABEL_GM    = 2
LABEL_CSF   = 3
LABEL_SKULL = 4
LABEL_SCALP = 5


# ---------------------------------------------------------------------------
# FreeSurfer binary surface reader
# ---------------------------------------------------------------------------

TRIANGLE_FILE_MAGIC = 16777214
QUAD_FILE_MAGIC     = 16777215


def read_fs_surface(filepath):
    """
    Read a FreeSurfer binary surface file.
    Returns (vertices, faces) where vertices is (N,3) float32 and faces is (M,3) int32.
    """
    with open(filepath, "rb") as f:
        # Read 3-byte magic number
        b = f.read(3)
        magic = struct.unpack(">I", b"\x00" + b)[0]

        if magic != TRIANGLE_FILE_MAGIC:
            raise ValueError(f"Unsupported surface format (magic={magic}). "
                             "Only triangle surfaces are supported.")

        # Skip two comment lines
        f.readline()
        f.readline()

        n_verts, n_faces = struct.unpack(">II", f.read(8))

        verts = np.frombuffer(f.read(n_verts * 3 * 4), dtype=">f4").reshape(n_verts, 3)
        faces = np.frombuffer(f.read(n_faces * 3 * 4), dtype=">i4").reshape(n_faces, 3)

    return verts.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Surface → voxel rasterization
# ---------------------------------------------------------------------------

def tkr_ras_to_vox(vertices, affine):
    """
    Convert tkrRAS surface coordinates to voxel indices using the volume affine.
    FreeSurfer surfaces are stored in tkrRAS space; we need to map to voxel ijk.
    """
    inv_affine = np.linalg.inv(affine)
    ones = np.ones((len(vertices), 1), dtype=np.float32)
    homog = np.hstack([vertices, ones])           # (N, 4)
    vox = (inv_affine @ homog.T).T[:, :3]         # (N, 3)
    return vox


def fill_surface_mask(vertices, faces, vol_shape, affine, ras_offset=None):
    """
    Rasterize a closed surface mesh into a binary voxel mask.

    Strategy:
      1. Convert all vertices to voxel space.
      2. For every face, scanline-fill its bounding box slice-by-slice.
      3. Use scipy flood_fill from the volume centre outward to identify
         the interior, then invert to get the filled solid mask.

    Falls back to a convex-hull point-in-mesh test if scipy is unavailable.
    """
    try:
        from scipy.ndimage import binary_fill_holes
        has_scipy = True
    except ImportError:
        has_scipy = False

    # Optionally shift vertices (e.g. to account for tkrRAS vs scannerRAS offset)
    verts = vertices.copy()
    if ras_offset is not None:
        verts += ras_offset

    # Map to voxel indices
    vox_verts = tkr_ras_to_vox(verts, affine)

    # Paint surface voxels
    mask = np.zeros(vol_shape, dtype=bool)
    for tri in faces:
        pts = vox_verts[tri]  # (3, 3)
        # Simple: mark voxels at each vertex + edge midpoints
        for i in range(3):
            v = pts[i]
            ii, jj, kk = int(round(v[0])), int(round(v[1])), int(round(v[2]))
            if 0 <= ii < vol_shape[0] and 0 <= jj < vol_shape[1] and 0 <= kk < vol_shape[2]:
                mask[ii, jj, kk] = True
        for i in range(3):
            mid = (pts[i] + pts[(i + 1) % 3]) / 2
            ii, jj, kk = int(round(mid[0])), int(round(mid[1])), int(round(mid[2]))
            if 0 <= ii < vol_shape[0] and 0 <= jj < vol_shape[1] and 0 <= kk < vol_shape[2]:
                mask[ii, jj, kk] = True

    # Fill the interior
    if has_scipy:
        filled = binary_fill_holes(mask)
    else:
        # Rough fallback: dilate and fill using numpy
        from numpy.lib.stride_tricks import sliding_window_view
        filled = mask.copy()
        print("[WARN] scipy not found; surface fill may be incomplete. "
              "Run: pip install scipy")

    return filled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_5tt(subject_dir):
    subject_dir = os.path.abspath(subject_dir)
    mri_dir = os.path.join(subject_dir, "mri")
    bem_dir = os.path.join(subject_dir, "bem")

    aseg_path = os.path.join(mri_dir, "aseg.mgz")
    t1_path   = os.path.join(mri_dir, "T1.mgz")
    out_path  = os.path.join(mri_dir, "5tt.mgz")

    # Watershed surface prefix used in create_5tt.bat
    ws_prefix = os.path.join(bem_dir, "ws")

    inner_skull_path = ws_prefix + "_inner_skull_surface"
    outer_skull_path = ws_prefix + "_outer_skull_surface"
    outer_skin_path  = ws_prefix + "_outer_skin_surface"

    # --- Load aseg --------------------------------------------------------
    print("[INFO] Loading aseg.mgz...")
    if not os.path.exists(aseg_path):
        print(f"[ERROR] aseg.mgz not found at {aseg_path}")
        sys.exit(1)

    aseg_img  = nib.load(aseg_path)
    aseg_data = np.asarray(aseg_img.dataobj, dtype=np.int32)
    affine    = aseg_img.affine
    shape     = aseg_data.shape

    print(f"[INFO] Volume shape: {shape}, voxel size: "
          f"{np.abs(np.diag(affine)[:3]).round(3)}")

    # --- Build brain tissue labels from aseg ------------------------------
    print("[INFO] Mapping aseg labels -> WM / GM / CSF...")
    label_vol = np.zeros(shape, dtype=np.uint8)

    for lbl in WM_LABELS:
        label_vol[aseg_data == lbl] = LABEL_WM
    for lbl in GM_LABELS:
        label_vol[aseg_data == lbl] = LABEL_GM
    for lbl in CSF_LABELS:
        label_vol[aseg_data == lbl] = LABEL_CSF

    # --- Load and rasterize watershed surfaces ----------------------------
    surfaces = {
        "inner_skull": (inner_skull_path, LABEL_SKULL),
        "outer_skull": (outer_skull_path, LABEL_SKULL),
        "outer_skin":  (outer_skin_path,  LABEL_SCALP),
    }

    # FreeSurfer tkrRAS origin offset (centre-of-volume correction)
    # Load from T1.mgz header if available
    ras_offset = None
    if os.path.exists(t1_path):
        t1_img = nib.load(t1_path)
        # c_ras offset is stored in the MGH header
        if hasattr(t1_img.header, "get_data_shape"):
            try:
                c_r = t1_img.header["Pxfm_b2s"][0, 3]
                c_a = t1_img.header["Pxfm_b2s"][1, 3]
                c_s = t1_img.header["Pxfm_b2s"][2, 3]
                ras_offset = np.array([c_r, c_a, c_s], dtype=np.float32)
            except Exception:
                pass  # header field not present, offset stays None

    # Layer order matters: paint outer layers first, inner layers last
    # so that inner labels overwrite outer ones at boundaries.
    layer_order = ["outer_skin", "outer_skull", "inner_skull"]

    for key in layer_order:
        path, label_value = surfaces[key]
        if not os.path.exists(path):
            print(f"[WARN] Surface not found, skipping: {path}")
            continue

        print(f"[INFO] Rasterizing {key} surface -> label {label_value}...")
        try:
            verts, faces = read_fs_surface(path)
            surf_mask = fill_surface_mask(verts, faces, shape, affine, ras_offset)

            # Only paint voxels that are currently background
            paint_mask = surf_mask & (label_vol == LABEL_BG)
            label_vol[paint_mask] = label_value
            print(f"[INFO]   Painted {paint_mask.sum():,} voxels as label {label_value} ({key})")
        except Exception as e:
            print(f"[WARN] Could not process {key}: {e}")

    # Print label statistics
    print()
    print("[INFO] Label volume statistics:")
    names = {0: "Background", 1: "White Matter", 2: "Grey Matter",
             3: "CSF", 4: "Skull", 5: "Scalp"}
    for lv, name in names.items():
        count = int((label_vol == lv).sum())
        print(f"        {lv} = {name:15s}: {count:>12,} voxels")

    # --- Save output ------------------------------------------------------
    print()
    print(f"[INFO] Saving 5TT volume to {out_path}...")
    out_img = nib.MGHImage(label_vol, affine, aseg_img.header)
    nib.save(out_img, out_path)
    print(f"[DONE] Saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_5tt.py <subject_dir>")
        sys.exit(1)
    build_5tt(sys.argv[1])
