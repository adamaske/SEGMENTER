"""
downsample_5tt.py
Downsamples a 5-tissue-type label volume (5tt.mgz) to a lower resolution
and exports both a resampled .mgz and a raw binary .bin for MCX input.

Usage:
    python downsample_5tt.py <path_to_5tt.mgz> [--res MM] [--out DIR]

Examples:
    python downsample_5tt.py output/sub-116/mri/5tt.mgz
    python downsample_5tt.py output/sub-116/mri/5tt.mgz --res 2
    python downsample_5tt.py output/sub-116/mri/5tt.mgz --res 2 --out output/sub-116/mcx

Arguments:
    --res MM    Target isotropic resolution in mm (default: 2)
    --out DIR   Output directory (default: same directory as input file)

Output files:
    5tt_<MM>mm.mgz              Downsampled label volume
    5tt_<MM>mm.bin              Raw binary voxel file for MCX (uint8, Fortran order)
    5tt_<MM>mm_mcx_domain.json  MCX Domain JSON snippet

Dependencies:
    pip install nibabel numpy scipy
"""

import sys
import os
import json
import argparse
import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("[ERROR] nibabel not installed. Run: pip install nibabel")
    sys.exit(1)

try:
    from scipy.ndimage import zoom
except ImportError:
    print("[ERROR] scipy not installed. Run: pip install scipy")
    sys.exit(1)


TISSUES = {
    0: "Background",
    1: "White Matter",
    2: "Grey Matter",
    3: "CSF",
    4: "Skull",
    5: "Scalp",
}

# Default optical properties at 750nm (mua mm^-1, mus mm^-1, g, n)
MCX_OPTICAL_PROPS = {
    0: [0.000, 0.00, 1.00, 1.00],  # Background / air
    1: [0.019, 7.80, 0.89, 1.37],  # White Matter
    2: [0.020, 9.00, 0.89, 1.37],  # Grey Matter
    3: [0.004, 0.09, 0.89, 1.37],  # CSF
    4: [0.013, 8.60, 0.89, 1.37],  # Skull
    5: [0.017, 7.50, 0.89, 1.37],  # Scalp
}


def get_voxel_size_mm(affine):
    """
    Get isotropic voxel size from affine matrix using column norms.
    FreeSurfer affines are often rotation matrices — the diagonal alone
    is NOT the voxel size. Column norms give the correct voxel dimensions.
    """
    col_norms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    return float(col_norms.mean())


def update_affine_for_downsample(affine, factor):
    """
    Scale the spatial part of the affine to reflect larger voxels.
    Multiplies the rotation/scale columns by 1/factor (voxels are bigger).
    """
    new_affine = affine.copy().astype(np.float64)
    new_affine[:3, :3] /= factor
    return new_affine


def downsample_labels(vol, factor):
    """Nearest-neighbour downsample — preserves discrete label values."""
    if abs(factor - 1.0) < 1e-6:
        return vol.copy()
    return zoom(vol.astype(np.float32), factor, order=0).astype(np.uint8)


def build_mcx_json(vol_shape, bin_filename, res_mm):
    props = []
    for label in sorted(MCX_OPTICAL_PROPS.keys()):
        p = MCX_OPTICAL_PROPS[label]
        props.append({"mua": p[0], "mus": p[1], "g": p[2], "n": p[3]})
    return {
        "_comment": (
            f"MCX Domain section for 5TT volume at {res_mm}mm resolution. "
            "Adjust optical properties for your wavelength. "
            "Add Session, Optode, and Forward sections for a complete input file."
        ),
        "Domain": {
            "VolumeFile": bin_filename,
            "Dim": list(vol_shape),
            "VoxelSize": res_mm,
            "BackgroundFlag": 0,
            "Media": props,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Downsample 5tt.mgz and export for MCX Monte Carlo simulation"
    )
    parser.add_argument("path", help="Path to 5tt.mgz")
    parser.add_argument("--res", type=float, default=2.0,
                        help="Target resolution in mm (default: 2)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"[ERROR] File not found: {args.path}")
        sys.exit(1)

    out_dir = args.out if args.out else os.path.dirname(os.path.abspath(args.path))
    os.makedirs(out_dir, exist_ok=True)

    res_mm = args.res
    tag = f"{res_mm:.0f}mm" if res_mm == int(res_mm) else f"{res_mm}mm"

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Loading {args.path} ...")
    img = nib.load(args.path)
    vol = np.asarray(img.dataobj, dtype=np.uint8)
    affine = img.affine

    # Use column norms — correct for FreeSurfer rotated affines
    current_res = get_voxel_size_mm(affine)
    factor = current_res / res_mm

    print(f"[INFO] Input  resolution : {current_res:.4f} mm  (from affine column norms)")
    print(f"[INFO] Target resolution : {res_mm:.2f} mm")
    print(f"[INFO] Downsample factor : {factor:.6f}")
    print(f"[INFO] Input  shape      : {vol.shape}")

    # ── Downsample ────────────────────────────────────────────────────────────
    if abs(factor - 1.0) > 1e-4:
        direction = "Downsampling" if factor < 1.0 else "Upsampling"
        if factor > 1.0:
            print("[WARN] Target resolution is finer than input — upsampling adds no information.")
        else:
            print("[INFO] Downsampling (nearest-neighbour, label-preserving) ...")
        vol_ds = downsample_labels(vol, factor)
    else:
        print("[INFO] Resolution matches input — no resampling needed.")
        vol_ds = vol.copy()

    affine_ds = update_affine_for_downsample(affine, factor)
    print(f"[INFO] Output shape      : {vol_ds.shape}")

    # ── Label statistics ──────────────────────────────────────────────────────
    print()
    print("[INFO] Label distribution after downsampling:")
    total = vol_ds.size
    for label, name in TISSUES.items():
        n_before = int((vol == label).sum())
        n_after  = int((vol_ds == label).sum())
        pct = 100 * n_after / total
        ratio = n_after / n_before if n_before > 0 else 0.0
        print(f"        {label} = {name:15s}: {n_after:>10,} voxels "
              f"({pct:5.1f}%)  [was {n_before:,}, ratio {ratio:.3f}]")

    # ── Save .mgz ─────────────────────────────────────────────────────────────
    mgz_path = os.path.join(out_dir, f"5tt_{tag}.mgz")
    print(f"\n[INFO] Saving {mgz_path} ...")
    out_img = nib.MGHImage(vol_ds, affine_ds, img.header)
    nib.save(out_img, mgz_path)
    print(f"[OK]   Saved MGZ : {mgz_path}")

    # ── Save raw .bin for MCX ─────────────────────────────────────────────────
    # MCX reads volumes in Fortran (column-major) order by default (-a 0)
    bin_filename = f"5tt_{tag}.bin"
    bin_path = os.path.join(out_dir, bin_filename)
    print(f"[INFO] Saving {bin_path} ...")
    vol_ds.astype(np.uint8).flatten(order='F').tofile(bin_path)
    size_mb = os.path.getsize(bin_path) / 1024 / 1024
    print(f"[OK]   Saved BIN : {bin_path}  ({size_mb:.1f} MB)")

    # ── Save MCX JSON snippet ─────────────────────────────────────────────────
    json_path = os.path.join(out_dir, f"5tt_{tag}_mcx_domain.json")
    mcx_data = build_mcx_json(list(vol_ds.shape), bin_filename, res_mm)
    with open(json_path, "w") as f:
        json.dump(mcx_data, f, indent=2)
    print(f"[OK]   Saved MCX JSON snippet : {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    sx, sy, sz = vol_ds.shape
    speedup = int(round((1 / factor) ** 3)) if factor < 1 else 1
    print()
    print("=" * 60)
    print(f"  Resolution : {res_mm} mm isotropic")
    print(f"  Dimensions : {sx} x {sy} x {sz} voxels")
    print(f"  FOV        : {sx*res_mm:.0f} x {sy*res_mm:.0f} x {sz*res_mm:.0f} mm")
    if speedup > 1:
        print(f"  Speedup vs {current_res:.0f}mm : ~{speedup}x fewer voxels")
    print()
    print("  Output files:")
    print(f"    {os.path.basename(mgz_path):<35} Downsampled label volume")
    print(f"    {bin_filename:<35} MCX raw binary (Fortran order)")
    print(f"    {os.path.basename(json_path):<35} MCX Domain JSON snippet")
    print("=" * 60)
    print()
    print("[INFO] To run with MCX:")
    print(f'         mcx -f simulation.json')
    print(f'  where simulation.json includes:')
    print(f'         "VolumeFile": "{bin_filename}"')
    print(f'         "Dim": {list(vol_ds.shape)}')
    print(f'         "VoxelSize": {res_mm}')


if __name__ == "__main__":
    main()
