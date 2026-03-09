# NVIZ Segmentation

Generate cortex, white matter, CSF, skull, and scalp meshes from T1-weighted MRI scans using [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) via Docker. Designed for import into [NIRSViz](https://github.com/adamaske/NIRSViz).

```
T1w MRI (.nii.gz)  →  recon-all (Docker)  →  5-tissue label volume  →  NIRSViz
```

---

## Prerequisites

### 1. Docker Desktop
Download and install from [docker.com](https://www.docker.com/products/docker-desktop/). Enable the **WSL2 backend** (Settings → General → Use WSL2). After installing, open Docker Desktop and wait for it to fully start.

Pull the FreeSurfer image once (≈15 GB):
```
docker pull freesurfer/freesurfer:7.4.1
```

### 2. FreeSurfer License (free)
FreeSurfer requires a license file to run.

1. Register at: https://surfer.nmr.mgh.harvard.edu/registration.html
2. You will receive `license.txt` by email
3. Place it at `license\license.txt` in this folder

### 3. Python 3 + packages
```
pip install nibabel numpy scipy
```

---

## Quick Start

```
1. Place your T1w scan (.nii.gz) in the input\ folder
2. Place your FreeSurfer license.txt in the license\ folder
3. Run:  run_recon_all.bat sub-116_T1w.nii.gz sub-116
4. Wait ~5-7 hours
5. Run:  create_5tt.bat sub-116
6. Find your 5-tissue label volume at output\sub-116\mri\5tt.mgz
```

---

## Scripts

### `run_recon_all.bat` — Step 1: Full cortical reconstruction

Runs the complete FreeSurfer `recon-all` pipeline on a T1-weighted MRI scan.

**Usage:**
```
run_recon_all.bat <filename.nii.gz> <subject_id>
```

**Example:**
```
run_recon_all.bat sub-116_T1w.nii.gz sub-116
```

**Configuration:** Open the file and set `NUM_CORES` at the top to the number of CPU cores you want to use (recommended: your total cores minus 2).

**What it does:**
- Skull strips, intensity normalises, and registers the T1 scan to MNI space
- Runs deep white matter and subcortical segmentation (`aseg.mgz`)
- Reconstructs pial and white matter surfaces for both hemispheres
- Uses `-parallel -openmp N` to process left and right hemispheres concurrently

**Runtime:** ~5–7 hours with parallelism (vs 6–12 hours single-core). Surface reconstruction is CPU-bound and cannot be GPU-accelerated.

**Output:** `output\<subject_id>\` — see Output Structure below.

---

### `create_5tt.bat` — Step 2: 5-tissue label volume

Generates a single merged volumetric label file covering all five tissue types needed for forward modelling and voxelisation. Runs two sub-steps internally.

**Usage:**
```
create_5tt.bat <subject_id>
```

**Example:**
```
create_5tt.bat sub-116
```

**What it does:**

**Sub-step 1 — Watershed BEM surfaces (Docker / FreeSurfer)**

Runs `mri_watershed -surf` on `T1.mgz` to generate four nested boundary surfaces into `output\<subject_id>\bem\`:

| File | Description |
|---|---|
| `ws_brain_surface` | Brain / CSF boundary |
| `ws_inner_skull_surface` | Inner table of the skull |
| `ws_outer_skull_surface` | Outer table of the skull |
| `ws_outer_skin_surface` | Scalp / air boundary |

These surfaces define the skull and scalp compartments that are not present in `aseg.mgz`.

**Sub-step 2 — Merge into label volume (Python)**

`scripts\build_5tt.py` loads `aseg.mgz`, maps FreeSurfer integer labels to tissue classes, rasterises the watershed surfaces into the same voxel grid using `scipy.ndimage.binary_fill_holes`, and writes the merged result.

Tissue layers are painted in order from outside in — scalp first, then skull, then the brain compartments from `aseg` — so that inner structures always take priority at boundaries.

**Output:** `output\<subject_id>\mri\5tt.mgz`

---

## 5TT Label Volume

The output `5tt.mgz` is a 3D integer volume in the same voxel space as `aseg.mgz` (FreeSurfer conformed space, 1 mm isotropic). Every voxel is assigned one of the following integer labels:

| Label | Tissue | Source |
|---|---|---|
| `0` | Background / air | — |
| `1` | White Matter | `aseg.mgz` labels 2, 41, 7, 46, 16, 28, 60 |
| `2` | Grey Matter | `aseg.mgz` labels 3, 42, 8, 47, 10–13, 17–18, 26, 49–54, 58 |
| `3` | CSF | `aseg.mgz` labels 4, 5, 14, 15, 24, 43, 44, 72 |
| `4` | Skull | `ws_inner_skull_surface` → `ws_outer_skull_surface` shell |
| `5` | Scalp | `ws_outer_skull_surface` → `ws_outer_skin_surface` shell |

**White Matter (1)** includes cerebral and cerebellar white matter, brain stem, and ventral DC. Corpus callosum subdivisions are included where labelled.

**Grey Matter (2)** includes cortical grey matter for both hemispheres and cerebellum, plus subcortical grey matter structures (thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens).

**CSF (3)** includes the lateral ventricles, third and fourth ventricles, and the general CSF compartment (label 24). This represents intracranial CSF only — not the subarachnoid space, which is implicitly background between the brain and skull.

**Skull (4)** is the voxelised shell between the inner skull and outer skull watershed surfaces.

**Scalp (5)** is the voxelised shell between the outer skull and outer skin watershed surfaces. This includes skin, subcutaneous fat, and muscle.

### Loading in Python
```python
import nibabel as nib
import numpy as np

img = nib.load("output/sub-116/mri/5tt.mgz")
vol = np.asarray(img.dataobj, dtype=np.uint8)

wm_mask    = vol == 1
gm_mask    = vol == 2
csf_mask   = vol == 3
skull_mask = vol == 4
scalp_mask = vol == 5
```

### Loading in MATLAB
```matlab
vol = MRIread('output/sub-116/mri/5tt.mgz');
labels = vol.vol;   % 3D array of uint8 label values
wm = labels == 1;
gm = labels == 2;
```

---

## Output Structure

```
output/<subject_id>/
├── mri/
│   ├── T1.mgz                       # Intensity-normalised T1 in FreeSurfer space
│   ├── orig.mgz                     # Original T1w converted to MGZ
│   ├── aseg.mgz                     # Subcortical + cortical segmentation (95 labels)
│   ├── aparc+aseg.mgz               # Full cortical parcellation (Desikan-Killiany)
│   ├── brainmask.mgz                # Skull-stripped brain mask
│   ├── wm.mgz                       # White matter binary mask
│   └── 5tt.mgz                      # ← 5-tissue label volume (created by create_5tt.bat)
├── surf/
│   ├── lh.pial / rh.pial            # Pial surfaces (FreeSurfer binary format)
│   └── lh.white / rh.white          # White matter surfaces
├── bem/
│   ├── ws_brain_surface             # Brain boundary surface
│   ├── ws_inner_skull_surface       # Inner skull surface
│   ├── ws_outer_skull_surface       # Outer skull surface
│   └── ws_outer_skin_surface        # Scalp surface
├── stats/                           # Volumetric and parcellation statistics
└── label/                           # Cortical parcellation labels
```

---

## Performance

| Stage | Cores | Time |
|---|---|---|
| recon-all (single-core) | 1 | 6–12 hours |
| recon-all (`-parallel -openmp 8`) | 8 | 5–7 hours |
| mri_watershed | 1 | ~5–10 minutes |
| build_5tt.py | 1 | ~1–3 minutes |

Recommended Docker resources: 16 GB RAM, 4+ CPU cores, 20 GB free disk space.

## References

- **FreeSurfer:** Dale AM, Fischl B, Sereno MI. *Cortical Surface-Based Analysis I: Segmentation and Surface Reconstruction.* NeuroImage 1999. [DOI: 10.1006/nimg.1998.0395](https://doi.org/10.1006/nimg.1998.0395)
- FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
- Docker: https://www.docker.com/

## License

This pipeline uses FreeSurfer (FreeSurfer License, free registration required) and Docker (Apache 2.0).