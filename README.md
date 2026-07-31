# Stereo GPU-PIV Processing (raw im7 input, DaVis dewarping)

Processes **raw (non-dewarped)** stereo `.im7` buffers from a **single
LaVision/DaVis set** (each buffer holds both cameras' synchronized
double-frame images -- no separate per-camera folders to keep in sync),
dewarps each camera's images onto a shared world grid using the exact
3rd-order polynomial from DaVis's calibration report ("Mapping of world
(x'/y') to raw coordinates (x/y)"), runs
[`openpiv-python-gpu`](https://github.com/ali-sh-96/openpiv-python-gpu)'s
`piv_gpu` on each camera's dewarped pair, then combines the two in-plane
displacement fields into 3-component (U, V, W) stereo velocity.

## ⚠️ Before trusting any output

- **`cam0_mapping`/`cam1_mapping` in the config file are both real now**
  (calibration time 260724_211326, plate 204-15-3). If you re-run DaVis's
  calibration later, update both from the new report.
- **`alpha1_deg`/`alpha2_deg` (and `beta1_deg`/`beta2_deg`) are placeholders
  too.** This single-Z-plane calibration doesn't carry Z sensitivity on its
  own -- it needs either (a) DaVis's reported per-camera viewing angle to
  the sheet normal specifically (not necessarily the "Min/max angle 1-2"
  figure, which reads like the angle *between* the two cameras rather than
  either one's angle to the normal -- worth confirming which it is), or
  (b) a second calibration at a different Z, differenced numerically the
  same way as the polynomial terms. **Don't trust W (or U/V, which also
  depend on these angles) until this is pinned down.**
- **Which physical camera is on which side is still unconfirmed.** Swapping
  `alpha1_deg`/`alpha2_deg` only flips the sign of the reconstructed W --
  if W comes out inverted relative to a known reference (mean flow
  direction, or DaVis's own W sign convention), swap them.
- **`stereo_frame_order` is an assumption, not a verified fact.** A combined
  stereo buffer's 4 frames (2 cameras x 2 frames) are read in
  `"camera_major"` order (`[cam0_A, cam0_B, cam1_A, cam1_B]`) by default.
  If `cam0_mapping` visibly dewarps the wrong camera's raw image
  (garbled/black output), switch it to `"frame_major"`
  (`[cam0_A, cam1_A, cam0_B, cam1_B]`). Only relevant in `"set"` mode, or
  in `"loose"` mode when the two cameras are combined into one file.
- **`piv_settings.search_size_iters` is a best-effort translation, not a
  verified value.** `piv_gpu`'s real signature takes it as a tuple, one
  entry per multi-pass resolution level (length = number of passes; each
  entry = deformation iterations at that pass's window size) -- `[1, 1, 1]`
  is a guess at reproducing the pipeline's original 3-pass intent. Confirm
  the exact semantics against `openpiv-python-gpu`'s own docs if precise
  convergence behavior matters for your data.

## What it does

- Reads raw stereo `.im7` images directly via `lvpyio` in one of two
  `input_mode`s:
  - `"set"` -- a DaVis image set (folder or `.set` file), iterated in
    native LaVision-container order
  - `"loose"` -- a plain folder of standalone `.im7` files, auto-detecting
    whether each file already combines both cameras' 4 exposures, or each
    camera's double-frame pair is a separate file matched by
    `suffix_cam0`/`suffix_cam1`
- Dewarps each camera's raw images onto a shared world grid using DaVis's
  own 3rd-order polynomial mapping (`CameraMapping`), caching the coordinate
  grid per camera so it's only computed once per run, not once per frame
- Runs `piv_gpu` on each camera's dewarped pair independently, with the same
  post-processing options as the planar pipeline (outlier rejection, invalid
  vector interpolation, smoothing)
- Combines the two cameras' in-plane displacement fields into 3-component
  (U, V, W) displacement via least-squares stereo reconstruction
  (`reconstruct_stereo`), which degrades gracefully in the degenerate case
  where both cameras share the same y-tilt (`beta1 == beta2`)
- Saves results per pair as `.npz` (and optionally a stereo quiver plot
  colored by W), plus an optional CSV summary across the batch

## Requirements

- Python 3.9+
- An NVIDIA GPU + CUDA Toolkit (11.2-11.8 or 12.x)
- [`openpiv-python-gpu`](https://github.com/ali-sh-96/openpiv-python-gpu)
  (not on PyPI -- installed by cloning, see below)

## System prerequisites (non-Python)

These need to be in place *before* `pip install`-ing anything, since CuPy is
just a wrapper around a working CUDA install:

- **NVIDIA GPU driver** -- the latest driver that supports your target CUDA
  Toolkit version ([download](https://www.nvidia.com/Download/index.aspx)).
- **CUDA Toolkit** -- version 11.2-11.8 or 12.x
  ([download](https://developer.nvidia.com/cuda-downloads)), matching
  whichever `cupy-cudaXXx` wheel you install below. This is a system-level
  install, not something `pip` provides.
- **Windows only: Visual Studio C++ build tools** -- the CUDA Toolkit
  installer on Windows requires Visual Studio with the "Desktop development
  with C++" workload installed first (the free
  [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
  installer is sufficient; you don't need the full IDE).
- **Linux**: the equivalent (`gcc`/`build-essential`) is usually already
  present on most distros.

Not required: no DaVis license or LaVision software is needed -- `lvpyio`
reads `.im7` files directly as a pure pip package. No separate compiler is
needed just to install CuPy itself (the `cupy-cudaXXx` wheels are prebuilt);
the compiler requirement above comes from CUDA Toolkit's own installer.

## Installation

1. Install CuPy for your CUDA Toolkit version:

   ```bash
   pip install cupy-cuda11x   # CUDA Toolkit 11.2 - 11.8
   # or
   pip install cupy-cuda12x   # CUDA Toolkit 12.x
   ```

2. Install the remaining Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Clone `openpiv-python-gpu` and add it to your `PYTHONPATH` (it's a
   source dependency, not a PyPI package):

   ```bash
   git clone https://github.com/ali-sh-96/openpiv-python-gpu
   ```

   ```python
   import sys
   sys.path.append("/path/to/openpiv-python-gpu")
   ```

   Alternatively, set `PYTHONPATH` in your shell before running the script.

## Configuration file

All pipeline settings live in a JSON file -- `stereo_piv_config.json` next
to `Stereo-PIV.py` by default, or pass a different path as the first
argument: `python Stereo-PIV.py my_config.json`. On first run, if that
file doesn't exist, the script writes one out populated with its built-in
defaults and proceeds using them -- so it works out of the box, and after
that you just edit the JSON and re-run; no need to touch the `.py` file
for day-to-day tuning. You only need to include the keys you're actually
changing in the file -- anything you leave out falls back to the default.

## Usage

1. Run `python Stereo-PIV.py` once to generate `stereo_piv_config.json`
   with default values.
2. Fill in `cam0_mapping` / `cam1_mapping` in that file with both cameras'
   real DaVis calibration report coefficients (`x0`, `x_span`, `y0`,
   `y_span`, `dx_coefs`, `dy_coefs`).
3. Set the stereo geometry angles (`alpha1_deg`/`alpha2_deg`,
   `beta1_deg`/`beta2_deg`) per the warning above.
4. Set `input_mode`/`input_path` to point at your stereo set (or loose
   folder), and confirm `stereo_frame_order` (or `suffix_cam0`/
   `suffix_cam1` in loose mode) matches how your data actually separates
   the two cameras (see warning above).
5. Edit the rest of the config file, then run:

   ```bash
   python Stereo-PIV.py
   ```

### Key settings (`stereo_piv_config.json`)

| Setting | Description |
|---|---|
| `input_mode` | `"set"` (DaVis image set) or `"loose"` (plain folder of `.im7` files) |
| `input_path` | Raw (non-dewarped) stereo `.im7` source -- a `.set` file/set folder (`"set"` mode) or a plain folder (`"loose"` mode) |
| `multiset_index` | Which sub-set to use when `input_path` is a DaVis multi-set (`"set"` mode) |
| `stereo_frame_order` | How a combined buffer/file's 4 frames are ordered: `"camera_major"` (`[cam0_A, cam0_B, cam1_A, cam1_B]`, default) or `"frame_major"` (`[cam0_A, cam1_A, cam0_B, cam1_B]`) |
| `suffix_cam0` / `suffix_cam1` | (`"loose"` mode only) filename suffixes used to pair each camera's double-frame file when the two cameras aren't combined into one file |
| `loose_glob` | (`"loose"` mode only) glob pattern used to find files in `input_path` |
| `cam0_mapping` / `cam1_mapping` | Objects holding each camera's DaVis calibration polynomial coefficients (`x0`, `x_span`, `y0`, `y_span`, `dx_coefs`, `dy_coefs`, `name`) |
| `world_shape` | Shape of the shared dewarped output grid, from DaVis's "Size of dewarped image" |
| `world_scale_px_per_mm` | World-grid scale factor, from DaVis's calibration report |
| `dewarp_order` | Interpolation order for the dewarp (1 = bilinear) |
| `min_search_size` | Interrogation window size in px (required by `piv_gpu`, multiples of 8/powers of 2 only) |
| `piv_settings` | Forwarded to `piv_gpu(frame_shape, min_search_size, **piv_settings)` -- `search_size_iters`, `overlap_ratio`, `dt`, and any other `piv_gpu` kwarg. Unrecognized keys are warned about (typo check), not silently dropped. |
| `global_outlier_std` | Reject vectors more than N standard deviations from the mean (`None` disables) |
| `replace_invalid` | Interpolate over invalid/NaN vectors, per camera, before combining |
| `smooth_field` / `smooth_sigma` | Gaussian-smooth each camera's field before combining |
| `alpha1_deg` / `alpha2_deg` / `beta1_deg` / `beta2_deg` | Stereo viewing angles used in the U/V/W reconstruction -- see warning above |
| `frame_dt_s` | s between frames; `None` keeps displacement units instead of velocity |
| `apply_v_sign_flip` | Flip the sign of each camera's `v` before combining |
| `save_npz` / `save_plot` / `save_summary_csv` | Which output artifacts to write |

## Output

For each image pair `<pair_id>`, in `output_dir`:

- `<pair_id>_stereo_velocity.npz` -- contains `x`, `y`, `U`, `V`, `W`, `valid` arrays
- `<pair_id>_stereo_quiver.png` -- quiver plot colored by `W` (if `save_plot = True`)

For the whole batch:

- `stereo_processing_summary.csv` -- `pair_id, process_time_s, n_valid, n_total` (if `save_summary_csv = True`)

## Performance note

Dewarping a full-resolution DaVis image (`world_shape`) costs ~11-12s the
first time per camera, almost entirely in evaluating the calibration
polynomial over every output pixel (not the interpolation itself, ~1.3s).
Because the mapping is identical for every frame from a given camera,
`CameraMapping` caches that coordinate grid after the first call, so it's
paid once per camera per run, not once per frame. If dewarping is still a
bottleneck for a large dataset, swap `from scipy.ndimage import
map_coordinates` for `from cupyx.scipy.ndimage import map_coordinates`
(CuPy is already a dependency via `piv_gpu`) to run the warp on the GPU too.
