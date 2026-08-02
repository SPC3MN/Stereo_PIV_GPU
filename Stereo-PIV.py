"""
Batch stereo GPU-PIV pipeline -- im7 input via lvpyio, dewarped via DaVis's
own calibration polynomial
================================================================================
Reads LaVision .im7 stereo images directly (no TIFF conversion step) -- each
buffer/file provides BOTH cameras' raw double-frame images -- dewarps each
camera's images onto a shared world grid using the exact 3rd-order
polynomial from DaVis's calibration report ("Mapping of world (x'/y') to
raw coordinates (x/y)"), runs openpiv_gpu's `piv_gpu` on each camera's
dewarped pair, then combines the two in-plane displacement fields into
3-component (U, V, W) velocity. For a no-GPU CPU fallback using plain
openpiv-python, see CPU_Stereo_Processing.py instead -- it shares this
file's calibration/dewarping/reconstruction code via stereo_common.py.

Requires: pip install lvpyio   (LaVision's official reader, cross-platform,
pure pip install -- no DaVis license or C++ build tools needed to read data)

CONFIG FILE
-----------
Every setting below (formerly the CONTROLS class) now lives in a JSON file
-- CONFIG_PATH, default "stereo_piv_config.json" next to this script (or
pass a different path as argv[1]: `python Stereo-PIV.py my_config.json`).
If that file doesn't exist yet, load_controls() writes one out populated
with DEFAULT_CONFIG's values and proceeds using them -- so a first run
works out of the box, and after that you just edit the JSON (calibration
coefficients, input_path, piv_settings, etc.) and re-run; no need to touch
this .py file for day-to-day tuning. You only need to include the keys
you're actually changing -- anything missing from the file falls back to
DEFAULT_CONFIG.

INPUT_MODE options (set in the config file):
  "set"   -- point input_path at either:
               - a single DaVis stereo image set (a .set path, or a plain
                 folder lvpyio can read directly) -- processes just that
                 one set.
               - a folder that itself contains multiple *.set entries --
                 every set inside is batch-processed in turn, each into
                 its own subfolder of output_dir.
             See piv_common.resolve_set_paths() for the exact detection
             rule. lvpyio iterates the buffers directly in native
             LaVision-container order; each buffer's 4 frames (2 cameras x
             2 exposures) are split per STEREO_FRAME_ORDER below -- no
             manual file pairing needed.
  "loose" -- a plain folder of standalone .im7 files, e.g. ones you've
             copied out of a project. Auto-detects whether each file
             already contains all 4 exposures (one combined file per
             stereo pair) or each camera's double-frame pair is a SEPARATE
             file, matched by suffix_cam0/suffix_cam1. Always treated as a
             single run (no folder-of-sets batching).

SINGLE-SET PREVIEW
-------------------
When input_mode="set" and input_path points at exactly one set (not a
folder of several), the FIRST pair's 3-component velocity field is
computed, plotted, and opened for review before the rest of that set is
processed -- see piv_common.preview_first_snapshot(). Declining at the
prompt aborts the run. This step is skipped in folder-of-sets batch mode
and in "loose" mode, so unattended batch runs aren't blocked on a prompt.

WHAT'S REAL VS. A PLACEHOLDER RIGHT NOW
----------------------------------------
- cam0_mapping and cam1_mapping in the config file are both built from
  your actual DaVis calibration report coefficients now (calibration time
  260724_211326, plate 204-15-3, "Plane 1"/"Plane 2") -- verified:
  world_to_raw() reproduces the same numbers computed by hand from your
  screenshots. If you re-run DaVis's calibration later, update both here.
- alpha1_deg/alpha2_deg/beta1_deg/beta2_deg (used only in the final
  combination step) are also placeholders. This single-Z-plane
  calibration doesn't carry Z sensitivity on its own -- that has to come
  from either (a) DaVis's reported per-camera viewing angle to the sheet
  normal specifically (not necessarily the "Min/max angle 1-2" figure,
  which reads like the angle BETWEEN the two cameras rather than either
  one's angle to the normal -- worth confirming which it is), or (b) a
  second calibration at a different Z, differenced numerically the same
  way as the polynomial terms here. Don't trust W (or U/V, which also
  depend on these angles) until this is pinned down.
- stereo_frame_order (in the config file) is an ASSUMPTION about how
  DaVis interleaves the two cameras' frames within one combined
  buffer/file -- "camera_major" ([cam0_A, cam0_B, cam1_A, cam1_B]) is the
  default guess. If cam0_mapping visibly dewarps the wrong camera's image
  (garbled/black output), flip it to "frame_major"
  ([cam0_A, cam1_A, cam0_B, cam1_B]) -- see stereo_common.frames_from_stereo_buffer().
  Only relevant when a single buffer/file actually holds all 4 exposures;
  the "loose" separate-file case sidesteps this entirely since each
  camera's pair is its own file.
- piv_settings.search_size_iters (in the config file) is a best-effort
  translation of the pipeline's original multi-pass intent into the
  actual piv_gpu API's tuple-per-pass format -- see the PIV WINDOW
  SETTINGS note in DEFAULT_CONFIG below before trusting the vector field's
  spatial resolution/convergence.

PERFORMANCE NOTE
-----------------
Dewarping a full-resolution DaVis image (WORLD_SHAPE below) costs ~11-12s
the first time per camera, almost entirely in evaluating the polynomial
over every output pixel -- NOT the interpolation itself (~1.3s). Since the
mapping is identical for every frame from a given camera, CameraMapping
caches that coordinate grid after the first call, so it's paid once per
camera per run, not once per frame. If it's still a bottleneck for a large
dataset, swap `from scipy.ndimage import map_coordinates` for
`from cupyx.scipy.ndimage import map_coordinates` (CuPy is already a
dependency via piv_gpu) to run the warp on the GPU as well.

GPU MEMORY NOTE
----------------
Each camera's piv_gpu instance is built, run, and torn down (see
run_camera()) rather than both cameras' instances being created once and
held for the whole run -- only one camera's correlation buffers are
resident on the GPU at a time, which roughly halves peak VRAM. This costs
some speed, since FFT plans are rebuilt for every camera on every pair
instead of being reused across the run. On a memory-constrained GPU (a
few GB VRAM, especially if it's also driving a display or shared with
DaVis) this tradeoff is worth it. The other big lever if VRAM is still
too tight is piv_settings in the config file: lowering overlap_ratio
shrinks the number of simultaneous correlation windows (at the cost of
vector spatial resolution) -- this matters far more than min_search_size
does, since (for a fixed overlap_ratio) the correlation buffer memory is
roughly invariant to window size: n_windows scales as 1/step**2 while
each window's padded FFT buffer scales as window_size**2, and
step = window_size*(1-overlap_ratio), so those cancel except for the
overlap_ratio term.
"""

import os
import sys
import csv
import numpy as np

import piv_common as pc
import stereo_common as sc


# ======================================================================
# Config -- all pipeline settings, defaulted here and overridable via a
# JSON file (see load_controls() and the CONFIG FILE note in the module
# docstring above)
# ======================================================================
CONFIG_PATH = "stereo_piv_config.json"

DEFAULT_CONFIG = {
    # ---------------- Input source ----------------
    "input_mode": "set",                     # "set" or "loose"
    "input_path": "D:\\messy_data\\Stereo\\6-12_5.set",  # .set file / set folder / plain folder / folder-of-sets

    # Only used for input_mode == "set", if a given set turns out to be a
    # DaVis multiset. Which sub-set to process; 0 is usually camera 1 --
    # but for a proper stereo set both cameras live in the SAME sub-set's
    # buffers (see stereo_frame_order), so this is rarely needed here.
    "multiset_index": 0,

    # How the 4 frames in a combined stereo buffer/file are ordered (see
    # the module docstring above). Only matters when a single buffer/file
    # holds all 4 exposures.
    # "camera_major": [cam0_A, cam0_B, cam1_A, cam1_B]
    # "frame_major":  [cam0_A, cam1_A, cam0_B, cam1_B]
    "stereo_frame_order": "camera_major",

    # Only used for input_mode == "loose" when the two cameras turn out to
    # be SEPARATE double-frame files rather than one combined 4-frame file
    # (auto-detected from the first file). "cam1"/"cam2" here follows
    # DaVis's own (1-indexed) camera naming; they map to cam0_mapping /
    # cam1_mapping below. Adjust to match your actual naming convention.
    "suffix_cam0": "_cam1.im7",
    "suffix_cam1": "_cam2.im7",
    "loose_glob": "*.im7",                   # glob used to find files in "loose" mode

    # ---------------- Calibration mappings ----------
    # Both real now -- from DaVis's calibration report, calibration time
    # 260724_211326, plate 204-15-3, "Plane 1"/"Plane 2" (camera 1/camera
    # 2 in DaVis's own 1-indexed naming -> cam0_mapping/cam1_mapping here).
    # Fields match stereo_common.CameraMapping.__init__ above exactly.
    "cam0_mapping": {
        "x0": 2806.99, "x_span": 4096.00, "y0": 1387.18, "y_span": 3008.00,
        "dx_coefs": {"1": 882.1674, "s": 629.5431, "s2": -74.6835, "s3": -4.4885,
                     "t": -0.6616, "t2": 0.2021, "t3": -0.0545,
                     "st": -0.6915, "s2t": -0.0594, "t2s": -0.1322},
        "dy_coefs": {"1": 19.4802, "s": 17.2524, "s2": 1.4413, "s3": -0.0423,
                     "t": 65.1278, "t2": -0.3800, "t3": -0.5897,
                     "st": -76.3895, "s2t": -3.7352, "t2s": -0.2700},
        "name": "cam0 (Plane 1)",
    },
    "cam1_mapping": {
        "x0": 2806.99, "x_span": 4119.58, "y0": 1387.18, "y_span": 3025.32,
        "dx_coefs": {"1": 846.8601, "s": 633.6056, "s2": -75.5333, "s3": -4.8160,
                     "t": -1.0925, "t2": -0.0019, "t3": 0.6212,
                     "st": -0.6421, "s2t": -0.2899, "t2s": -0.6121},
        "dy_coefs": {"1": 19.2035, "s": 16.9346, "s2": 0.6940, "s3": 0.3712,
                     "t": 67.3521, "t2": -0.5134, "t3": -0.1081,
                     "st": -76.9309, "s2t": -4.0639, "t2s": -0.3362},
        "name": "cam1 (Plane 2)",
    },

    # World/dewarped output grid, from DaVis's calibration report
    # ("Size of dewarped image" / "Scale factor"). (rows, cols).
    "world_shape": (3067, 5874),
    "world_scale_px_per_mm": 17.92,
    "dewarp_order": 1,             # interpolation order for the warp (1=bilinear)

    # ---------------- Output ----------------
    "output_dir": "stereo_piv_output",

    # ---------------- PIV window size / passes / core settings ----------
    # piv_gpu.__init__(frame_shape, min_search_size, **kwargs) -- so
    # min_search_size is kept separate here (it's a required positional
    # arg, not part of piv_settings/**kwargs). piv_settings is forwarded
    # as piv_gpu(frame_shape, min_search_size, **piv_settings);
    # unrecognized keys are only WARNED about, never dropped -- see
    # piv_common.check_piv_settings() (piv_gpu itself ignores unknown
    # kwargs safely).
    #
    # search_size_iters is a TUPLE, one entry per multi-pass resolution
    # level -- its length sets the number of passes (window size doubles
    # per level going up from min_search_size), and each entry is the
    # number of deformation iterations to run at that pass's window size.
    # [1, 1, 1] here is a best-effort translation of the pipeline's
    # original intent (3 passes, refining down to min_search_size=32) into
    # this tuple format -- confirm the exact per-element semantics against
    # openpiv_gpu's own PIVGPU docs/source if precise convergence behavior
    # matters for your data.
    # Default schedule: one pass at 64px/50% overlap, then three passes at
    # 32px/75% overlap (search_size_iters=[1, 3] means 1 iteration at the
    # coarse 64px level, then 3 refinement iterations at the finest
    # min_search_size=32px level -- window size doubles per level going up
    # from min_search_size, see the note above).
    "min_search_size": 32,
    "piv_settings": {
        "search_size_iters": [1, 3],
        "overlap_ratio": [0.5, 0.75],
        "dt": 1.0,
    },

    # ---------------- Per-camera post-processing (before combining) -------
    "global_outlier_std": None,
    "replace_invalid": True,
    "smooth_field": False,
    "smooth_sigma": 1.0,

    # ---------------- Tiling (large frames / limited VRAM) --------------
    # When enabled, EACH CAMERA's dewarped frame pair is split into an
    # n_tiles_y x n_tiles_x grid of halo-padded tiles, each run through
    # its OWN piv_gpu instance (built, run, and freed before the next
    # tile) instead of one piv_gpu call on the whole (often very large --
    # world_shape above, not the raw camera resolution) dewarped frame --
    # peak VRAM is then bounded by one tile's window count, not the whole
    # frame's. Both cameras use the identical tile geometry (same
    # world_shape, same tiling settings), so their combined per-tile
    # results stay aligned point-for-point for reconstruct_stereo() --
    # no different handling needed there. Only worth turning on if the
    # full dewarped frame is too large to fit in VRAM at your desired
    # min_search_size/overlap_ratio; leave n_tiles at 1x1 (equivalent to
    # disabled) otherwise. margin_px null picks a safe default
    # automatically -- see piv_common.default_tile_margin().
    #
    # NOTE: tiled output is a flat/unstructured point set, not a
    # (ny, nx) grid -- and smooth_field above is skipped (with a warning)
    # for tiled output, since Gaussian smoothing needs a regular grid.
    # See piv_common.compute_tiles()'s module-level comment for why.
    "tiling": {
        "enabled": False,
        "n_tiles_y": 1,
        "n_tiles_x": 1,
        "margin_px": None,
    },

    # ---------------- Stereo geometry ----------------
    # Rig: each camera tilted 45 deg from the calibration plate normal,
    # 90 deg apart from each other -- a symmetric, single-plane-tilt
    # configuration (no relative vertical tilt between the cameras).
    # Using DaVis's measured separation (89.53 deg, "Min/max angle 1-2")
    # split symmetrically rather than the nominal 45/45 design value.
    # beta1/beta2 = 0 (no y-z tilt) is exactly the degenerate case
    # reconstruct_stereo was verified against above.
    #
    # STILL TO CONFIRM: which physical camera (cam0/"Plane 1" vs
    # cam1/"Plane 2") is on which side -- this only flips the sign of the
    # reconstructed W, so if W comes out inverted relative to a known
    # reference (e.g. mean flow direction, or DaVis's own W sign
    # convention), swap alpha1_deg/alpha2_deg.
    "alpha1_deg": -89.53 / 2,   # -44.765
    "alpha2_deg": 89.53 / 2,    # +44.765
    "beta1_deg": 0.0,
    "beta2_deg": 0.0,

    # ---------------- Units ----------------
    "frame_dt_s": None,            # s between frames; None keeps displacement units
    "apply_v_sign_flip": True,

    # ---------------- Output artifacts ----------------
    "save_npz": True,
    "save_plot": True,
    "save_summary_csv": True,
    "plot_dpi": 150,
    "quiver_scale": 1000,
    "show_plots": False,

    "verbose": True,
}


class CONTROLS:
    """Populated at runtime by load_controls() -- see DEFAULT_CONFIG and
    the CONFIG FILE note in the module docstring above."""
    pass


def _fixup_stereo_controls(config, ctrl):
    ctrl.world_shape = tuple(ctrl.world_shape)
    ctrl.cam0_mapping = sc.CameraMapping(**config["cam0_mapping"])
    ctrl.cam1_mapping = sc.CameraMapping(**config["cam1_mapping"])


def load_controls(config_path):
    return pc.load_controls(config_path, DEFAULT_CONFIG, CONTROLS, on_loaded=_fixup_stereo_controls)


# ======================================================================
# Per-camera / per-pair processing
# ======================================================================
def run_camera(frame_a, frame_b, ctrl):
    """Build a fresh piv_gpu instance for ONE camera's dewarped pair, run
    it, then free its GPU memory before returning. This keeps only one
    camera's correlation buffers resident on the GPU at a time instead of
    holding both cameras' piv_gpu instances simultaneously -- roughly
    halves peak VRAM, at the cost of rebuilding FFT plans for every
    camera, every pair (no plan reuse across the run). Worth it on a
    memory-constrained GPU; if VRAM isn't the bottleneck, hoist the
    init_gpu_processor() calls back out to main() and reuse the same
    process0/process1 across all pairs instead.

    If ctrl.tiling is enabled, this camera's dewarped frame is instead
    processed tile by tile (piv_common.process_frames_tiled()) -- same
    per-camera memory-bounding idea, just applied at a finer grain, for
    cases where even ONE camera's full dewarped frame doesn't fit in
    VRAM at the desired window/overlap settings."""
    tiling = ctrl.tiling
    if tiling["enabled"]:
        margin = tiling["margin_px"] or pc.default_tile_margin(ctrl.min_search_size, ctrl.piv_settings)
        init_raw_fn = lambda shape: pc._init_gpu_processor_raw(shape, ctrl.min_search_size, ctrl.piv_settings)
        x, y, u, v, valid, elapsed = pc.process_frames_tiled(
            frame_a, frame_b, ctrl, init_raw_fn,
            tiling["n_tiles_y"], tiling["n_tiles_x"], margin,
            report_gpu_mem=True, free_pools_fn=pc.free_gpu_pools,
        )
        return u, v, valid, elapsed, x, y

    process, x, y = pc.init_gpu_processor(frame_a.shape, ctrl.min_search_size, ctrl.piv_settings)
    u, v, valid, elapsed = pc.process_frames(process, frame_a, frame_b, ctrl, report_gpu_mem=True)
    del process
    pc.free_gpu_pools()
    return u, v, valid, elapsed, x, y


def handle_pair(pair_id, dw_a0, dw_b0, dw_a1, dw_b1, ctrl, angles, output_dir):
    if ctrl.verbose:
        print(f"Processing {pair_id} ...", end=" ", flush=True)

    alpha1, alpha2, beta1, beta2 = angles
    u1, v1, valid1, elapsed1, x, y = run_camera(dw_a0, dw_b0, ctrl)
    u2, v2, valid2, elapsed2, _, _ = run_camera(dw_a1, dw_b1, ctrl)
    valid = valid1 & valid2
    elapsed = elapsed1 + elapsed2

    # world grid is in pixels at ctrl.world_scale_px_per_mm px/mm
    u1_mm, v1_mm, u2_mm, v2_mm = (a / ctrl.world_scale_px_per_mm
                                   for a in (u1, v1, u2, v2))

    U, V, W = sc.reconstruct_stereo(u1_mm, v1_mm, u2_mm, v2_mm,
                                     alpha1, alpha2, beta1, beta2)

    if ctrl.frame_dt_s is not None:
        U, V, W = (a / ctrl.frame_dt_s for a in (U, V, W))

    U = np.where(valid, U, np.nan)
    V = np.where(valid, V, np.nan)
    W = np.where(valid, W, np.nan)

    n_valid, n_total = int(valid.sum()), int(valid.size)
    if ctrl.verbose:
        print(f"{elapsed:.3f} s, {n_valid}/{n_total} valid vectors")

    if ctrl.save_npz:
        np.savez(os.path.join(output_dir, f"{pair_id}_stereo_velocity.npz"),
                 x=x, y=y, U=U, V=V, W=W, valid=valid)

    if ctrl.save_plot:
        sc.plot_and_save_stereo(x, y, U, V, W, valid,
                                 os.path.join(output_dir, f"{pair_id}_stereo_quiver.png"),
                                 ctrl, title=f"Stereo PIV -- {pair_id}")

    row = (pair_id, elapsed, n_valid, n_total)
    return row, x, y, U, V, W, valid


def process_pairs(pair_source, ctrl, angles, output_dir, interactive_preview):
    summary_rows = []
    for idx, (pair_id, fa0, fb0, fa1, fb1) in enumerate(pair_source):
        # dewarp both frames of both cameras onto the shared world grid
        dw_a0 = ctrl.cam0_mapping.dewarp_image(fa0, ctrl.world_shape, ctrl.dewarp_order)
        dw_b0 = ctrl.cam0_mapping.dewarp_image(fb0, ctrl.world_shape, ctrl.dewarp_order)
        dw_a1 = ctrl.cam1_mapping.dewarp_image(fa1, ctrl.world_shape, ctrl.dewarp_order)
        dw_b1 = ctrl.cam1_mapping.dewarp_image(fb1, ctrl.world_shape, ctrl.dewarp_order)

        row, x, y, U, V, W, valid = handle_pair(pair_id, dw_a0, dw_b0, dw_a1, dw_b1,
                                                  ctrl, angles, output_dir)
        summary_rows.append(row)

        if idx == 0 and interactive_preview:
            preview_path = os.path.join(output_dir, f"{pair_id}_first_snapshot_preview.png")
            sc.plot_and_save_stereo(x, y, U, V, W, valid, preview_path, ctrl,
                                     title=f"First snapshot preview -- {pair_id}")
            pc.preview_first_snapshot(preview_path)

    return summary_rows


def write_summary(summary_rows, output_dir, ctrl):
    if not summary_rows:
        return
    if ctrl.save_summary_csv:
        csv_path = os.path.join(output_dir, "stereo_processing_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "process_time_s", "n_valid", "n_total"])
            writer.writerows(summary_rows)
        print(f"Summary written to {csv_path}")

    total_time = sum(row[1] for row in summary_rows)
    print(f"Done: {len(summary_rows)} pair(s) in {total_time:.3f} s "
          f"({total_time / len(summary_rows):.3f} s/pair average)")


# ======================================================================
# Main
# ======================================================================
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    ctrl = load_controls(config_path)
    os.makedirs(ctrl.output_dir, exist_ok=True)

    if ctrl.input_mode == "set":
        set_paths, is_batch = pc.resolve_set_paths(ctrl.input_path)
    elif ctrl.input_mode == "loose":
        set_paths, is_batch = [ctrl.input_path], False
    else:
        sys.exit(f"Unknown input_mode: {ctrl.input_mode!r} (use 'set' or 'loose')")

    if is_batch:
        print(f"[info] '{ctrl.input_path}' contains {len(set_paths)} set(s) -- "
              "batch-processing each (no first-snapshot preview in this mode)")

    angles = (np.deg2rad(ctrl.alpha1_deg), np.deg2rad(ctrl.alpha2_deg),
              np.deg2rad(ctrl.beta1_deg), np.deg2rad(ctrl.beta2_deg))

    grand_summary = []
    for set_path in set_paths:
        output_dir = (os.path.join(ctrl.output_dir, pc.set_label(set_path))
                       if is_batch else ctrl.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if ctrl.input_mode == "set":
            print(f"[info] processing set '{set_path}'")
            pair_source = sc.iter_stereo_from_set(ctrl, set_path)
        else:
            pair_source = sc.iter_stereo_from_loose_files(ctrl)

        summary_rows = process_pairs(pair_source, ctrl, angles, output_dir,
                                      interactive_preview=not is_batch)
        if not summary_rows:
            print(f"[warn] no stereo pairs were processed for '{set_path}'")
            continue

        write_summary(summary_rows, output_dir, ctrl)
        grand_summary.extend(summary_rows)

    if not grand_summary:
        sys.exit("No stereo pairs were processed -- check input_mode/input_path")


if __name__ == "__main__":
    main()
