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
3-component (U, V, W) velocity.

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
  "set"   -- point at a DaVis stereo image set (a folder or a .set file,
             still inside its original project structure). lvpyio iterates
             the buffers directly in native LaVision-container order; each
             buffer's 4 frames (2 cameras x 2 exposures) are split per
             STEREO_FRAME_ORDER below -- no manual file pairing needed.
             This is the fastest path if your im7s are still sitting where
             DaVis wrote them.
  "loose" -- a plain folder of standalone .im7 files, e.g. ones you've
             copied out of a project. Auto-detects whether each file
             already contains all 4 exposures (one combined file per
             stereo pair) or each camera's double-frame pair is a SEPARATE
             file, matched by suffix_cam0/suffix_cam1.

WHAT'S REAL VS. A PLACEHOLDER RIGHT NOW
----------------------------------------
- cam0_mapping in the config file is built from your actual "Plane 1"
  calibration report coefficients -- verified: world_to_raw() reproduces
  the same numbers computed by hand from your screenshot.
- cam1_mapping is a PLACEHOLDER (clearly marked in DEFAULT_CONFIG below)
  so the pipeline is exercisable end-to-end. Replace it with camera 2's
  ("Plane 2") actual coefficients from the same DaVis report panel, in
  the config file, before trusting any output.
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
  ([cam0_A, cam1_A, cam0_B, cam1_B]) -- see frames_from_stereo_buffer()
  below. Only relevant when a single buffer/file actually holds all 4
  exposures; the "loose" separate-file case sidesteps this entirely since
  each camera's pair is its own file.
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
DaVis) this tradeoff is worth it; if VRAM isn't tight, build process0/
process1 once in main() (as the planar pipeline does with a single
process) and reuse them pair to pair instead. The other big lever if
VRAM is still too tight is piv_settings in the config file: lowering
overlap_ratio shrinks the number of simultaneous correlation windows (at
the cost of vector spatial resolution) -- this matters far more than
min_search_size does, since (for a fixed overlap_ratio) the correlation
buffer memory is roughly invariant to window size: n_windows scales as
1/step**2 while each window's padded FFT buffer scales as window_size**2,
and step = window_size*(1-overlap_ratio), so those cancel except for the
overlap_ratio term.
"""

import os
import sys
import json
import glob
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cupy as cp
from scipy.ndimage import map_coordinates

import lvpyio as lv
from openpiv_gpu.gpu_process import piv_gpu


# ======================================================================
# Camera calibration: DaVis's polynomial world<->raw mapping
# ======================================================================
class CameraMapping:
    """
    DaVis 3rd-order polynomial calibration, single Z plane:
        s(x') = 2*(x' - x0) / x_span
        t(y') = 2*(y' - y0) / y_span
        x = x' - dx(s, t)
        y = y' - dy(s, t)
    dx_coefs / dy_coefs are dicts with keys
    '1','s','s2','s3','t','t2','t3','st','s2t','t2s' -- read directly off
    DaVis's calibration report panel, in order.
    """
    def __init__(self, x0, x_span, y0, y_span, dx_coefs, dy_coefs, name=""):
        self.x0, self.x_span = x0, x_span
        self.y0, self.y_span = y0, y_span
        self.dx_coefs, self.dy_coefs = dx_coefs, dy_coefs
        self.name = name
        self._cached_shape = None
        self._x_raw = self._y_raw = None

    def s(self, xp):
        return 2 * (xp - self.x0) / self.x_span

    def t(self, yp):
        return 2 * (yp - self.y0) / self.y_span

    @staticmethod
    def _poly(s, t, c):
        return (c['1'] + c['s'] * s + c['s2'] * s**2 + c['s3'] * s**3
                 + c['t'] * t + c['t2'] * t**2 + c['t3'] * t**3
                 + c['st'] * s * t + c['s2t'] * s**2 * t + c['t2s'] * t**2 * s)

    def world_to_raw(self, xp, yp):
        s, t = self.s(xp), self.t(yp)
        x = xp - self._poly(s, t, self.dx_coefs)
        y = yp - self._poly(s, t, self.dy_coefs)
        return x, y

    def _ensure_grid(self, world_shape):
        if self._cached_shape != world_shape:
            ny, nx = world_shape
            yp, xp = np.mgrid[0:ny, 0:nx].astype(np.float32)
            self._x_raw, self._y_raw = self.world_to_raw(xp, yp)
            self._cached_shape = world_shape

    def dewarp_image(self, raw_image, world_shape, order=1):
        """Backward-map raw_image onto a (world_shape) grid. Coordinate
        grid is computed once per (camera, world_shape) and cached."""
        self._ensure_grid(world_shape)
        return map_coordinates(raw_image, [self._y_raw, self._x_raw],
                                order=order, mode="constant", cval=0.0)


# ======================================================================
# Config -- all pipeline settings, defaulted here and overridable via a
# JSON file (see load_controls() and the CONFIG FILE note in the module
# docstring above)
# ======================================================================
CONFIG_PATH = "stereo_piv_config.json"

DEFAULT_CONFIG = {
    # ---------------- Input source ----------------
    "input_mode": "set",                     # "set" or "loose"
    "input_path": "D:\\messy_data\\Stereo\\6-12_5.set",  # .set file / set folder / plain folder

    # Only used for input_mode == "set", if that path turns out to be a
    # DaVis multi-set. Which sub-set to process; 0 is usually camera 1 --
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
    # cam0_mapping: your actual "Plane 1" calibration report coefficients.
    # cam1_mapping: PLACEHOLDER -- replace with camera 2's ("Plane 2")
    # actual coefficients from the same DaVis report panel before trusting
    # any output. Fields match CameraMapping.__init__ above exactly.
    "cam0_mapping": {
        "x0": 2806.99, "x_span": 4096.00, "y0": 1387.18, "y_span": 3008.00,
        "dx_coefs": {"1": 804.1028, "s": 628.9870, "s2": 84.0572, "s3": -5.4234,
                     "t": 3.1818, "t2": -0.3017, "t3": 0.2112,
                     "st": 0.6693, "s2t": -0.1956, "t2s": -0.4162},
        "dy_coefs": {"1": 28.8679, "s": 3.4689, "s2": -0.0086, "s3": -0.0561,
                     "t": -1.2230, "t2": 1.3855, "t3": -0.9813,
                     "st": 89.0937, "s2t": -5.3036, "t2s": -0.2961},
        "name": "cam0",
    },
    "cam1_mapping": {
        "x0": 2806.99, "x_span": 4096.00, "y0": 1387.18, "y_span": 3008.00,
        "dx_coefs": {"1": -780.0, "s": -610.0, "s2": 80.0, "s3": 5.0,
                     "t": -3.0, "t2": 0.3, "t3": -0.2,
                     "st": -0.6, "s2t": 0.2, "t2s": 0.4},
        "dy_coefs": {"1": 30.0, "s": -3.2, "s2": 0.01, "s3": 0.05,
                     "t": 1.3, "t2": -1.4, "t3": 0.98,
                     "st": -88.0, "s2t": 5.0, "t2s": 0.3},
        "name": "cam1 (PLACEHOLDER -- not your real data)",
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
    # check_piv_settings() (piv_gpu itself ignores unknown kwargs safely).
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
    "min_search_size": 32,
    "piv_settings": {
        "search_size_iters": [1, 1, 1],
        "overlap_ratio": 0.5,
        "dt": 1.0,
    },

    # ---------------- Per-camera post-processing (before combining) -------
    "global_outlier_std": None,
    "replace_invalid": True,
    "smooth_field": False,
    "smooth_sigma": 1.0,

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


def load_controls(config_path):
    """Load pipeline settings from a JSON config file, creating one
    (populated with DEFAULT_CONFIG) if it doesn't exist yet, then apply
    the merged result onto CONTROLS. Keys present in the file override
    DEFAULT_CONFIG's; anything the file doesn't mention falls back to the
    default -- you only need to include the settings you're changing."""
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path) as f:
            user_config = json.load(f)
        config.update(user_config)
        print(f"[info] loaded config from '{config_path}'")
    else:
        with open(config_path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        print(f"[info] '{config_path}' didn't exist -- wrote defaults there. "
              "Edit it (calibration coefficients, input_path, etc.) and "
              "re-run to customize.")

    for key, val in config.items():
        setattr(CONTROLS, key, val)

    CONTROLS.world_shape = tuple(CONTROLS.world_shape)
    CONTROLS.cam0_mapping = CameraMapping(**config["cam0_mapping"])
    CONTROLS.cam1_mapping = CameraMapping(**config["cam1_mapping"])
    return CONTROLS


# ======================================================================
# im7 -> numpy frame extraction
# ======================================================================
def frames_from_buffer(buf):
    """Pull raw intensity arrays for frame A and frame B out of a single
    camera's double-frame Buffer."""
    if len(buf.frames) < 2:
        raise ValueError(
            f"expected a double-frame buffer (2 frames), got {len(buf.frames)}"
        )
    frame_a = buf.frames[0].images[0]
    frame_b = buf.frames[1].images[0]
    return frame_a, frame_b


def frames_from_stereo_buffer(buf, frame_order):
    """Pull raw intensity arrays for BOTH cameras' frame A/B out of a
    combined stereo Buffer's 4 frames (2 cameras x 2 exposures).

    lvpyio doesn't label which frame belongs to which camera -- the ORDER
    is an assumption (see STEREO_FRAME_ORDER in the module docstring).
    Confirm it against your own set before trusting output; flip
    stereo_frame_order in CONTROLS if the wrong pair of frames ends up
    dewarped by each camera's mapping."""
    if len(buf.frames) != 4:
        raise ValueError(
            f"expected a 4-frame stereo buffer (2 cameras x 2 frames), got "
            f"{len(buf.frames)} -- if your cameras are stored as SEPARATE "
            "double-frame files, use input_mode='loose' with "
            "suffix_cam0/suffix_cam1 set correctly"
        )
    f0, f1, f2, f3 = (f.images[0] for f in buf.frames)
    if frame_order == "camera_major":     # [cam0_A, cam0_B, cam1_A, cam1_B]
        return f0, f1, f2, f3
    elif frame_order == "frame_major":    # [cam0_A, cam1_A, cam0_B, cam1_B]
        return f0, f2, f1, f3
    raise ValueError(f"unknown stereo_frame_order {frame_order!r}")


def iter_stereo_from_set(ctrl):
    """Yield (pair_id, fa0, fb0, fa1, fb1) from a DaVis stereo image set."""
    if lv.is_multiset(ctrl.input_path):
        print(f"[info] '{ctrl.input_path}' is a multi-set -- using sub-set "
              f"index {ctrl.multiset_index}")
        sets = lv.read_set(ctrl.input_path)
        dataset = sets[ctrl.multiset_index]
        owns_dataset = False
    else:
        dataset = lv.read_set(ctrl.input_path)
        owns_dataset = True

    try:
        n = len(dataset)
        for i in range(n):
            pair_id = f"{i:04d}"
            buf = dataset[i]
            fa0, fb0, fa1, fb1 = frames_from_stereo_buffer(buf, ctrl.stereo_frame_order)
            yield pair_id, fa0, fb0, fa1, fb1
    finally:
        if owns_dataset:
            dataset.close()


def iter_stereo_from_loose_files(ctrl):
    """Yield (pair_id, fa0, fb0, fa1, fb1) from a plain folder of .im7
    files -- either one combined 4-frame file per stereo pair, or each
    camera's double-frame pair as a separate file (auto-detected)."""
    paths = sorted(glob.glob(os.path.join(ctrl.input_path, ctrl.loose_glob)))
    if not paths:
        sys.exit(f"No files matching '{ctrl.loose_glob}' in '{ctrl.input_path}'")

    first_buf = lv.read_buffer(paths[0])
    combined = len(first_buf.frames) >= 4

    if combined:
        for path in paths:
            pair_id = os.path.splitext(os.path.basename(path))[0]
            buf = lv.read_buffer(path)
            fa0, fb0, fa1, fb1 = frames_from_stereo_buffer(buf, ctrl.stereo_frame_order)
            yield pair_id, fa0, fb0, fa1, fb1
    else:
        # each camera's double-frame pair is a SEPARATE file, matched by suffix
        files0 = sorted(p for p in paths if p.endswith(ctrl.suffix_cam0))
        if not files0:
            sys.exit(
                f"Files aren't combined 4-frame stereo buffers but none end "
                f"in '{ctrl.suffix_cam0}' -- set suffix_cam0/suffix_cam1 in "
                "the config file to match your naming"
            )
        for path0 in files0:
            path1 = path0[: -len(ctrl.suffix_cam0)] + ctrl.suffix_cam1
            if not os.path.exists(path1):
                print(f"[warn] no match for {os.path.basename(path0)} "
                      f"(expected {os.path.basename(path1)}) -- skipping")
                continue
            pair_id = os.path.basename(path0)[: -len(ctrl.suffix_cam0)]
            fa0, fb0 = frames_from_buffer(lv.read_buffer(path0))
            fa1, fb1 = frames_from_buffer(lv.read_buffer(path1))
            yield pair_id, fa0, fb0, fa1, fb1


# ======================================================================
# Shared PIV / post-processing helpers (same logic as the planar pipeline)
# ======================================================================
# The real keyword names piv_gpu.__init__(frame_shape, min_search_size,
# **kwargs) accepts -- min_search_size itself is a separate required
# positional arg, NOT one of these. piv_gpu already ignores unrecognized
# kwargs safely (each one is read via `kwargs["x"] if "x" in kwargs else
# DEFAULT`, never strict signature binding), so this set exists only to
# WARN about likely typos in piv_settings, not to filter/drop anything --
# unlike the old inspect.signature-based approach, which was actively
# wrong here (it matched against **kwargs's own parameter name, not the
# individual setting names it swallows, so it dropped everything real).
PIV_GPU_SETTINGS_KEYS = frozenset({
    "search_size_iters", "overlap_ratio", "shrink_ratio", "center",
    "normalize", "mask_zero", "subpixel_method", "n_fft", "deforming_par",
    "batch_size", "s2n_method", "s2n_size", "validation_size", "s2n_tol",
    "median_tol", "mad_tol", "mean_tol", "rms_tol", "num_replacing_iters",
    "replacing_method", "replacing_size", "revalidate", "smooth",
    "smoothing_par", "dt", "scaling_par", "mask", "dtype_f",
})


def check_piv_settings(piv_settings):
    unknown = sorted(set(piv_settings) - PIV_GPU_SETTINGS_KEYS)
    if unknown:
        print(f"[warn] piv_settings has keys piv_gpu won't recognize: "
              f"{unknown} -- check spelling against piv_gpu's __init__ "
              "kwargs (they're silently ignored, not an error)")


def global_outlier_mask(u, v, n_std):
    if n_std is None:
        return np.zeros_like(u, dtype=bool)
    u_mean, u_std = np.nanmean(u), np.nanstd(u)
    v_mean, v_std = np.nanmean(v), np.nanstd(v)
    return (np.abs(u - u_mean) > n_std * u_std) | (np.abs(v - v_mean) > n_std * v_std)


def replace_invalid_vectors(x, y, u, v, valid_mask):
    from scipy.interpolate import griddata
    invalid = ~valid_mask
    if not invalid.any():
        return u, v
    pts_valid = np.column_stack([x[valid_mask], y[valid_mask]])
    u_out, v_out = u.copy(), v.copy()
    u_out[invalid] = griddata(pts_valid, u[valid_mask], (x[invalid], y[invalid]), method="linear")
    v_out[invalid] = griddata(pts_valid, v[valid_mask], (x[invalid], y[invalid]), method="linear")
    still_bad = np.isnan(u_out)
    if still_bad.any():
        u_out[still_bad] = griddata(pts_valid, u[valid_mask], (x[still_bad], y[still_bad]), method="nearest")
        v_out[still_bad] = griddata(pts_valid, v[valid_mask], (x[still_bad], y[still_bad]), method="nearest")
    return u_out, v_out


def smooth_vector_field(u, v, sigma):
    from scipy.ndimage import gaussian_filter
    mask = (~np.isnan(u)).astype(float)
    u0, v0 = np.nan_to_num(u), np.nan_to_num(v)
    wsum = np.clip(gaussian_filter(mask, sigma), 1e-8, None)
    return gaussian_filter(u0, sigma) / wsum, gaussian_filter(v0, sigma) / wsum


def reconstruct_stereo(dx1, dy1, dx2, dy2, alpha1, alpha2, beta1, beta2):
    """Combine two cameras' in-plane displacement fields (on a common
    dewarped grid) into 3-component displacement (dX, dY, dZ) by solving,
    in a least-squares sense:
        dx1 = dX          - dZ*tan(alpha1)
        dx2 = dX          - dZ*tan(alpha2)
        dy1 =       dY    - dZ*tan(beta1)
        dy2 =       dY    - dZ*tan(beta2)
    This handles degenerate cases automatically -- e.g. beta1 == beta2, as
    in a rig where the two cameras are tilted apart in only one plane (no
    relative vertical tilt): the y-pair becomes redundant rather than a
    divide-by-zero, and dZ is correctly identified from the x-pair alone.
    Verified against synthetic ground truth for the general case, this
    degenerate case, and per-point angle-map inputs (angles may be scalars
    or arrays matching dx1's shape)."""
    ta1, ta2 = np.tan(alpha1), np.tan(alpha2)
    tb1, tb2 = np.tan(beta1), np.tan(beta2)
    shape = np.broadcast(dx1, ta1, ta2, tb1, tb2).shape
    ta1b, ta2b, tb1b, tb2b = (np.broadcast_to(np.asarray(a, dtype=float), shape)
                               for a in (ta1, ta2, tb1, tb2))
    zeros, ones = np.zeros(shape), np.ones(shape)
    A = np.stack([
        np.stack([ones,  zeros, -ta1b], axis=-1),
        np.stack([ones,  zeros, -ta2b], axis=-1),
        np.stack([zeros, ones,  -tb1b], axis=-1),
        np.stack([zeros, ones,  -tb2b], axis=-1),
    ], axis=-2)  # (..., 4, 3)
    b = np.stack(np.broadcast_arrays(dx1, dx2, dy1, dy2), axis=-1)  # (..., 4)
    AtA = np.einsum('...ki,...kj->...ij', A, A)   # (..., 3, 3)
    Atb = np.einsum('...ki,...k->...i', A, b)     # (..., 3)
    sol = np.linalg.solve(AtA, Atb[..., None])[..., 0]  # (..., 3)
    return sol[..., 0], sol[..., 1], sol[..., 2]


def plot_and_save(x, y, U, V, W, valid, out_path, ctrl, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.quiver(x[valid], y[valid], U[valid], V[valid], W[valid],
                    cmap="coolwarm", scale=ctrl.quiver_scale)
    fig.colorbar(sc, ax=ax, label="W (out-of-plane)")
    ax.set_title(title)
    ax.set_xlabel("x (world px)")
    ax.set_ylabel("y (world px)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=ctrl.plot_dpi)
    if ctrl.show_plots:
        plt.show()
    plt.close(fig)


def process_camera(process, frame_a, frame_b, ctrl):
    """Run piv_gpu on one camera's already-dewarped pair and apply
    per-camera post-processing. Returns (u, v, valid, elapsed) in world
    px/frame."""
    t0 = time.time()
    u, v = process(frame_a, frame_b)
    free, total = cp.cuda.runtime.memGetInfo()
    if ctrl.verbose:
        print(f"GPU free: {free / 1024 ** 3:.2f} GB / {total / 1024 ** 3:.2f} GB")
    elapsed = time.time() - t0

    if ctrl.apply_v_sign_flip:
        v = -v

    valid = ~process.val_locations
    if ctrl.global_outlier_std is not None:
        valid = valid & ~global_outlier_mask(u, v, ctrl.global_outlier_std)

    u_out, v_out = u.copy(), v.copy()
    u_out[~valid] = np.nan
    v_out[~valid] = np.nan

    if ctrl.replace_invalid:
        x, y = process.coords
        u_out, v_out = replace_invalid_vectors(x, y, u_out, v_out, valid)

    if ctrl.smooth_field:
        u_out, v_out = smooth_vector_field(u_out, v_out, ctrl.smooth_sigma)

    return u_out, v_out, valid, elapsed


def init_processor(frame_shape, ctrl):
    check_piv_settings(ctrl.piv_settings)
    process = piv_gpu(frame_shape, ctrl.min_search_size, **ctrl.piv_settings)
    x, y = process.coords
    y = frame_shape[0] * process.scaling_par - y
    return process, x, y


def run_camera(frame_a, frame_b, ctrl):
    """Build a fresh piv_gpu instance for ONE camera's dewarped pair, run
    it, then free its GPU memory before returning. This keeps only one
    camera's correlation buffers resident on the GPU at a time instead of
    holding both cameras' piv_gpu instances simultaneously -- roughly
    halves peak VRAM, at the cost of rebuilding FFT plans for every
    camera, every pair (no plan reuse across the run). Worth it on a
    memory-constrained GPU; if VRAM isn't the bottleneck, hoist
    init_processor() calls back out to main() and reuse the same process0/
    process1 across all pairs instead."""
    process, x, y = init_processor(frame_a.shape, ctrl)
    u, v, valid, elapsed = process_camera(process, frame_a, frame_b, ctrl)
    del process
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return u, v, valid, elapsed, x, y


def handle_pair(pair_id, dw_a0, dw_b0, dw_a1, dw_b1, ctrl, angles):
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

    U, V, W = reconstruct_stereo(u1_mm, v1_mm, u2_mm, v2_mm,
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
        np.savez(os.path.join(ctrl.output_dir, f"{pair_id}_stereo_velocity.npz"),
                 x=x, y=y, U=U, V=V, W=W, valid=valid)

    if ctrl.save_plot:
        plot_and_save(x, y, U, V, W, valid,
                      os.path.join(ctrl.output_dir, f"{pair_id}_stereo_quiver.png"),
                      ctrl, title=f"Stereo PIV -- {pair_id}")

    return (pair_id, elapsed, n_valid, n_total)


# ======================================================================
# Main
# ======================================================================
def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    ctrl = load_controls(config_path)
    os.makedirs(ctrl.output_dir, exist_ok=True)

    if ctrl.input_mode == "set":
        pair_source = iter_stereo_from_set(ctrl)
    elif ctrl.input_mode == "loose":
        pair_source = iter_stereo_from_loose_files(ctrl)
    else:
        sys.exit(f"Unknown input_mode: {ctrl.input_mode!r} (use 'set' or 'loose')")

    angles = (np.deg2rad(ctrl.alpha1_deg), np.deg2rad(ctrl.alpha2_deg),
              np.deg2rad(ctrl.beta1_deg), np.deg2rad(ctrl.beta2_deg))

    summary_rows = []

    for pair_id, fa0, fb0, fa1, fb1 in pair_source:
        # dewarp both frames of both cameras onto the shared world grid
        dw_a0 = ctrl.cam0_mapping.dewarp_image(fa0, ctrl.world_shape, ctrl.dewarp_order)
        dw_b0 = ctrl.cam0_mapping.dewarp_image(fb0, ctrl.world_shape, ctrl.dewarp_order)
        dw_a1 = ctrl.cam1_mapping.dewarp_image(fa1, ctrl.world_shape, ctrl.dewarp_order)
        dw_b1 = ctrl.cam1_mapping.dewarp_image(fb1, ctrl.world_shape, ctrl.dewarp_order)

        # each camera's piv_gpu instance is built and torn down inside
        # handle_pair()/run_camera() -- see run_camera()'s docstring for
        # why (peak VRAM vs. speed tradeoff)
        summary_rows.append(handle_pair(pair_id, dw_a0, dw_b0, dw_a1, dw_b1,
                                          ctrl, angles))

    if not summary_rows:
        sys.exit("No stereo pairs were processed -- check input_mode/input_path")

    if ctrl.save_summary_csv:
        csv_path = os.path.join(ctrl.output_dir, "stereo_processing_summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_id", "process_time_s", "n_valid", "n_total"])
            writer.writerows(summary_rows)
        print(f"Summary written to {csv_path}")

    total_time = sum(row[1] for row in summary_rows)
    print(f"Done: {len(summary_rows)} pair(s) in {total_time:.3f} s "
          f"({total_time / len(summary_rows):.3f} s/pair average)")


if __name__ == "__main__":
    main()
