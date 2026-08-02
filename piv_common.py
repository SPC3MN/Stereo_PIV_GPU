"""
Shared helpers for this project's batch PIV pipelines:
  Planar.py                  -- GPU planar   (openpiv_gpu)
  Stereo-PIV.py               -- GPU stereo   (openpiv_gpu, uses stereo_common.py too)
  CPU_Planar_Processing.py    -- CPU planar   (openpiv-python)
  CPU_Stereo_Processing.py    -- CPU stereo   (openpiv-python, uses stereo_common.py too)

This module covers what's generic across ALL four: JSON config loading,
displacement post-processing (outlier rejection / invalid-vector
interpolation / smoothing), the two PIV "engine" adapters (GPU/CPU), plain
(non-stereo) im7 frame extraction, single-set-vs-folder-of-sets input
resolution, and the first-snapshot preview/confirm step used by single-set
runs. Stereo-specific pieces (camera calibration/dewarping, stereo frame
splitting, 3-component reconstruction) live in stereo_common.py instead.

Both PIV engine adapters expose the same minimal interface the pipelines
rely on: `.coords` (x, y arrays), `.val_locations` (bool array, True =
invalid, set after a call), `.scaling_par`, and being callable as
process(frame_a, frame_b) -> (u, v). This lets process_frames() below stay
identical regardless of which engine built `process`.
"""

import os
import sys
import glob
import json
import time
import dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# Config loading
# ======================================================================
def load_controls(config_path, default_config, controls_cls, on_loaded=None):
    """Load pipeline settings from a JSON config file onto controls_cls,
    creating the file (populated with default_config) if it doesn't exist
    yet. Keys present in the file override default_config's; anything the
    file doesn't mention falls back to the default -- only the settings
    being changed need to be in the file.

    on_loaded(config, controls_cls), if given, runs after every key has
    been set as an attribute, for pipeline-specific fixups (e.g. building
    CameraMapping objects, tuple-ifying world_shape)."""
    config = dict(default_config)
    if os.path.exists(config_path):
        with open(config_path) as f:
            user_config = json.load(f)
        config.update(user_config)
        print(f"[info] loaded config from '{config_path}'")
    else:
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        print(f"[info] '{config_path}' didn't exist -- wrote defaults there. "
              "Edit it (input_path, piv settings, etc.) and re-run to "
              "customize.")

    for key, val in config.items():
        setattr(controls_cls, key, val)

    if on_loaded is not None:
        on_loaded(config, controls_cls)

    return controls_cls


# ======================================================================
# Shared displacement post-processing
# ======================================================================
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


def apply_calibration(u, v, ctrl):
    """Planar px/frame -> physical units, if pixel_pitch_mm and frame_dt_s
    are both set in the config; otherwise a no-op (stays px/frame)."""
    if getattr(ctrl, "pixel_pitch_mm", None) is None or getattr(ctrl, "frame_dt_s", None) is None:
        return u, v
    scale = (ctrl.pixel_pitch_mm / 1000.0) / ctrl.frame_dt_s
    return u * scale, v * scale


def process_frames(process, frame_a, frame_b, ctrl, report_gpu_mem=False):
    """Run one engine (`process`, from init_gpu_processor/init_cpu_processor)
    on a frame pair and apply the shared outlier/invalid/smoothing
    post-processing. Returns (u, v, valid, elapsed) in px/frame -- any
    further calibration (planar) or stereo combination happens in the
    caller."""
    t0 = time.time()
    u, v = process(frame_a, frame_b)
    if report_gpu_mem and ctrl.verbose:
        gpu_free_report()
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


# ======================================================================
# GPU engine (openpiv_gpu.piv_gpu) -- lazy-imported so the CPU pipelines
# never need cupy/openpiv_gpu installed at all.
# ======================================================================
# The real keyword names piv_gpu.__init__(frame_shape, min_search_size,
# **kwargs) accepts -- min_search_size itself is a separate required
# positional arg, NOT one of these. piv_gpu already ignores unrecognized
# kwargs safely, so this set exists only to WARN about likely typos in
# piv_settings, not to filter/drop anything.
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


def init_gpu_processor(frame_shape, min_search_size, piv_settings):
    from openpiv_gpu.gpu_process import piv_gpu
    check_piv_settings(piv_settings)
    # piv_gpu asserts isinstance(..., tuple) on sequence-valued settings
    # (search_size_iters, overlap_ratio, ...) -- JSON only has lists, so
    # anything loaded from the config file needs converting back to a
    # tuple, or piv_gpu rejects it even though the values are correct.
    piv_settings = {k: (tuple(v) if isinstance(v, list) else v) for k, v in piv_settings.items()}
    process = piv_gpu(frame_shape, min_search_size, **piv_settings)
    x, y = process.coords
    y = frame_shape[0] * process.scaling_par - y
    return process, x, y


def gpu_free_report():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"GPU free: {free / 1024 ** 3:.2f} GB / {total / 1024 ** 3:.2f} GB")


def free_gpu_pools():
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ======================================================================
# CPU engine -- multi-pass, window-deformation openpiv-python processing
# ======================================================================
class CPUPIVProcess:
    """Adapter around openpiv-python's own multi-pass pipeline
    (`openpiv.windef.first_pass` / `multipass_img_deform`, driven by an
    `openpiv.settings.PIVSettings` object), shaped to match the small
    subset of piv_gpu's interface the pipelines rely on (.coords,
    .val_locations, .scaling_par, and being callable as
    process(frame_a, frame_b) -> (u, v)) so process_frames() above doesn't
    need to branch on backend.

    This replicates the per-pair body of `openpiv.windef.piv()` (coarse
    grid, decreasing window size per pass, image deformation between
    passes, sig2noise/global/median validation, iterative outlier
    replacement, optional smoothn) directly on in-memory frame arrays --
    i.e. the same multi-pass + validation + replacement feature set as
    piv_gpu, just via openpiv-python's implementation of it instead of
    piv_gpu's own. cpu_settings keys are `PIVSettings` field names (e.g.
    windowsizes, overlap, sig2noise_threshold, filter_method, smoothn,
    ...) -- see openpiv.settings.PIVSettings for the full list; unknown
    keys are warned about, not silently dropped."""

    def __init__(self, frame_shape, **cpu_settings):
        from openpiv.settings import PIVSettings
        from openpiv.pyprocess import get_rect_coordinates

        settings = PIVSettings()
        valid_fields = {f.name for f in dataclasses.fields(settings)}
        unknown = sorted(set(cpu_settings) - valid_fields)
        if unknown:
            print(f"[warn] cpu_settings has keys PIVSettings won't recognize: "
                  f"{unknown} -- check spelling against "
                  "openpiv.settings.PIVSettings's fields")
        for key, val in cpu_settings.items():
            if key in valid_fields:
                setattr(settings, key, val)

        settings.windowsizes = tuple(settings.windowsizes)
        settings.overlap = tuple(settings.overlap)
        if len(settings.overlap) != len(settings.windowsizes):
            raise ValueError(
                f"cpu_settings.overlap (length {len(settings.overlap)}) must "
                f"have the same length as windowsizes (length "
                f"{len(settings.windowsizes)}) -- one entry per pass"
            )
        settings.num_iterations = len(settings.windowsizes)

        self._settings = settings
        self.scaling_par = 1.0
        self.coords = get_rect_coordinates(frame_shape, settings.windowsizes[-1], settings.overlap[-1])
        self.val_locations = None

    def __call__(self, frame_a, frame_b):
        from openpiv import windef, validation, filters

        settings = self._settings
        frame_a = np.asarray(frame_a, dtype=np.float32)
        frame_b = np.asarray(frame_b, dtype=np.float32)

        # -- pass 0 (coarsest window) --
        x, y, u, v, s2n = windef.first_pass(frame_a, frame_b, settings)
        grid_mask = np.zeros_like(u, dtype=bool)
        u = np.ma.masked_array(u, mask=grid_mask)
        v = np.ma.masked_array(v, mask=grid_mask)

        if settings.validation_first_pass:
            flags = validation.typical_validation(u, v, s2n, settings)
        else:
            flags = np.zeros_like(u, dtype=bool)

        if (settings.num_iterations == 1 and settings.replace_vectors) or settings.num_iterations > 1:
            u, v = filters.replace_outliers(
                u, v, flags, method=settings.filter_method,
                max_iter=settings.max_filter_iteration,
                kernel_size=settings.filter_kernel_size,
            )

        if settings.smoothn:
            from openpiv import smoothn as _smoothn
            u, *_ = _smoothn.smoothn(u, s=settings.smoothn_p)
            v, *_ = _smoothn.smoothn(v, s=settings.smoothn_p)
            u = np.ma.masked_array(u, mask=grid_mask)
            v = np.ma.masked_array(v, mask=grid_mask)

        # -- passes 1..N-1 (decreasing window size, image deformation) --
        for i in range(1, settings.num_iterations):
            x, y, u, v, grid_mask, flags = windef.multipass_img_deform(
                frame_a, frame_b, i, x, y, u, v, settings)
            if settings.smoothn and i < settings.num_iterations - 1:
                from openpiv import smoothn as _smoothn
                u, *_ = _smoothn.smoothn(u, s=settings.smoothn_p)
                v, *_ = _smoothn.smoothn(v, s=settings.smoothn_p)
            u = np.ma.masked_array(u, np.ma.nomask)
            v = np.ma.masked_array(v, np.ma.nomask)

        u = np.ma.filled(u, 0.0)
        v = np.ma.filled(v, 0.0)
        u = u / settings.dt
        v = v / settings.dt

        # flags from the final (finest) pass -- True = invalid, same
        # convention as piv_gpu's val_locations
        self.val_locations = np.asarray(flags, dtype=bool)
        return u, v


def init_cpu_processor(frame_shape, cpu_settings):
    process = CPUPIVProcess(frame_shape, **cpu_settings)
    x, y = process.coords
    y = frame_shape[0] * process.scaling_par - y
    return process, x, y


# ======================================================================
# Plain (non-stereo) im7 -> numpy frame extraction, shared by the two
# planar pipelines
# ======================================================================
def frames_from_buffer(buf):
    """Pull raw intensity arrays for frame A and frame B out of a Buffer."""
    if len(buf.frames) < 2:
        raise ValueError(
            f"expected a double-frame buffer (2 frames), got {len(buf.frames)} "
            "-- your im7s likely store frame A/B as separate files; use "
            "input_mode='loose' with suffix_a/suffix_b set correctly"
        )
    frame_a = buf.frames[0].images[0]
    frame_b = buf.frames[1].images[0]
    return frame_a, frame_b


def iter_pairs_from_set(ctrl, set_path):
    """Yield (pair_id, frame_a, frame_b) from a DaVis image set."""
    import lvpyio as lv
    if lv.is_multiset(set_path):
        print(f"[info] '{set_path}' is a multi-set (e.g. multiple cameras) "
              f"-- using sub-set index {ctrl.multiset_index}")
        sets = lv.read_set(set_path)
        dataset = sets[ctrl.multiset_index]
        owns_dataset = False
    else:
        dataset = lv.read_set(set_path)
        owns_dataset = True

    try:
        n = len(dataset)
        for i in range(n):
            pair_id = f"{i:04d}"
            buf = dataset[i]
            frame_a, frame_b = frames_from_buffer(buf)
            yield pair_id, frame_a, frame_b
    finally:
        if owns_dataset:
            dataset.close()


def iter_pairs_from_loose_files(ctrl):
    """Yield (pair_id, frame_a, frame_b) from a plain folder of .im7 files."""
    import lvpyio as lv
    paths = sorted(glob.glob(os.path.join(ctrl.input_path, ctrl.loose_glob)))
    if not paths:
        sys.exit(f"No files matching '{ctrl.loose_glob}' in '{ctrl.input_path}'")

    first_buf = lv.read_buffer(paths[0])
    double_frame = len(first_buf.frames) >= 2

    if double_frame:
        for path in paths:
            pair_id = os.path.splitext(os.path.basename(path))[0]
            buf = lv.read_buffer(path)
            frame_a, frame_b = frames_from_buffer(buf)
            yield pair_id, frame_a, frame_b
    else:
        # frame A / frame B are separate single-frame files, matched by suffix
        files_a = sorted(p for p in paths if p.endswith(ctrl.suffix_a))
        if not files_a:
            sys.exit(
                f"Files are single-frame but none end in '{ctrl.suffix_a}' "
                "-- set suffix_a/suffix_b in the config to match your naming"
            )
        for path_a in files_a:
            path_b = path_a[: -len(ctrl.suffix_a)] + ctrl.suffix_b
            if not os.path.exists(path_b):
                print(f"[warn] no match for {os.path.basename(path_a)} "
                      f"(expected {os.path.basename(path_b)}) -- skipping")
                continue
            pair_id = os.path.basename(path_a)[: -len(ctrl.suffix_a)]
            frame_a = lv.read_buffer(path_a).frames[0].images[0]
            frame_b = lv.read_buffer(path_b).frames[0].images[0]
            yield pair_id, frame_a, frame_b


def plot_and_save_planar(x, y, u, v, valid, out_path, ctrl, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(x[valid], y[valid], u[valid], v[valid], color="red", scale=ctrl.quiver_scale)
    ax.set_title(title)
    ax.set_xlabel("pixels")
    ax.set_ylabel("pixels")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=ctrl.plot_dpi)
    if ctrl.show_plots:
        plt.show()
    plt.close(fig)


# ======================================================================
# Single-set vs. folder-of-sets input resolution
# ======================================================================
def resolve_set_paths(input_path):
    """Decide whether input_path is ONE DaVis set to process directly, or a
    folder holding MULTIPLE sets to batch through one after another.

    - a path ending in '.set' is always treated as a single set.
    - otherwise, if it's a directory containing nested '*.set' entries,
      those are treated as the sets to batch over (folder-of-sets mode).
    - otherwise input_path itself is treated as the (single) set -- e.g. a
      raw DaVis project folder not named with a '.set' suffix.

    Returns (set_paths, is_batch)."""
    if input_path.lower().endswith(".set"):
        return [input_path], False
    if os.path.isdir(input_path):
        nested = sorted(glob.glob(os.path.join(input_path, "*.set")))
        if nested:
            return nested, True
    return [input_path], False


def set_label(set_path):
    """Short name for a set path, used for per-set output subfolders and
    logging -- strips a trailing '.set' if present."""
    base = os.path.basename(os.path.normpath(set_path))
    if base.lower().endswith(".set"):
        base = base[: -len(".set")]
    return base


# ======================================================================
# First-snapshot preview + confirmation (single-set runs only)
# ======================================================================
def preview_first_snapshot(png_path, prompt="Continue processing the remaining pairs in this set?"):
    """Open the just-saved preview PNG (OS default image viewer) and block
    on a y/n prompt at the terminal before the caller proceeds with the
    rest of a single-set run. Exits the process if the user declines."""
    print(f"[info] first-snapshot preview saved to '{png_path}'")
    try:
        if sys.platform == "win32":
            os.startfile(png_path)
        else:
            import subprocess
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, png_path])
    except OSError as e:
        print(f"[warn] couldn't auto-open preview image ({e}) -- open it "
              f"manually: {png_path}")

    while True:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
        if answer in ("y", "yes"):
            return
        if answer in ("", "n", "no"):
            sys.exit("Aborted after reviewing the first snapshot.")
        print("Please answer 'y' or 'n'.")
