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


def _init_gpu_processor_raw(frame_shape, min_search_size, piv_settings):
    """Like init_gpu_processor(), but returns coords BEFORE the top-down-
    to-bottom-up y-flip -- used directly by the non-tiled path (which
    flips using the whole frame's height right away) and by the tiling
    code in run_tiled() (which needs each tile's coords in a shared,
    un-flipped global frame before it can stitch tiles together and flip
    ONCE using the full frame's height, not each tile's)."""
    from openpiv_gpu.gpu_process import piv_gpu
    check_piv_settings(piv_settings)
    # piv_gpu asserts isinstance(..., tuple) on sequence-valued settings
    # (search_size_iters, overlap_ratio, ...) -- JSON only has lists, so
    # anything loaded from the config file needs converting back to a
    # tuple, or piv_gpu rejects it even though the values are correct.
    piv_settings = {k: (tuple(v) if isinstance(v, list) else v) for k, v in piv_settings.items()}
    process = piv_gpu(frame_shape, min_search_size, **piv_settings)
    x, y = process.coords
    return process, x, y


def init_gpu_processor(frame_shape, min_search_size, piv_settings):
    process, x, y = _init_gpu_processor_raw(frame_shape, min_search_size, piv_settings)
    y = frame_shape[0] * process.scaling_par - y
    return process, x, y


def gpu_free_report():
    import cupy as cp
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"GPU free: {free / 1024 ** 3:.2f} GB / {total / 1024 ** 3:.2f} GB")


def free_gpu_pools():
    """Release cupy's memory pools back to the driver. Calls gc.collect()
    first -- piv_gpu instances hold internal reference cycles, so a bare
    `del process` doesn't actually drop their arrays' refcount to zero;
    free_all_blocks() only reclaims blocks whose Python objects are
    ALREADY garbage-collected, so without this, memory silently
    accumulates across repeated build/run/free cycles (very noticeable
    across many tiles or many stereo camera pairs) even though each
    individual `del process; free_gpu_pools()` call looks like it should
    have freed everything."""
    import gc
    import cupy as cp
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ======================================================================
# Spatial tiling -- process very large frames as a grid of smaller,
# halo-padded tiles instead of all at once, so peak GPU memory is bounded
# by ONE tile's window count rather than the whole frame's. Each tile's
# PIV engine is built, run, and freed before the next tile starts (same
# build/run/free-per-unit pattern the stereo pipeline already uses per
# camera per pair, just applied per tile too).
#
# WHY A HALO: a window sitting exactly on a tile's edge needs real image
# data around it for its search area, not a hard crop at the tile
# boundary -- each tile is read out padded by `margin_px` on every side
# (clipped at the real frame edges), so its OWN piv_gpu call sees enough
# context. After the correlation for a tile, vectors from that halo are
# discarded, keeping only vectors that fall inside the tile's exclusive
# "core" region -- every pixel of the original frame is covered by
# exactly one tile's core, so nothing is double-counted or skipped
# between neighboring tiles.
#
# WHY THE OUTPUT IS FLAT (not a grid): each tile's local window grid
# starts fresh at that tile's own origin, so neighboring tiles' kept
# vectors don't generally land on a single shared (ny, nx) lattice --
# the combined result is an unstructured point set (1-D x/y/u/v/valid
# arrays), not a rectilinear grid like the non-tiled path returns.
# Quiver plotting and griddata-based invalid-vector replacement both
# work fine on flat arrays; Gaussian-filter-based smoothing does not
# (it assumes a regular grid), so smooth_field is skipped for tiled
# output -- see process_frames_tiled().
# ======================================================================
def compute_tiles(shape, n_tiles_y, n_tiles_x, margin_px):
    """Split a (H, W) frame into an n_tiles_y x n_tiles_x grid of tiles.
    Each tile dict has:
      "row"/"col"  -- this tile's position in the tile grid
      "core"       -- (y0, y1, x0, x1) this tile's EXCLUSIVE region in
                       the frame's global pixel coordinates
      "padded"     -- (py0, py1, px0, px1) the core expanded by
                       margin_px on each side, clipped to the frame"""
    H, W = shape
    y_edges = np.linspace(0, H, n_tiles_y + 1).round().astype(int)
    x_edges = np.linspace(0, W, n_tiles_x + 1).round().astype(int)
    tiles = []
    for row in range(n_tiles_y):
        y0, y1 = int(y_edges[row]), int(y_edges[row + 1])
        py0, py1 = max(0, y0 - margin_px), min(H, y1 + margin_px)
        for col in range(n_tiles_x):
            x0, x1 = int(x_edges[col]), int(x_edges[col + 1])
            px0, px1 = max(0, x0 - margin_px), min(W, x1 + margin_px)
            tiles.append({
                "row": row, "col": col,
                "core": (y0, y1, x0, x1),
                "padded": (py0, py1, px0, px1),
            })
    return tiles


def default_tile_margin(min_search_size, piv_settings):
    """A safe default halo margin -- the coarsest pass's full window
    extent (window size doubles per level going up from min_search_size,
    same convention as piv_gpu itself), so windows near a tile's edge
    still see real image data across their whole search area rather than
    being clipped at the tile boundary."""
    search_size_iters = piv_settings.get("search_size_iters", 1)
    num_passes = 1 if isinstance(search_size_iters, int) else len(search_size_iters)
    return min_search_size * (2 ** (num_passes - 1))


def run_tiled(frame_a, frame_b, ctrl, init_raw_fn, n_tiles_y, n_tiles_x, margin_px,
              report_gpu_mem=False, free_pools_fn=None):
    """Run a PIV engine (built per-tile via init_raw_fn(tile_shape) ->
    (process, x, y), e.g. a partial application of
    _init_gpu_processor_raw) across spatial tiles of a large frame pair
    instead of the whole frame at once -- see the module-level comment
    above for the halo/core scheme and why the result is flat.

    Returns (x, y, u, v, valid, elapsed) -- valid here is directly from
    each tile's val_locations (True = invalid, inverted to the "valid"
    convention), BEFORE any outlier-std/replace/smooth post-processing;
    see process_frames_tiled() for the full pipeline including that."""
    H, W = frame_a.shape
    tiles = compute_tiles((H, W), n_tiles_y, n_tiles_x, margin_px)

    xs, ys, us, vs, valids = [], [], [], [], []
    elapsed_total = 0.0
    scaling_par = None

    for i, tile in enumerate(tiles):
        y0, y1, x0, x1 = tile["core"]
        py0, py1, px0, px1 = tile["padded"]
        tile_a = frame_a[py0:py1, px0:px1]
        tile_b = frame_b[py0:py1, px0:px1]

        process, tx, ty = init_raw_fn(tile_a.shape)
        if scaling_par is None:
            scaling_par = process.scaling_par

        t0 = time.time()
        u, v = process(tile_a, tile_b)
        elapsed_total += time.time() - t0
        if report_gpu_mem and ctrl.verbose:
            gpu_free_report()

        val_locations = np.asarray(process.val_locations)
        del process
        if free_pools_fn is not None:
            free_pools_fn()

        # tile-local (still un-flipped, image-row-order) -> global coords
        gx = tx + px0
        gy = ty + py0

        # keep only vectors whose GLOBAL location falls in this tile's
        # exclusive core -- discards the halo, which exists only so this
        # tile's own windows had real context, not to be double-counted
        keep = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)

        xs.append(gx[keep]); ys.append(gy[keep])
        us.append(u[keep]); vs.append(v[keep])
        valids.append(~val_locations[keep])

        if ctrl.verbose:
            print(f"  tile {i + 1}/{len(tiles)} (row {tile['row']}, col {tile['col']}): "
                  f"{int(keep.sum())} vectors, {elapsed_total:.3f}s cumulative")

    x = np.concatenate(xs)
    u = np.concatenate(us)
    v = np.concatenate(vs)
    valid = np.concatenate(valids)
    # single global y-flip using the FULL frame's height, not each tile's
    # -- matches init_gpu_processor()'s flip, just deferred until every
    # tile's coordinates are already in the same (global) frame
    y = H * scaling_par - np.concatenate(ys)

    return x, y, u, v, valid, elapsed_total


def process_frames_tiled(frame_a, frame_b, ctrl, init_raw_fn, n_tiles_y, n_tiles_x, margin_px,
                          report_gpu_mem=False, free_pools_fn=None):
    """Tiled counterpart to process_frames() -- runs the PIV engine tile
    by tile (run_tiled()) to bound peak GPU memory on very large frames,
    then applies the SAME outlier-std/invalid-replacement post-processing
    process_frames() does. Returns (x, y, u, v, valid, elapsed); unlike
    process_frames() (which doesn't return coords -- the caller already
    has them from a single init_gpu_processor() call reused across
    pairs), tiled coords are re-derived from tile geometry every call, so
    they're returned here too.

    smooth_field is INTENTIONALLY skipped for tiled output (with a
    warning) -- see the module-level comment above compute_tiles()."""
    x, y, u, v, valid_raw, elapsed = run_tiled(
        frame_a, frame_b, ctrl, init_raw_fn, n_tiles_y, n_tiles_x, margin_px,
        report_gpu_mem=report_gpu_mem, free_pools_fn=free_pools_fn,
    )

    if ctrl.apply_v_sign_flip:
        v = -v

    valid = valid_raw
    if ctrl.global_outlier_std is not None:
        valid = valid & ~global_outlier_mask(u, v, ctrl.global_outlier_std)

    u_out, v_out = u.copy(), v.copy()
    u_out[~valid] = np.nan
    v_out[~valid] = np.nan

    if ctrl.replace_invalid:
        u_out, v_out = replace_invalid_vectors(x, y, u_out, v_out, valid)

    if ctrl.smooth_field:
        print("[warn] smooth_field is ignored for tiled output -- Gaussian "
              "smoothing needs a regular (ny, nx) grid, and tiled results "
              "are an unstructured point set stitched from multiple tiles' "
              "own local grids instead")

    return x, y, u_out, v_out, valid, elapsed


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
