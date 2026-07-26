"""
Stereo PIV -- raw im7 input, dewarped via DaVis's own calibration polynomial
=============================================================================
Reads RAW (non-dewarped) .im7 pairs from two cameras, dewarps each camera's
images onto a shared world grid using the exact 3rd-order polynomial from
DaVis's calibration report ("Mapping of world (x'/y') to raw coordinates
(x/y)"), runs piv_gpu on each camera's dewarped pair, then combines the two
in-plane displacement fields into 3-component (U, V, W) velocity.

WHAT'S REAL VS. A PLACEHOLDER RIGHT NOW
----------------------------------------
- CAM0_MAPPING below is built from your actual "Plane 1" calibration report
  coefficients -- verified: world_to_raw() reproduces the same numbers
  computed by hand from your screenshot.
- CAM1_MAPPING is a PLACEHOLDER (clearly marked below) so the pipeline is
  exercisable end-to-end. Replace it with camera 2's ("Plane 2") actual
  coefficients from the same DaVis report panel before trusting any output.
- ANGLE1/ANGLE2 (alpha/beta, used only in the final combination step) are
  also placeholders. This single-Z-plane calibration doesn't carry Z
  sensitivity on its own -- that has to come from either (a) DaVis's
  reported per-camera viewing angle to the sheet normal specifically (not
  necessarily the "Min/max angle 1-2" figure, which reads like the angle
  BETWEEN the two cameras rather than either one's angle to the normal --
  worth confirming which it is), or (b) a second calibration at a different
  Z, differenced numerically the same way as the polynomial terms here.
  Don't trust W (or U/V, which also depend on these angles) until this is
  pinned down.

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
"""

import os
import sys
import csv
import time
import glob
import inspect
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# --- Camera 0 ("Plane 1"): your actual calibration report coefficients ---
CAM0_MAPPING = CameraMapping(
    x0=2806.99, x_span=4096.00, y0=1387.18, y_span=3008.00,
    dx_coefs=dict(**{'1': 804.1028, 's': 628.9870, 's2': 84.0572, 's3': -5.4234,
                      't': 3.1818, 't2': -0.3017, 't3': 0.2112,
                      'st': 0.6693, 's2t': -0.1956, 't2s': -0.4162}),
    dy_coefs=dict(**{'1': 28.8679, 's': 3.4689, 's2': -0.0086, 's3': -0.0561,
                      't': -1.2230, 't2': 1.3855, 't3': -0.9813,
                      'st': 89.0937, 's2t': -5.3036, 't2s': -0.2961}),
    name="cam0",
)

# --- Camera 1 ("Plane 2"): PLACEHOLDER -- replace with the real coefficients ---
CAM1_MAPPING = CameraMapping(
    x0=2806.99, x_span=4096.00, y0=1387.18, y_span=3008.00,
    dx_coefs=dict(**{'1': -780.0, 's': -610.0, 's2': 80.0, 's3': 5.0,
                      't': -3.0, 't2': 0.3, 't3': -0.2,
                      'st': -0.6, 's2t': 0.2, 't2s': 0.4}),
    dy_coefs=dict(**{'1': 30.0, 's': -3.2, 's2': 0.01, 's3': 0.05,
                      't': 1.3, 't2': -1.4, 't3': 0.98,
                      'st': -88.0, 's2t': 5.0, 't2s': 0.3}),
    name="cam1 (PLACEHOLDER -- not your real data)",
)


# ======================================================================
# CONTROLS
# ======================================================================
class CONTROLS:
    # ---------------- Input: RAW (non-dewarped) im7 per camera ----------
    cam0_input_path = "cam0_raw"
    cam1_input_path = "cam1_raw"
    multiset_index = 0

    # ---------------- Calibration mappings ----------
    cam0_mapping = CAM0_MAPPING
    cam1_mapping = CAM1_MAPPING

    # World/dewarped output grid, from DaVis's calibration report
    # ("Size of dewarped image" / "Scale factor"). (rows, cols).
    world_shape = (3067, 5874)
    world_scale_px_per_mm = 17.92
    dewarp_order = 1              # interpolation order for the warp (1=bilinear)

    # ---------------- PIV window size / passes / core settings ----------
    piv_kwargs = dict(
        min_search_size=32,
        search_size_iters=3,
        overlap_ratio=0.5,
        dt=1.0,
        validation_method="mean_velocity",
    )

    # ---------------- Per-camera post-processing (before combining) -------
    global_outlier_std = None
    replace_invalid = True
    smooth_field = False
    smooth_sigma = 1.0

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
    alpha1_deg = -89.53 / 2   # -44.765
    alpha2_deg = 89.53 / 2    # +44.765
    beta1_deg = 0.0
    beta2_deg = 0.0

    # ---------------- Units ----------------
    frame_dt_s = None            # s between frames; None keeps displacement units
    apply_v_sign_flip = True

    # ---------------- Output ----------------
    output_dir = "stereo_piv_output"
    save_npz = True
    save_plot = True
    save_summary_csv = True
    plot_dpi = 150
    quiver_scale = 1000
    show_plots = False
    verbose = True


# ======================================================================
# im7 reading (same auto-detecting behavior as the planar pipeline)
# ======================================================================
def frames_from_buffer(buf):
    if len(buf.frames) < 2:
        raise ValueError(f"expected a double-frame buffer, got {len(buf.frames)}")
    return buf.frames[0], buf.frames[1]


def open_camera_source(input_path, multiset_index):
    """Yield (pair_id, frame_a, frame_b) as lvpyio Frame objects (raw,
    NOT dewarped)."""
    try:
        if lv.is_multiset(input_path):
            print(f"[info] '{input_path}' is a multi-set -- using sub-set "
                  f"index {multiset_index}")
            dataset = lv.read_set(input_path)[multiset_index]
            owns = False
        else:
            dataset = lv.read_set(input_path)
            owns = True
        if len(dataset) > 0:
            print(f"[info] opened '{input_path}' as a DaVis set ({len(dataset)} buffers)")

            def gen():
                try:
                    for i in range(len(dataset)):
                        fa, fb = frames_from_buffer(dataset[i])
                        yield f"{i:04d}", fa, fb
                finally:
                    if owns:
                        dataset.close()
            return gen()
        if owns:
            dataset.close()
    except Exception as exc:
        print(f"[info] '{input_path}' isn't a DaVis set ({exc}) -- "
              "trying it as a plain folder / single buffer")

    if os.path.isdir(input_path):
        paths = sorted(glob.glob(os.path.join(input_path, "*.im7")))
        if not paths:
            sys.exit(f"No .im7 files found in '{input_path}'")

        def gen():
            for path in paths:
                pair_id = os.path.splitext(os.path.basename(path))[0]
                fa, fb = frames_from_buffer(lv.read_buffer(path))
                yield pair_id, fa, fb
        return gen()

    buf = lv.read_buffer(input_path)
    fa, fb = frames_from_buffer(buf)
    pair_id = os.path.splitext(os.path.basename(os.path.normpath(input_path)))[0]
    return iter([(pair_id, fa, fb)])


# ======================================================================
# Shared PIV / post-processing helpers
# ======================================================================
def filter_supported_kwargs(func, kwargs):
    sig = inspect.signature(func)
    accepted, dropped = {}, []
    for key, val in kwargs.items():
        if val is None:
            continue
        if key in sig.parameters:
            accepted[key] = val
        else:
            dropped.append(key)
    if dropped:
        print(f"[info] piv_gpu does not accept {dropped} -- dropped")
    return accepted


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


def process_one_camera(process, dewarped_a, dewarped_b, ctrl):
    """Run piv_gpu on one camera's already-dewarped pair and apply
    per-camera post-processing. Returns (u, v, valid) in world px/frame."""
    u, v = process(dewarped_a, dewarped_b)
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

    return u_out, v_out, valid


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


# ======================================================================
# Main
# ======================================================================
def main():
    ctrl = CONTROLS
    os.makedirs(ctrl.output_dir, exist_ok=True)

    src0 = open_camera_source(ctrl.cam0_input_path, ctrl.multiset_index)
    src1 = open_camera_source(ctrl.cam1_input_path, ctrl.multiset_index)

    alpha1, alpha2 = np.deg2rad(ctrl.alpha1_deg), np.deg2rad(ctrl.alpha2_deg)
    beta1, beta2 = np.deg2rad(ctrl.beta1_deg), np.deg2rad(ctrl.beta2_deg)

    process0 = process1 = None
    x = y = None
    summary_rows = []

    for (pid0, fa0, fb0), (pid1, fa1, fb1) in zip(src0, src1):
        if pid0 != pid1 and ctrl.verbose:
            print(f"[warn] pair id mismatch: cam0={pid0} cam1={pid1} "
                  "-- proceeding assuming matched order")

        if ctrl.verbose:
            print(f"Processing {pid0} ...", end=" ", flush=True)
        t0 = time.time()

        # dewarp both frames of both cameras onto the shared world grid
        dw_a0 = ctrl.cam0_mapping.dewarp_image(fa0.images[0], ctrl.world_shape, ctrl.dewarp_order)
        dw_b0 = ctrl.cam0_mapping.dewarp_image(fb0.images[0], ctrl.world_shape, ctrl.dewarp_order)
        dw_a1 = ctrl.cam1_mapping.dewarp_image(fa1.images[0], ctrl.world_shape, ctrl.dewarp_order)
        dw_b1 = ctrl.cam1_mapping.dewarp_image(fb1.images[0], ctrl.world_shape, ctrl.dewarp_order)

        if process0 is None:
            piv_kwargs = filter_supported_kwargs(piv_gpu.__init__, ctrl.piv_kwargs)
            process0 = piv_gpu(dw_a0.shape, **piv_kwargs)
            process1 = piv_gpu(dw_a1.shape, **piv_kwargs)
            x, y = process0.coords
            y = dw_a0.shape[0] * process0.scaling_par - y

        u1, v1, valid1 = process_one_camera(process0, dw_a0, dw_b0, ctrl)
        u2, v2, valid2 = process_one_camera(process1, dw_a1, dw_b1, ctrl)
        valid = valid1 & valid2

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

        elapsed = time.time() - t0
        n_valid, n_total = int(valid.sum()), int(valid.size)
        if ctrl.verbose:
            print(f"{elapsed:.3f} s, {n_valid}/{n_total} valid vectors")

        if ctrl.save_npz:
            np.savez(os.path.join(ctrl.output_dir, f"{pid0}_stereo_velocity.npz"),
                     x=x, y=y, U=U, V=V, W=W, valid=valid)
        if ctrl.save_plot:
            plot_and_save(x, y, U, V, W, valid,
                          os.path.join(ctrl.output_dir, f"{pid0}_stereo_quiver.png"),
                          ctrl, title=f"Stereo PIV -- {pid0}")

        summary_rows.append((pid0, elapsed, n_valid, n_total))

    if not summary_rows:
        sys.exit("No pairs were processed -- check cam0_input_path/cam1_input_path")

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