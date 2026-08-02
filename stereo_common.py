"""
Shared stereo-specific helpers for Stereo-PIV.py (GPU) and
CPU_Stereo_Processing.py (CPU) -- DaVis camera calibration/dewarping,
stereo im7 frame extraction/iteration, two-camera 3-component
reconstruction, and the stereo quiver plot. Everything backend-agnostic
(config loading, post-processing, engine adapters, set-folder resolution,
preview/confirm) lives in piv_common.py instead.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


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
        from scipy.ndimage import map_coordinates
        self._ensure_grid(world_shape)
        return map_coordinates(raw_image, [self._y_raw, self._x_raw],
                                order=order, mode="constant", cval=0.0)


# ======================================================================
# im7 -> numpy frame extraction (stereo)
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
    is an assumption (see stereo_frame_order in the config). Confirm it
    against your own set before trusting output; flip stereo_frame_order
    if the wrong pair of frames ends up dewarped by each camera's
    mapping."""
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


def iter_stereo_from_set(ctrl, set_path):
    """Yield (pair_id, fa0, fb0, fa1, fb1) from a DaVis stereo image set."""
    import lvpyio as lv
    if lv.is_multiset(set_path):
        print(f"[info] '{set_path}' is a multi-set -- using sub-set "
              f"index {ctrl.multiset_index}")
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
            fa0, fb0, fa1, fb1 = frames_from_stereo_buffer(buf, ctrl.stereo_frame_order)
            yield pair_id, fa0, fb0, fa1, fb1
    finally:
        if owns_dataset:
            dataset.close()


def iter_stereo_from_loose_files(ctrl):
    """Yield (pair_id, fa0, fb0, fa1, fb1) from a plain folder of .im7
    files -- either one combined 4-frame file per stereo pair, or each
    camera's double-frame pair as a separate file (auto-detected)."""
    import lvpyio as lv
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
# Two-camera combination
# ======================================================================
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
    Angles may be scalars or arrays matching dx1's shape."""
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


def plot_and_save_stereo(x, y, U, V, W, valid, out_path, ctrl, title):
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
