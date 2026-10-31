import numpy as np
from csaps import csaps
from scipy.signal import hilbert


SMOOTH_P_X = 0.9999
SMOOTH_P_T = 0.995
MASK_THRESHOLD = 0.04
MASK_TRIM_START = 200
MASK_TRIM_END = 150


def grad1(values, step, boundary="periodic"):
    """Apply the legacy fourth-order stencil, whose sign is opposite d/dx."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("grad1 expects a two-dimensional array")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be a finite positive number")
    if values.shape[1] < 5:
        raise ValueError("grad1 requires at least five columns")

    if boundary == "periodic":
        padded = np.concatenate((values[:, -2:], values, values[:, :2]), axis=1)
    elif boundary == "edge":
        padded = np.pad(values, ((0, 0), (2, 2)), mode="edge")
    else:
        raise ValueError("boundary must be 'periodic' or 'edge'")

    return (
        -padded[:, :-4]
        + 8 * padded[:, 1:-3]
        - 8 * padded[:, 3:-1]
        + padded[:, 4:]
    ) / (12 * step)


def spline_smooth_5(values, t, x):
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if values.ndim != 2 or values.shape != (x.size, t.size):
        raise ValueError("values must have shape (len(x), len(t))")

    values = csaps(x, values.T, x, smooth=SMOOTH_P_X).T
    values = csaps(t, values, t, smooth=SMOOTH_P_T)
    return csaps(x, values.T, x, smooth=SMOOTH_P_X).T


def check_periodicity_assumption(field, label="field"):
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("periodicity checks expect a two-dimensional field")
    left_val = field[0, :]
    right_val = field[-1, :]
    diff = np.abs(left_val - right_val)
    rel = diff / (np.abs(left_val) + np.abs(right_val) + 1e-12)
    print(
        f"[{label}] mean|left-right| = {diff.mean():.4g}, "
        f"mean relative diff = {rel.mean():.4g} "
        "(near 0 supports periodicity; large values suggest it's not periodic)"
    )


def build_case_dataset(sfr3, t, x, wave_period):
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if len(t) < 5 or len(x) < 5:
        raise ValueError("time and spatial grids must each contain at least five points")
    if not np.isfinite(wave_period) or wave_period <= 0:
        raise ValueError("wave_period must be a finite positive number")

    required_fields = ("eta", "ux", "uy", "dud_y", "dwd_y")
    fields = {}
    for key in required_fields:
        if key not in sfr3:
            raise KeyError(f"Required field {key!r} is missing from sfr3")
        values = np.asarray(sfr3[key], dtype=float)
        if values.shape != (x.size, t.size):
            raise ValueError(
                f"sfr3[{key!r}] has shape {values.shape}; expected {(x.size, t.size)}"
            )
        fields[key] = spline_smooth_5(values, t, x)

    dx = x[1] - x[0]
    dt = t[1] - t[0]
    if dx <= 0 or dt <= 0:
        raise ValueError("time and spatial grids must be strictly increasing")

    deta_dx = -grad1(np.asarray(sfr3["eta"], dtype=float).T, dx).T
    deta_dx_smooth = spline_smooth_5(deta_dx, t, x)
    deta_dx2 = -grad1(deta_dx.T, dx).T
    deta_dx2_smooth = spline_smooth_5(deta_dx2, t, x)

    ux_raw = np.asarray(sfr3["ux"], dtype=float)
    uy_raw = np.asarray(sfr3["uy"], dtype=float)
    dudx_smooth = spline_smooth_5(-grad1(ux_raw.T, dx).T, t, x)
    duydx_smooth = spline_smooth_5(-grad1(uy_raw.T, dx).T, t, x)
    du_dt_smooth = spline_smooth_5(-grad1(ux_raw, dt), t, x)

    rhs = (
        -9.81 * deta_dx_smooth
        - dudx_smooth * fields["ux"]
        - duydx_smooth * fields["uy"]
        - fields["uy"] * duydx_smooth * deta_dx_smooth**2
        - fields["uy"]**2 * deta_dx_smooth * deta_dx2_smooth
    )
    residual = du_dt_smooth - rhs

    mask = np.abs(residual) >= MASK_THRESHOLD * np.max(np.abs(residual))
    mask[:, :MASK_TRIM_START] = False
    mask[:, -MASK_TRIM_END:] = False

    eta_h_time = hilbert(fields["eta"], axis=1)
    eta_h_space = hilbert(fields["eta"], axis=0)
    k = (2 * np.pi / wave_period) ** 2 / 9.81
    omega = 2 * np.pi / wave_period
    n_points = int(mask.sum())

    features = np.column_stack(
        (
            deta_dx_smooth[mask],
            fields["uy"][mask],
            fields["ux"][mask],
            dudx_smooth[mask],
            fields["dud_y"][mask],
            duydx_smooth[mask],
            fields["dwd_y"][mask],
            deta_dx2[mask],
            np.imag(eta_h_time[mask]),
            np.imag(eta_h_space[mask]),
            np.abs(eta_h_time[mask]),
            np.abs(eta_h_space[mask]),
            fields["eta"][mask],
            np.full(n_points, k),
            np.full(n_points, omega),
        )
    )

    return {
        "features": features,
        "target_full": du_dt_smooth[mask],
        "target_residual": residual[mask],
        "mask": mask,
        "rhs": rhs,
        "residual_field": residual,
        "du_dt_smooth": du_dt_smooth,
        "deta_dx_smooth": deta_dx_smooth,
    }
