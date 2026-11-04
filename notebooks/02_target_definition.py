# 02_target_definition — build and inspect the closure target δW = w_DNS − W_pf[η, ψ]
# v2 (2 Nov): W_pf from an exact conformal-mapping DtN; target resolution k_c = 20 k_p.
# VS Code / Jupyter: each "# %%" line starts a cell (VS Code can export this to .ipynb).
# Arrays are (time, space). Peak RAM ≈ 1–1.5 GB; full δW loop ≈ 1 min.

# %% [cell 1] imports & paths
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

G = 9.81
def find_root(marker="data"):
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / marker).is_dir():
            return p
    raise FileNotFoundError(f"No parent folder of {Path.cwd()} contains '{marker}/'")

ROOT = find_root()
DATA = ROOT / "data"

# %% [cell 2] loading
def parse_case_name(name):
    """'test_BK_0d224_1d073_bak_surf1_pert.mat' -> ('BK', [0.224, 1.073])."""
    parts = Path(name).stem.split("_")
    family = parts[1]
    params = [float(p.replace("d", ".")) for p in parts[2:] if p[0].isdigit() and "d" in p]
    return family, params

def load_case(path, Tp=None, fields=("eta", "ux", "uy")):
    """Return dict with fields as (t, x) float64 arrays plus grid and wave scales."""
    path = Path(path)
    if path.suffix == ".npz":
        d = np.load(path)
        arrs = {k: d[k].astype(np.float64) for k in fields}           # already (t, x)
        x, t = d["x"].ravel(), d["t"].ravel()
        case = str(d["case"])
    else:
        from pymatreader import read_mat
        d = read_mat(str(path), variable_names=["sfr3", "x1", "time_vec", "time_len_min", "case_name"])
        nt = int(d["time_len_min"])
        arrs = {k: np.asarray(d["sfr3"][k], dtype=np.float64)[:, :nt].T for k in fields}  # (x,t) -> (t,x)
        x, t = np.asarray(d["x1"]).ravel(), np.asarray(d["time_vec"]).ravel()[:nt]
        case = str(d.get("case_name", path.stem))
    if np.isnan(arrs["eta"]).all():
        raise ValueError(f"{path.name}: eta is entirely NaN (failed simulation)")
    N = x.size
    dx = (x[-1] - x[0]) / (N - 1)
    L = dx * N                                                         # periodic length
    family, params = parse_case_name(case if case.startswith("test_") else path.name)
    if Tp is None:
        if family in ("BK", "MXBK"):
            Tp = params[1]
        else:
            raise ValueError(f"Give Tp explicitly for family {family} (params {params})")
    omega = 2 * np.pi / Tp
    kp = omega**2 / G
    return dict(**arrs, x=x, t=t, N=N, dx=dx, L=L, dt=t[1] - t[0], case=case,
                family=family, params=params, Tp=Tp, omega=omega, kp=kp, cp=omega / kp)

# %% [cell 3] spectral helpers (periodic in x, last axis)
def wavenumbers(N, dx):
    return 2 * np.pi * np.fft.rfftfreq(N, d=dx)

def ddx(f, k):
    return np.fft.irfft(1j * k * np.fft.rfft(f, axis=-1), f.shape[-1], axis=-1)

def lowpass(f, k, kc):
    F = np.fft.rfft(f, axis=-1); F[..., k > kc] = 0
    return np.fft.irfft(F, f.shape[-1], axis=-1)

def integrate_x(q, k):
    """Periodic antiderivative; returns (psi with zero mean, mean of q)."""
    Q = np.fft.rfft(q, axis=-1)
    qbar = Q[..., 0].real / q.shape[-1]
    Q[..., 0] = 0
    Q[..., 1:] /= 1j * k[1:]
    return np.fft.irfft(Q, q.shape[-1], axis=-1), qbar

def ddt4(f, dt):
    """4th-order central difference along axis 0 (time); NaN at the 2 edge rows each side."""
    out = np.full_like(f, np.nan)
    out[2:-2] = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12 * dt)
    return out

# %% [cell 4] HOS Dirichlet-to-Neumann: W = phi_z at z = eta
def hos_vertical_velocity(eta, psi, k, M=4, depth=np.inf):
    N = eta.size
    th = np.ones_like(k) if np.isinf(depth) else np.tanh(k * depth)
    def dz(f, n):
        fac = k**n * (th if n % 2 else 1.0)
        return np.fft.irfft(np.fft.rfft(f) * fac, N)
    fact = np.cumprod([1.0] + list(range(1, M + 1)))
    phi = [psi]
    for m in range(2, M + 1):
        phi.append(-sum(eta**l / fact[l] * dz(phi[m - l - 1], l) for l in range(1, m)))
    return sum(eta**l / fact[l] * dz(phi[m - 1], l + 1)
               for m in range(1, M + 1) for l in range(M - m + 1))

# %% [cell 5] unit test of the HOS operator against an exact potential field
def test_hos(N=2048, steep=0.2):
    L = 2 * np.pi; x = np.arange(N) * L / N; k = wavenumbers(N, L / N)
    ks, amps = np.array([8, 10, 12, 15]), np.array([1, .8, .6, .3]); ph = np.array([0.3, 1.9, 4.0, 5.2])
    a = amps / amps.sum() * steep / ks.mean(); A = a * np.sqrt(G / ks)
    eta = sum(ai * np.cos(kk * x + p) for ai, kk, p in zip(a, ks, ph))
    psi = sum(Ai * np.exp(kk * eta) * np.sin(kk * x + p) for Ai, kk, p in zip(A, ks, ph))
    Wex = sum(Ai * kk * np.exp(kk * eta) * np.sin(kk * x + p) for Ai, kk, p in zip(A, ks, ph))
    for M in (2, 4, 6):
        err = np.abs(hos_vertical_velocity(eta, psi, k, M) - Wex).max() / np.abs(Wex).max()
        print(f"  M={M}: rel. max error {err:.1e}")

# %% [cell 5b] exact potential-flow DtN by conformal mapping (no small-slope expansion)
from scipy.interpolate import CubicSpline

def _hilbert(f, kk):
    """Hilbert transform with symbol -i*sign(k) (rfft wavenumbers kk >= 0)."""
    return np.fft.irfft(-1j * np.fft.rfft(f) * (kk > 0), f.size)

def _dxi(f, kk):
    return np.fft.irfft(1j * kk * np.fft.rfft(f), f.size)

def _periodic_interp(xg, fg, L, xq):
    """Periodic cubic-spline interpolation of fg(xg) at points xq."""
    xe = np.append(xg, xg[0] + L); fe = np.append(fg, fg[0])
    cs = CubicSpline(xe, fe, bc_type="periodic")
    return cs(xg[0] + np.mod(xq - xg[0], L))

def conformal_vertical_velocity(eta, psi_per, x, L, tol=1e-10, maxit=500, relax=0.5,
                                return_info=False):
    """Exact deep-water potential-flow W = phi_z at z = eta(x) given psi_per(x) (periodic
    surface potential, zero mean). Single-valued periodic surface; no small-slope expansion.
    Conformal map z(w) = w + i*h0 + F(w), F analytic & decaying in the lower half-plane."""
    N = x.size
    kk = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
    xi = x.copy()                                       # uniform grid in the mapped variable
    y = eta.copy()
    for it in range(maxit):                             # solve y(xi) = eta(xi + H[y](xi))
        xs = xi + _hilbert(y, kk)
        y_new = _periodic_interp(x, eta, L, xs)
        err = np.abs(y_new - y).max()
        y = (1 - relax) * y + relax * y_new
        if err < tol:
            break
    xs = xi + _hilbert(y, kk)
    Phi = _periodic_interp(x, psi_per, L, xs)           # surface potential at mapped points
    Psi = -_hilbert(Phi, kk)                            # harmonic conjugate on the boundary
    zxi = 1 + _dxi(xs - xi, kk) + 1j * _dxi(y, kk)      # dz/dxi
    fxi = _dxi(Phi, kk) + 1j * _dxi(Psi, kk)            # df/dxi
    cvel = fxi / zxi                                    # u - i w
    w_xi = -cvel.imag
    if np.any(np.diff(np.append(xs, xs[0] + L)) <= 0):
        raise ValueError("conformal map is not monotone in x (surface too steep / under-resolved)")
    W = _periodic_interp(xs, w_xi, L, x)               # back to the physical uniform grid
    return (W, dict(iterations=it + 1, residual=err)) if return_info else W


def test_conformal(N=4096):
    L = 2 * np.pi; x = np.arange(N) * L / N
    ks, amps, ph = np.array([8, 10, 12, 15]), np.array([1, .8, .6, .3]), np.array([0.3, 1.9, 4.0, 5.2])
    for steep in (0.4, 0.8, 1.2):
        a = amps / amps.sum() * steep / ks.mean(); A = a * np.sqrt(G / ks)
        eta = sum(ai * np.cos(kk * x + p) for ai, kk, p in zip(a, ks, ph))
        psi = sum(Ai * np.exp(kk * eta) * np.sin(kk * x + p) for Ai, kk, p in zip(A, ks, ph))
        Wex = sum(Ai * kk * np.exp(kk * eta) * np.sin(kk * x + p) for Ai, kk, p in zip(A, ks, ph))
        W = conformal_vertical_velocity(eta, psi - psi.mean(), x, L)
        print(f"  max slope {np.abs(np.gradient(eta, x)).max():.2f}: rel. max error {np.abs(W - Wex).max() / np.abs(Wex).max():.1e}")

print("conformal unit test:"); test_conformal()

# %% [cell 6] load one case (the .npz export, or the original .mat via pymatreader)
print("HOS unit test:"); test_hos()
case = load_case(DATA / "for_claude" / "BK_0d224_1d073_core.npz")
# case = load_case(DATA / "test_BK_0d224_1d073_bak_surf1_pert.mat")   # same data, slower, ~1.3 GB RAM
eta, ux, uy, t = case["eta"], case["ux"], case["uy"], case["t"]
k = wavenumbers(case["N"], case["dx"])
kp, cp = case["kp"], case["cp"]
print(f"{case['case']}: L={case['L']:.2f} m, dx={case['dx']*1e3:.2f} mm, Tp={case['Tp']} s, "
      f"kp={kp:.2f} rad/m, cp={cp:.3f} m/s, L/lambda_p={case['L']*kp/2/np.pi:.1f}")

# %% [cell 7] kinematic consistency: eta_t = a*w + b*u*eta_x  (expect a=1, b=-1)
eta_x = ddx(eta, k)
eta_t = ddt4(eta, case["dt"])
def kinematic_fit(i0, i1):
    A = np.column_stack([uy[i0:i1].ravel(), (ux * eta_x)[i0:i1].ravel()])
    y = eta_t[i0:i1].ravel()
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - (uy - ux * eta_x)[i0:i1].ravel()
    return a, b, np.sqrt(np.mean(r**2) / np.mean(y**2))
for lab, (t0, t1) in {"calm": (0.5, 4.5), "breaking": (6.9, 8.0)}.items():
    i0, i1 = np.searchsorted(t, [t0, t1])
    a, b, rel = kinematic_fit(i0, i1)
    print(f"kinematic fit {lab} {t0}-{t1}s: a={a:.3f}, b={b:.3f}, rel. residual={rel:.3f}")

# %% [cell 8] delta W for every snapshot: conformal (exact) DtN; HOS M=6 kept for comparison
KC = 20 * kp            # target resolution (simulator scale)
T_SKIP = 1.5            # discard start-up transient
nt, N = eta.shape
dW = np.full((nt, N), np.nan, np.float32)        # conformal (the target)
dW_hos = np.full((nt, N), np.nan, np.float32)    # HOS M=6 (diagnostic only)
slope_f = np.zeros((nt, N), np.float32)
for i in range(nt):
    e, w, u = (lowpass(f[i], k, KC) for f in (eta, uy, ux))
    ex = ddx(e, k)
    slope_f[i] = np.abs(ex)
    psi, _ = integrate_x(u + w * ex, k)
    dW_hos[i] = w - lowpass(hos_vertical_velocity(e, psi, k, 6), k, KC)
    try:
        dW[i] = w - conformal_vertical_velocity(e, psi, case["x"], case["L"])
    except ValueError:                           # map not monotone: overturning at this scale
        pass
dW_nd = dW / cp
B = ux / cp
valid = np.isfinite(dW) & (slope_f < 1.3) & (t[:, None] >= T_SKIP)
print(f"valid fraction: {valid.mean():.3f}; failed snapshots: {np.isnan(dW).all(1).sum()}")

# %% [cell 9] time summary
rms_w = np.sqrt((uy**2).mean(1))
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(t, B.max(1)); ax[0].axhline(0.85, ls="--", c="k"); ax[0].set_ylabel(r"max $\tilde u/c_p$")
ax[1].plot(t, np.abs(eta_x).max(1)); ax[1].set_ylabel(r"max $|\eta_x|$"); ax[1].set_yscale("log")
ax[2].plot(t, np.sqrt((dW**2).mean(1)) / rms_w, label="all x")
ax[2].plot(t, [np.sqrt((dW[i][valid[i]]**2).mean()) / rms_w[i] if valid[i].any() else np.nan
               for i in range(nt)], label="valid only")
ax[2].set_ylabel(r"rms $\delta W$ / rms $\tilde w$"); ax[2].set_xlabel("t [s]"); ax[2].legend()
fig.tight_layout(); plt.show()

# %% [cell 10] x-t maps zoomed on the wave group
env = np.abs(eta).max(0)                                    # max over time at each x
xin = case["x"][env > 0.05 * env.max()]
xlo, xhi = xin.min() - 2, xin.max() + 2                      # group footprint + 2 m padding
sel = (case["x"] >= xlo) & (case["x"] <= xhi)
xs = case["x"][sel]
ext = [xlo, xhi, t[0], t[-1]]
fig, ax = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
panels = [(np.where(valid, dW_nd, np.nan)[:, sel], r"$\delta W/c_p$ (valid)", "RdBu_r", (-0.1, 0.1)),
          (B[:, sel], r"$\tilde u/c_p$", "magma", (0, 1.2)),
          (eta[:, sel] * kp, r"$k_p\eta$", "viridis", (None, None))]
for a_, (Z, title, cmap, (lo, hi)) in zip(ax, panels):
    im = a_.imshow(Z, aspect="auto", origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi,
                   interpolation="nearest")
    a_.contour(xs, t, B[:, sel], levels=[0.85], colors="lime", linewidths=0.7)
    a_.set_title(title); a_.set_xlabel("x [m]"); fig.colorbar(im, ax=a_)
ax[0].set_ylabel("t [s]")
fig.suptitle("green contour: $\\tilde u/c_p = 0.85$")
fig.tight_layout(); plt.show()

# %% [cell 11] calm-floor diagnostic: peak-band transfer function w / W_pf over time
def transfer(i, band=(0.6, 1.5)):
    e, w, u = (lowpass(f[i], k, KC) for f in (eta, uy, ux))
    psi, _ = integrate_x(u + w * ddx(e, k), k)
    Fw, Fp = np.fft.rfft(w), np.fft.rfft(w - np.nan_to_num(dW[i]))
    m = (k > band[0] * kp) & (k < band[1] * kp)
    H = (Fw[m] * np.conj(Fp[m])).sum() / (np.abs(Fp[m])**2).sum()
    return abs(H), np.degrees(np.angle(H))
for tt in (0.5, 1, 2, 3, 4, 6, 9):
    amp, ph = transfer(np.searchsorted(t, tt))
    print(f"t={tt:4.1f}s  |w/W_pf|={amp:.3f}  phase={ph:+.1f} deg")
# %% [cell 12] stationary background: time-mean of δW at each x over a calm window
T0, T1 = 1.5, 4.0
win = (t >= T0) & (t <= T1)
dW_stat = np.nanmean(dW[win], axis=0)
grp = (case["x"] > xlo) & (case["x"] < xhi)
D = np.nan_to_num(dW[win][:, grp])
frac = 1 - ((D - dW_stat[grp])**2).sum() / (D**2).sum()
print(f"stationary part explains {frac:.2f} of calm δW variance in the group region")
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
ax[0].plot(case["x"][grp], dW_stat[grp] / cp); ax[0].set_ylabel(r"stationary $\delta W/c_p$")
ax[1].plot(case["x"][grp], ux[win][:, grp].mean(0)); ax[1].set_ylabel(r"time-mean $\tilde u$ [m/s]")
ax[1].set_xlabel("x [m]"); fig.tight_layout(); plt.show()

# %% [cell 13] crest δW: conformal vs HOS, against u/cp
ci = np.argmax(ux, axis=1)
rows = np.arange(nt)
crest_cm, crest_hos = dW[rows, ci] / cp, dW_hos[rows, ci] / cp
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(t, crest_cm, label="conformal (exact)"); ax[0].plot(t, crest_hos, alpha=0.6, label="HOS M=6")
ax[0].set_ylim(-0.5, 0.5); ax[0].set_xlabel("t [s]"); ax[0].set_ylabel(r"crest $\delta W/c_p$"); ax[0].legend()
ok = valid[rows, ci]
ax[1].scatter(B[rows, ci][ok], crest_cm[ok], s=6, c=t[ok], cmap="viridis")
ax[1].axvline(0.85, ls="--", c="k"); ax[1].set_xlabel(r"crest $\tilde u/c_p$"); ax[1].set_ylabel(r"crest $\delta W/c_p$ (valid)")
fig.tight_layout(); plt.show()