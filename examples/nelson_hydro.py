"""
Tracer integration of the Nelson network along gravitational-collapse simulation tracers.

Solves each tracer in datasets/grav_collapse/ twice, once with the environment held piecewise constant between hydro snapshots and once with it cubic-spline interpolated.

  - Photorates use the three-band radiation field (Network(pd_fac=True), see pdfac.py).
  - Charge conservation is enforced as in nelson_charge.py: x(e-) = sum of cation abundances throughout.

For each interpolation, the output is a single array with one row per (tracer, output time) and columns idx | t | environment | species, listed in columns.txt.  Output is written to examples/data/nelson_hydro/.

Run from the repository root or from within examples/:

    python examples/nelson_hydro.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfac
from examples.hydro import HydroData
from examples.nelson_charge import charges, electron_row_matrix
from parser import Network, load_abundances
from solver import QuadraticSolverTracer

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "nelson" / "abundances.in"
TRACE_FILE      = REPO_ROOT / "datasets" / "grav_collapse" / "R0.1_M6_trace_cells.npy"
TIME_FILE       = REPO_ROOT / "datasets" / "grav_collapse" / "R0.1_M6_time.npy"
SAVE_DIR = HERE / "data" / "nelson_hydro"

YEAR = 3600.0 * 24.0 * 365.25

N_TRACERS = 32
N_WORKERS = 32
PLOT_TRACER = 0       # which tracer to plot

# Tolerances for a scaled solve
RTOL      = 1e-3
ATOL      = 1e-6
MIN_SCALE = 1e-22

NSUB       = 5        # output points per hydro segment
CLIP_ABUND = 1e-28    # abundances are floored to this before saving
BAND_FLOOR = 1e-30    # pdfac takes log10 of the bands, so they must stay positive

INTERPOLATIONS = ("piecewise_constant", "cubic_spline")

# Environment columns, in the order HydroData(three_uv_bands=True) returns them
ENV_COLS = ["nH", "T", "Tgrain", "Av", "uv1", "uv3", "uv4"]
LOG_COLS = ["nH", "Av", "uv1", "uv3", "uv4"]     # spline-interpolated in log space

# ---------------------------------------------------------------------------
# Network, with charge conservation
#
# Kept at module level: the solver's worker processes import this file to find
# get_tensors, so everything it uses must be defined outside the __main__ block.
# ---------------------------------------------------------------------------

net = Network(grains=True, self_shielding=True, pd_fac=True)
net.load_from_disk(str(NETWORK_PATH))
net.drop_passive_species()

# R replaces the e- row of A and B with the sum of the cation rows, so that dx(e-)/dt = q . dx/dt and x(e-) = q . x holds for the whole run.
R = electron_row_matrix(net)


def get_tensors(env):
    A, B = net.get_operators_vectorized(env)
    return R @ A, R @ B


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    N  = len(net.species)
    ie = net.species_map["e-"]
    q  = charges(net)              # q @ x = total positive charge

    # -----------------------------------------------------------------------
    # Jacobian sparsity
    # -----------------------------------------------------------------------

    # The solver's fast Jacobian needs a fixed sparsity pattern that covers every nonzero of A and B.  After the charge fix, the e- row has a nonzero in every column that any cation row has, so add those columns to the network's own pattern.
    cation_rows = np.flatnonzero(q)

    def add_cation_columns_to_electron_row(rows, cols, n_cols):
        cation_cols = np.unique(cols[np.isin(rows, cation_rows)])
        keys = np.concatenate([rows * n_cols + cols, ie * n_cols + cation_cols])
        keys = np.unique(keys)
        return keys // n_cols, keys % n_cols

    ai, aj = net.get_A_structure()
    bi, bj, bk = net.get_B_structure()

    ai, aj = add_cation_columns_to_electron_row(ai, aj, N)
    bi, b_col = add_cation_columns_to_electron_row(bi, bj * N + bk, N * N)   # B column = j*N + k
    bj, bk = b_col // N, b_col % N

    jac_structure = (ai, aj, bi, bj, bk)

    # -----------------------------------------------------------------------
    # Initial conditions
    # -----------------------------------------------------------------------

    abund = load_abundances(str(ABUNDANCES_PATH))

    x0 = np.zeros(N)
    for name, val in abund.items():
        if name in net.species_map:
            x0[net.species_map[name]] = val

    x0[ie] = q @ x0                # neutral at t = 0

    print("Network")
    print(f"  species   = {N}")
    print(f"  reactions = {len(net.reactions)}")
    print(f"  x0(e-)    = {x0[ie]:.3e}")

    # -----------------------------------------------------------------------
    # Load tracers
    # -----------------------------------------------------------------------

    t_kyr = np.load(TIME_FILE)
    dt_hydro = (t_kyr[1] - t_kyr[0]) * 1e3 * YEAR

    h = HydroData(three_uv_bands=True)
    h.load_hydro_data(str(TRACE_FILE), max_tracers=N_TRACERS, dt_hydro=dt_hydro, abundances=ABUNDANCES_PATH)

    # pts has shape (N_TRACERS, n_snapshots, 7), columns ENV_COLS
    tracer_idx, pts = h.sample_tracers(m=N_TRACERS, indices=np.arange(N_TRACERS))

    band_idx = [ENV_COLS.index(c) for c in ("uv1", "uv3", "uv4")]
    pts[:, :, band_idx] = np.maximum(pts[:, :, band_idx], BAND_FLOOR)

    n_snapshots = pts.shape[1]
    n_segments  = n_snapshots - 1

    print("\nTracers")
    print(f"  tracers   = {N_TRACERS}")
    print(f"  snapshots = {n_snapshots}")
    print(f"  dt_hydro  = {dt_hydro / YEAR / 1e3:.3f} kyr")

    # -----------------------------------------------------------------------
    # Output time grid: NSUB evenly spaced points per segment, plus the final snapshot
    # -----------------------------------------------------------------------

    t_knots = np.arange(n_snapshots) * dt_hydro
    t_fine = [t_knots[i] + dt_hydro * np.arange(NSUB) / NSUB for i in range(n_segments)]
    t_fine = np.append(np.concatenate(t_fine), t_knots[-1])
    n_rows = t_fine.size           # output rows per tracer

    def env_on_grid(pt, interpolation):
        """Environment the chemistry saw at each output time, shape (n_rows, 7)."""
        env = np.empty((n_rows, len(ENV_COLS)))

        if interpolation == "piecewise_constant":
            # Each segment holds the value at its left snapshot; the final row belongs to the last segment.
            env[:-1] = np.repeat(pt[:-1], NSUB, axis=0)
            env[-1] = pt[-2]
        else:
            for col, name in enumerate(ENV_COLS):
                if name in LOG_COLS:
                    spline = CubicSpline(t_knots, np.log(pt[:, col]))
                    env[:, col] = np.exp(spline(t_fine))
                else:
                    spline = CubicSpline(t_knots, pt[:, col])
                    env[:, col] = spline(t_fine)

        # With pd_fac=True the network ignores the hydro's Av and derives its own from the three bands; save that one.
        av_col = ENV_COLS.index("Av")
        bands = pdfac.spline_init(env[:, band_idx[0]], env[:, band_idx[1]], env[:, band_idx[2]])
        env[:, av_col] = pdfac.av(bands)
        return env

    # -----------------------------------------------------------------------
    # Tracer solves
    # -----------------------------------------------------------------------

    tracer = QuadraticSolverTracer()
    abundances = {}                # interpolation -> (N_TRACERS, n_rows, N)

    for interpolation in INTERPOLATIONS:
        print(f"\nSolving {N_TRACERS} tracers [{interpolation}] on {N_WORKERS} workers...")
        results, failures = tracer.solve_multiple(
            dt_hydro=dt_hydro,
            tracers=list(pts),
            get_tensors=get_tensors,
            x0=x0,
            atol=ATOL,
            rtol=RTOL,
            min_scale=MIN_SCALE,
            method="BDF",
            interpolation=interpolation,
            use_scaling=True,
            log_cols=LOG_COLS,
            t_eval=t_fine,
            jac_structure=jac_structure,
            n_workers=N_WORKERS,
            return_failures=True,
            verbose=True,
        )

        for i, exc in failures:
            print(f"  [WARN] tracer {tracer_idx[i]} failed: {exc}")

        y = np.full((N_TRACERS, n_rows, N), np.nan)      # failed tracers stay NaN
        for b, result in enumerate(results):
            if result is None:
                continue
            _, out_y = result

            if interpolation == "piecewise_constant":
                # One array per segment, each including both of its end snapshots.  Keep the first NSUB points of each segment, then the final snapshot.
                segments = [seg[:, :NSUB] for seg in out_y]
                final = out_y[-1][:, -1:]
                out_y = np.hstack(segments + [final])

            y[b] = out_y.T
        abundances[interpolation] = y

        # Feature matrix: one row per (tracer, output time), columns idx | t | environment (7) | species (N)
        env = np.stack([env_on_grid(pt, interpolation) for pt in pts])
        idx_col = np.repeat(tracer_idx, n_rows)
        t_col   = np.tile(t_fine, N_TRACERS)
        env_cols     = env.reshape(N_TRACERS * n_rows, len(ENV_COLS))
        species_cols = np.maximum(y.reshape(N_TRACERS * n_rows, N), CLIP_ABUND)
        feat = np.column_stack([idx_col, t_col, env_cols, species_cols])

        out_dir = SAVE_DIR / interpolation
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / TRACE_FILE.name
        np.save(out_path, feat)

        print(f"  converged = {N_TRACERS - len(failures)}/{N_TRACERS}")
        print(f"  saved     = {out_path}  {feat.shape}")

    # Column legend.  The Av column holds the band-derived Av, hence the name.
    column_names = ["idx", "t"] + ["Av_pdfac" if c == "Av" else c for c in ENV_COLS] + list(net.species)
    with open(SAVE_DIR / "columns.txt", "w", encoding="ascii") as f:
        f.write("\n".join(column_names))

    # -----------------------------------------------------------------------
    # Plot: every species for one tracer, both interpolations
    # -----------------------------------------------------------------------

    linestyles = {"piecewise_constant": "-", "cubic_spline": "--"}
    t_plot_kyr = t_fine / (1e3 * YEAR)
    plot_path = SAVE_DIR / f"nelson_hydro_tracer_{PLOT_TRACER}.pdf"

    fig, axes = plt.subplots(3, 5, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    for ax, species_name in zip(axes, net.species):
        i = net.species_map[species_name]
        for interpolation in INTERPOLATIONS:
            ax.plot(t_plot_kyr, abundances[interpolation][PLOT_TRACER, :, i],
                    linestyle=linestyles[interpolation], label=interpolation)
        ax.set_yscale("log")
        ax.set_title(species_name)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")           # 14 species on a 3x5 grid
    axes[0].legend(frameon=False, fontsize=7)
    for ax in axes[-5:]:
        ax.set_xlabel("Time (kyr)")
    fig.suptitle(f"Tracer {tracer_idx[PLOT_TRACER]}")
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlot: {plot_path}")
    print(f"All output written to: {SAVE_DIR}")
