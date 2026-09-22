"""
Enforce charge conservation for the electron in the Nelson network.

Six reactions in the Nelson & Langer (1999) set destroy a positive charge
without consuming an electron (the lumped CHx/OHx species short-circuit the ionic intermediate that would have recombined).  The paper never integrates the electron; it fixes it from

    n(e) = n(C+) + n(He+) + n(H3+) + n(M+) + n(HCO+).

Integrating e- as an ordinary species instead lets it drift far above the cation sum, which over-drives recombination and parks carbon as neutral C, not CO.

Since x(e-) = q.x is linear in the state, differentiating the condition gives dx_e/dt = q.dx/dt -- so we just overwrite the e- row of A and B with the charge-weighted sum of the other rows.  The quadratic form is unchanged, and q.x - x(e-) then stays at its initial value of zero for the whole integration.
    python examples/nelson_charge.py
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import Network, load_abundances
from solver import QuadraticSolver

HERE = Path(__file__).resolve().parent
NETWORK_PATH = HERE.parent / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = HERE.parent / "networks" / "nelson" / "abundances.in"

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3
T_END = 1e7 * YEAR

ENVIRONMENTS = {
    "shielded core": dict(T=10.0, nH=2e4, Av=15.0, uv_flux=1.0),
    "diffuse edge": dict(T=50.0, nH=1e2, Av=0.5, uv_flux=1.0),
}


def charges(net):
    """Cation charge per species, with the electron slot zeroed, so q @ x is the total positive charge per H."""
    q = np.array([s.count("+") for s in net.species], dtype=np.float64)
    q[net.species_map["e-"]] = 0.0
    return q


def electron_row_matrix(net):
    """Identity with the e- row replaced by the cation charges; left-multiplying A and B by this rewrites that one row."""
    N = len(net.species)
    ie = net.species_map["e-"]
    q = charges(net)
    cats = np.flatnonzero(q)
    keep = [i for i in range(N) if i != ie]
    return sp.csr_matrix(
        ([1.0] * len(keep) + list(q[cats]),
         (keep + [ie] * cats.size, keep + list(cats))),
        shape=(N, N))


def fix_electron_row(net, A, B, R=None):
    """Replace the electron's rate equation with the charge-conservation condition.

    Pass a prebuilt ``R`` from :func:`electron_row_matrix` on hot paths -- the
    tracer solver calls ``get_tensors`` every solver step, not once per hydro step.
    """
    if R is None:
        R = electron_row_matrix(net)
    return R @ A, R @ B


if __name__ == "__main__":
    net = Network(grains=False, dust_attenuation=True)
    net.load_from_disk(str(NETWORK_PATH))
    net.drop_passive_species()

    ie = net.species_map["e-"]
    q = charges(net)

    abund = load_abundances(str(ABUNDANCES_PATH))
    x0 = np.zeros(len(net.species))
    for name, val in abund.items():
        if name in net.species_map:
            x0[net.species_map[name]] = val
    x0[ie] = float(q @ x0)               # neutrality at t = 0

    t_eval = np.logspace(0.0, np.log10(T_END), 200)
    solver = QuadraticSolver()

    for label, env in ENVIRONMENTS.items():
        A, B = net.get_operators(env)
        runs = {}
        for tag, (Ax, Bx) in (("free", (A, B)),
                              ("fixed", fix_electron_row(net, A, B))):
            _, runs[tag] = solver.solve(Ax, Bx, t_span=(t_eval[0], t_eval[-1]),
                                        x0=x0, atol=ATOL, rtol=RTOL, t_eval=t_eval)

        print(f"\n=== {label}: {env} ===")
        print(f"{'t [yr]':>9} | {'x(e-)':>11} {'q.x':>11} {'ratio':>9}"
              f" | {'x(e-)':>11} {'q.x':>11} {'ratio':>7}")
        print(f"{'':>9} | {'------ free e- ------':^35} | {'------ fixed ------':^31}")
        for j in (0, 100, 150, 199):
            row = []
            for tag in ("free", "fixed"):
                xe = runs[tag][ie, j]
                cat = float(q @ runs[tag][:, j])
                row += [xe, cat, xe / cat]
            print(f"{t_eval[j] / YEAR:9.3g} | {row[0]:11.4e} {row[1]:11.4e} {row[2]:9.3f}"
                  f" | {row[3]:11.4e} {row[4]:11.4e} {row[5]:7.4f}")

        print(f"  at {T_END / YEAR:.3g} yr (free -> fixed):")
        for s in ("C+", "C", "CO", "HCO+", "H3+", "e-"):
            i = net.species_map[s]
            print(f"    {s:5s} {runs['free'][i, -1]:11.4e} -> {runs['fixed'][i, -1]:11.4e}")
