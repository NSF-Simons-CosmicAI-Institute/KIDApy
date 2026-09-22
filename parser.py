"""
Gas-phase + pseudo-grain reaction network parser.

Parses a KIDA-format ``reactions.dat`` file and builds the sparse polynomial ODE tensors ``A`` and ``B`` for the system

    dx/dt = A x + B(x ⊗ x)

where ``x`` is a vector of species abundances per H nucleus.

Supported chemistry
-------------------
* **Gas-phase reactions** — formula types 1–5 (cosmic-ray ionisation, UV photodissociation, Kooij/ionpol rate laws).
* **Pseudo-grain H₂ formation** (``grains=True``) — H₂ formation on grain surfaces via the intermediate pseudo-species XH:

    H  →  XH             (frml 11, adsorption pseudo-rate)
    XH + XH  →  H₂       (frml 10, recombination)

* **Ion–grain recombination** (``grains=True``) — ion recombination with grains (frml 0, itype 0), involving the pseudo-species GRAIN- and GRAIN0.

Full grain-surface chemistry (adsorption/desorption, surface reactions) is not supported by this module.

Quick start
-----------
    from parser import Network, load_abundances

    net = Network(grains=True)
    net.load_from_disk("reactions.dat")

    env = dict(T=10.0, nH=1e4, Av=1.0, uv_flux=1.0)  # uv_flux: 1 = standard Draine field
    A, B = net.get_operators(env)

References
----------
Wakelam, V. et al. (2012). A Kinetic Database for Astrochemistry (KIDA). ApJS, 199, 21. https://doi.org/10.1088/0067-0049/199/1/21

Wakelam, V. et al. (2024). The 2024 KIDA network for interstellar chemistry. A&A, 689, A63. https://doi.org/10.1051/0004-6361/202450606
"""

__all__ = ["Network", "load_abundances"]

import os
import warnings
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import Dict

import pdfac
import shielding


# ---------------------------------------------------------------------------
# Chemistry constants
# ---------------------------------------------------------------------------


grain_radius: float = 1.0e-5

#: Grain number density per H nucleus: n_d = grain_gas_ratio * nH.
#: Wakelam et al. (2012) require ``π a² n_d ≈ 1–2e-21 nH``, i.e. n_d ≈ 5e-12 nH. 
grain_gas_ratio: float = 5.0e-12

#: Cosmic-ray ionisation rate [s⁻¹]
zeta_cr: float = 1.6e-17


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _shield_channel(rxn: dict):
    """Return "H2", "CO", or None for a parsed reaction."""
    if rxn["frml"] != 2 or len(rxn["reactants"]) != 1:
        return None
    return _SHIELDED_CHANNELS.get(
        (rxn["reactants"][0], frozenset(rxn["products"])))


def _uv1(env: dict) -> float:
    """The 8–13.6 eV band from the environment, under either key name.

    ``uv_flux`` and ``uv1`` are the same quantity (tracer column 11); both are accepted so the key can be named consistently with ``uv3``/``uv4`` or left as it was.
    """
    for key in ("uv1", "uv_flux"):
        if key in env:
            return float(env[key])
    raise KeyError("environment needs 'uv1' (or 'uv_flux') for the 8-13.6 eV band")


def load_abundances(path: str) -> Dict[str, float]:
    """
    Load initial abundances from a two-column text file.

    Lines starting with ``#`` are treated as comments.  Each data line has the form::

        <species>  <abundance>

    The electron abundance ``e-`` must **not** appear in the file; it is computed externally from charge neutrality.

    Parameters
    ----------
    path : str
        Path to the abundances file.

    Returns
    -------
    dict
        Mapping ``species_name -> float``.
    """
    abund: Dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            abund[parts[0]] = float(parts[1])
    return abund


# ---------------------------------------------------------------------------
# Pseudo-grain species that are excluded when grains=False
# ---------------------------------------------------------------------------

_PSEUDO_GRAIN_SPECIES = frozenset({"XH", "GRAIN-", "GRAIN0"})

#: Self-shielded channels, ``(reactant, frozenset(products)) -> channel``.  Products are matched too, to exclude the photoionisation channels.
_SHIELDED_CHANNELS = {
    ("H2", frozenset({"H"})): "H2",
    ("CO", frozenset({"C", "O"})): "CO",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Network:
    """
    Gas-phase + pseudo-grain reaction network.

    Reads a KIDA-format reactions file and builds sparse ODE tensors A (N×N) and B (N×N²) such that

        dx/dt = A x + B(x ⊗ x)

    where x is the vector of species abundances per H nucleus and nH has been absorbed into the 2-body rate coefficients.  Column j*N+k of B encodes the reactant pair (j, k).

    Parameters
    ----------
    grains : bool
        False (default) — pure gas-phase network.  Reactions with frml 10–11 and frml 0/itype 0 return zero rates; XH, GRAIN-, and GRAIN0 are excluded from the species list.

        True — gas-phase plus pseudo-grain reactions.  H₂ formation via XH (frml 10/11) and ion–grain recombination via GRAIN-/GRAIN0 (frml 0, itype 0) are active.  Grain temperature is not required.

    See Also
    --------
    get_operators : Builds A and B; documents the environment dict.
    """

    def __init__(self, grains: bool = False, self_shielding: bool = False,
                 dust_attenuation: bool = False, pd_fac: bool = False):
        """
        Parameters
        ----------
        grains : bool
            Activate pseudo-grain reactions (H₂ formation via XH and ion–grain recombination via GRAIN-/GRAIN0).
        self_shielding : bool
            Apply Lee et al. (1996) H₂ and CO shielding factors to the H₂ and CO photodissociation rates.  See :mod:`shielding`.
        dust_attenuation : bool
            Include the ``exp(-γ·Av)`` dust term in frml-2 photoreaction rates.  Off by default, on the assumption that ``uv_flux`` is already attenuated; enabling it as well would count the dust twice.
        pd_fac : bool
            Take frml-2 rates from the local radiation field, ``k = α · PDfac(XVAL)``, instead of ``uv_flux``.  See :mod:`pdfac`.  The environment then needs ``uv1, uv3, uv4`` instead of ``uv_flux`` and ``Av``, and ``Av`` is derived from the field.  Mutually exclusive with ``dust_attenuation``.

            With ``self_shielding`` also on, the H₂ and CO channels are overwritten with ``α · θ`` rather than multiplied, and ``θ_dust`` is folded into the CO factor — reproducing ``H2SHIELD``/``COSHIELD``, where those two rates carry no PDfac.
        """
        if pd_fac and dust_attenuation:
            raise ValueError("pd_fac and dust_attenuation are mutually "
                             "exclusive: PDfac already replaces exp(-gamma*Av)")

        self.grains: bool = grains
        self.self_shielding: bool = self_shielding
        self.dust_attenuation: bool = dust_attenuation
        self.pd_fac: bool = pd_fac

        self.species: list = []
        self.species_map: dict = {}
        self.reactions: list = []

        # Fields treated as external forcing (not ODE variables)
        self.external_fields = frozenset({"Photon", "CR", "CRP"})

        self.max_order: int = 0
        self._warned_high_order: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_from_disk(self, filepath: str) -> None:
        """
        Parse a KIDA-format ``reactions.dat`` file and populate self.species, self.species_map, and self.reactions.

        Parameters
        ----------
        filepath : str
            Path to the reactions file.

        Raises
        ------
        FileNotFoundError
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Reactions file not found: '{filepath}'")
        # print(f"Reading: {filepath}")
        with open(filepath) as fh:
            self._parse_lines(fh.readlines())

    def get_operators(self, env: dict):
        """
        Build the sparse ODE tensors ``A`` and ``B`` for the current environment.

        Physical conditions are passed as a plain dict::

            env = dict(
                T        = 10.0,   # gas temperature [K]
                nH       = 1e4,    # total H number density [cm⁻³]
                Av       = 1.0,    # visual extinction [mag]
                uv_flux  = 1.0,    # FUV field scaling (1 = standard Draine field)
            )

        ``uv1`` is accepted as a synonym for ``uv_flux``; both name the 8–13.6 eV band.  Unrecognised keys are ignored.

        With ``pd_fac=True`` the radiation field is instead given by the three band energy densities ``uv1``, ``uv3`` and ``uv4``, and ``Av`` is derived from them — any ``Av`` in the dict is ignored.

        ``Tcap_2body`` (bool, default True) clamps T to [Tmin, Tmax] when evaluating Kooij / ionpol rate coefficients.

        Parameters
        ----------
        env : dict
            Required: T, nH, uv_flux (or uv1).  Also Av, unless pd_fac=True.
            Required when pd_fac=True: uv3, uv4.
            Optional: Tcap_2body (bool, default True).

        Returns
        -------
        A : scipy.sparse.csr_matrix, shape (N, N)
        B : scipy.sparse.csr_matrix, shape (N, N*N)
        """
        T = float(env["T"])
        nH = float(env["nH"])
        uv_flux = _uv1(env)
        Tcap_2body = bool(env.get("Tcap_2body", True))
        if self.pd_fac:
            sd = pdfac.spline_init(uv_flux, env["uv3"], env["uv4"])
            Av = float(pdfac.av(sd))
        else:
            sd = None
            Av = float(env["Av"])
        shield = (shielding.shield_factors(Av, dust=self.pd_fac)
                  if self.self_shielding else None)

        N = len(self.species)
        A_rows, A_cols, A_data = [], [], []
        B_rows, B_cols, B_data = [], [], []

        reactions = self._select_multirange_entries(self.reactions, T)

        for rxn in reactions:
            r_idxs = [self.species_map[r] for r in rxn["reactants"]
                      if r in self.species_map]
            p_idxs = [self.species_map[p] for p in rxn["products"]
                      if p in self.species_map]

            order = len(r_idxs)
            if order == 0:
                continue

            k = self._calculate_rate(rxn, T, nH, Av, uv_flux, Tcap_2body,
                                     shield, sd)
            if k == 0.0:
                continue

            if order == 1:
                r1 = r_idxs[0]
                A_rows.append(r1); A_cols.append(r1); A_data.append(-k)
                for p in p_idxs:
                    A_rows.append(p); A_cols.append(r1); A_data.append(k)

            elif order == 2:
                r1, r2 = r_idxs
                col = r1 * N + r2
                B_rows.append(r1); B_cols.append(col); B_data.append(-k)
                B_rows.append(r2); B_cols.append(col); B_data.append(-k)
                for p in p_idxs:
                    B_rows.append(p); B_cols.append(col); B_data.append(k)

            else:
                if not self._warned_high_order:
                    warnings.warn(
                        "Reactions with >2 active reactants detected and "
                        "skipped (no third-order tensor is built).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_high_order = True

        def _build(rows, cols, data, shape):
            if not data:
                return sp.csr_matrix(shape, dtype=np.float64)
            mat = sp.coo_matrix(
                (np.asarray(data, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=shape,
            )
            mat.sum_duplicates()
            return mat.tocsr()

        A = _build(A_rows, A_cols, A_data, (N, N))
        B = _build(B_rows, B_cols, B_data, (N, N * N))
        return A, B

    def get_operators_vectorized(self, env: dict):
        """
        Vectorized :meth:`get_operators`: identical ``A``, ``B`` (to roundoff), ~40x faster for repeated calls. Structure is cached on first use.

        Parameters
        ----------
        env : dict
            As for :meth:`get_operators`.

        Returns
        -------
        A : scipy.sparse.csr_matrix, shape (N, N)
        B : scipy.sparse.csr_matrix, shape (N, N*N)
        """
        N = len(self.species)

        # ---- Build environment-independent structure once, then cache it. ----
        # Only rate values depend on the environment; the sparsity pattern, contribution map and multi-range groups do not.
        if getattr(self, "_fast_N", None) != N:
            supported = {0, 1, 2, 3, 4, 5, 10, 11}   # formula types handled here
            two_body = (4, 5, 6, 7, 8, 11)            # itypes whose rate absorbs nH
            rx, sm = self.reactions, self.species_map
            col = lambda key: np.array([r[key] for r in rx], dtype=np.float64)
            self._fr = np.array([r["frml"] for r in rx])
            self._it = np.array([r["itype"] for r in rx])
            if {int(f) for f in np.unique(self._fr)} - supported:
                raise ValueError("get_operators_vectorized: unsupported formula type present")
            self._al, self._be, self._ga = col("alpha"), col("beta"), col("gamma")
            self._tn, self._tx = col("tmin"), col("tmax")
            self._norange = (self._tn <= -9000) & (self._tx >= 9000)
            self._twobody = np.isin(self._it, two_body) & np.isin(self._fr, [3, 4, 5])
            ch = [_shield_channel(r) for r in rx]
            self._sh_h2 = np.array([c == "H2" for c in ch])
            self._sh_co = np.array([c == "CO" for c in ch])
            # Effective wavelength per frml-2 reaction.  Depends only on the reactant and gamma
            self._xv = (np.array([pdfac.xval(r["reactants"][0], r["gamma"])
                                  if r["frml"] == 2 and r["reactants"] else np.nan
                                  for r in rx])
                        if self.pd_fac else None)

            def coalesce(contribs, ncol):  # (row, col, sign, reaction) -> fixed CSR + scatter map
                if not contribs:
                    z = np.empty(0)
                    return dict(idx=np.empty(0, np.int32), ptr=np.zeros(N + 1, np.int32),
                                inv=z.astype(int), nd=0, sign=z, rxn=z.astype(int), ncol=ncol)
                row, c, sign, rxn = (np.array(x) for x in zip(*contribs))
                u, inv = np.unique(row.astype(int) * ncol + c.astype(int), return_inverse=True)
                ptr = np.r_[0, np.cumsum(np.bincount(u // ncol, minlength=N))].astype(np.int32)
                return dict(idx=(u % ncol).astype(np.int32), ptr=ptr, inv=inv, nd=u.size,
                            sign=sign.astype(float), rxn=rxn.astype(int), ncol=ncol)

            # Enumerate (row, col, sign, reaction) contributions once:
            #   1-body -> A: loss on the reactant diagonal, gain for each product
            #   2-body -> B (column r1*N+r2): loss for both reactants, gain per product
            A, B, high = [], [], False
            for i, r in enumerate(rx):
                ri = [sm[s] for s in r["reactants"] if s in sm]
                pi = [sm[s] for s in r["products"] if s in sm]
                if len(ri) == 1:
                    A += [(ri[0], ri[0], -1.0, i)] + [(p, ri[0], 1.0, i) for p in pi]
                elif len(ri) == 2:
                    c = ri[0] * N + ri[1]
                    B += [(ri[0], c, -1.0, i), (ri[1], c, -1.0, i)] + [(p, c, 1.0, i) for p in pi]
                elif len(ri) >= 3:
                    high = True
            if high:
                warnings.warn("Reactions with >2 active reactants detected and skipped "
                              "(no third-order tensor is built).", RuntimeWarning, stacklevel=2)
            self._Astruct, self._Bstruct = coalesce(A, N), coalesce(B, N * N)

            # Group reactions differing only by temperature range; the per-call selection (below) keeps the single entry valid at T.
            grp = defaultdict(list)
            for i, r in enumerate(rx):
                grp[(tuple(r["reactants"]), tuple(r["products"]), r["itype"])].append(i)
            g = [m for m in grp.values() if len(m) > 1]
            self._mm = np.array([i for m in g for i in m], dtype=int)
            self._gid = np.array([j for j, m in enumerate(g) for _ in m], dtype=int)
            self._loc = np.array([k for m in g for k in range(len(m))], dtype=float)
            self._fast_N = N

        # ---- Per call: rate coefficient for every reaction, vectorized. ----
        T, nH = float(env["T"]), float(env["nH"])
        uv, Tcap = _uv1(env), bool(env.get("Tcap_2body", True))
        if self.pd_fac:
            sd = pdfac.spline_init(uv, env["uv3"], env["uv4"])
            Av = float(pdfac.av(sd))
        else:
            sd = None
            Av = float(env["Av"])
        a, b, g, fr, it = self._al, self._be, self._ga, self._fr, self._it
        # Teff optionally clamped to each reaction's [tmin, tmax] (no clamp for -9999/9999)
        Teff = np.where(self._norange, T, np.clip(T, self._tn, self._tx)) if Tcap else np.full(a.shape, T)
        p = Teff > 0
        k = np.zeros(a.shape)
        m = fr == 1                          # cosmic-ray ionisation
        k[m] = a[m] * zeta_cr
        m = fr == 2                          # UV photodissociation
        if self.pd_fac:
            k[m] = a[m] * pdfac.pdfac(sd, self._xv[m])
        else:
            k[m] = a[m] * uv * (np.exp(-g[m] * Av) if self.dust_attenuation else 1.0)
        if self.self_shielding:
            f_h2, f_co = shielding.shield_factors(Av, dust=self.pd_fac)
            if self.pd_fac:
                # H2SHIELD/COSHIELD overwrite these two: theta only, no PDfac.
                k[self._sh_h2] = a[self._sh_h2] * f_h2
                k[self._sh_co] = a[self._sh_co] * f_co
            else:
                k[self._sh_h2] *= f_h2
                k[self._sh_co] *= f_co
        m = (fr == 3) & p                    # Kooij
        k[m] = a[m] * (Teff[m] / 300.0) ** b[m] * np.exp(-g[m] / Teff[m])
        m = (fr == 4) & p                    # ionpol1
        k[m] = a[m] * b[m] * (0.62 + 0.4767 * g[m] * np.sqrt(300.0 / Teff[m]))
        m = (fr == 5) & p                    # ionpol2
        s = np.sqrt(300.0 / Teff[m])
        k[m] = a[m] * b[m] * (1.0 + 0.0967 * g[m] * s + 28.501 * g[m] ** 2 / Teff[m])
        k[self._twobody] *= nH               # 2-body reactions absorb nH
        if self.grains:
            gd = grain_gas_ratio * nH
            m = (fr == 0) & (it == 0) & p    # ion-grain recombination
            k[m] = a[m] * (Teff[m] / 300.0) ** b[m] * nH
            m = (fr == 10) & p               # XH + XH -> H2
            k[m] = 8.64e6 * np.exp(-225.0 / Teff[m]) / gd * nH
            m = (fr == 11) & p               # H -> XH
            k[m] = a[m] * (Teff[m] / 300.0) ** b[m] * gd
        # Zero every multi-range entry except the one selected at T: order each group (in-range first, then by position; otherwise by nearest boundary) and keep the first member per group.
        if self._mm.size:
            tn, tx = self._tn[self._mm], self._tx[self._mm]
            ir = (tn <= T) & (T <= tx)
            o = np.lexsort((np.where(ir, self._loc, np.minimum(abs(T - tn), abs(T - tx))),
                            np.where(ir, 0, 1), self._gid))
            gs = self._gid[o]
            act = np.ones(fr.shape, dtype=bool)
            act[self._mm] = False
            act[self._mm[o[np.r_[True, gs[1:] != gs[:-1]]]]] = True
            k = k * act
        # Scatter signed rate contributions into the cached CSR pattern, summing per (row, col) slot with bincount.
        mk = lambda S: sp.csr_matrix(
            (np.bincount(S["inv"], weights=k[S["rxn"]] * S["sign"], minlength=S["nd"]),
             S["idx"], S["ptr"]), shape=(N, S["ncol"]))
        return mk(self._Astruct), mk(self._Bstruct)

    def get_A_structure(self):
        """
        Return the fixed COO index arrays ``(ai, aj)`` of the A matrix.

        The sparsity pattern is a property of the reaction graph alone: it is fixed by the 1-body reactions in the network (a loss on the reactant's diagonal, and a gain in the reactant's column for each product) and is independent of the environment (T, nH, Av, ...).  Rate *values* change with the environment; these *positions* do not.  No rate coefficients are evaluated here.

        The returned arrays span every 1-body reaction, so they are a superset of the nonzeros that ``get_operators`` produces in any single environment (where zero-rate reactions are dropped).

        Returns
        -------
        ai : np.ndarray of int64, shape (nnz,)
            Row indices (species receiving a net linear contribution).
        aj : np.ndarray of int64, shape (nnz,)
            Column indices (reactant species), with ``A[ai[n], aj[n]]`` the n-th nonzero entry after duplicate summation.
        """
        N = len(self.species)
        rows, cols = [], []
        for rxn in self.reactions:
            r_idxs = [self.species_map[r] for r in rxn["reactants"]
                      if r in self.species_map]
            p_idxs = [self.species_map[p] for p in rxn["products"]
                      if p in self.species_map]
            if len(r_idxs) != 1:
                continue
            r1 = r_idxs[0]
            for row in [r1] + p_idxs:
                rows.append(row)
                cols.append(r1)

        if not rows:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty

        dummy = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(N, N),
        )
        dummy.sum_duplicates()
        ai = dummy.row.astype(np.int64, copy=False)
        aj = dummy.col.astype(np.int64, copy=False)
        return ai, aj

    def get_B_structure(self):
        """
        Return the fixed COO index arrays ``(bi, bj, bk)`` of the B tensor.

        Like :meth:`get_A_structure`, the pattern is purely topological: it is fixed by which species appear as the two reactants / products of each 2-body reaction, and is independent of the environment.  No rate coefficients are evaluated here.

        Reactions sharing a reactant pair collapse onto the same B column and duplicate ``(row, col)`` entries are coalesced, so the returned arrays match the nonzero layout ``get_operators`` would produce if every 2-body reaction had a nonzero rate (a superset of any single environment).

        Returns
        -------
        bi : np.ndarray of int64, shape (nnz,)
            Row indices (species receiving a net quadratic contribution).
        bj, bk : np.ndarray of int64, shape (nnz,)
            Reactant-pair indices, with ``B[bi[n], bj[n]*N + bk[n]]`` the n-th nonzero entry after duplicate summation.
        """
        N = len(self.species)
        rows, cols = [], []
        for rxn in self.reactions:
            r_idxs = [self.species_map[r] for r in rxn["reactants"]
                      if r in self.species_map]
            p_idxs = [self.species_map[p] for p in rxn["products"]
                      if p in self.species_map]
            if len(r_idxs) != 2:
                continue
            r1, r2 = r_idxs
            col = r1 * N + r2
            for row in [r1, r2] + p_idxs:
                rows.append(row)
                cols.append(col)

        if not rows:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty

        dummy = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(N, N * N),
        )
        dummy.sum_duplicates()
        bi = dummy.row.astype(np.int64, copy=False)
        flat_col = dummy.col.astype(np.int64, copy=False)
        return bi, flat_col // N, flat_col % N

    def get_shielded_reactions(self) -> dict:
        """
        Return ``{"H2": [idx, ...], "CO": [idx, ...]}``, the indices into self.reactions of the self-shielded photodissociation channels.

        A channel absent from the network maps to an empty list.
        """
        out = {"H2": [], "CO": []}
        for i, rxn in enumerate(self.reactions):
            ch = _shield_channel(rxn)
            if ch is not None:
                out[ch].append(i)
        return out

    def get_passive_species(self) -> list:
        """
        Return species that never appear as a reactant (pure-sink species).

        These species contribute no columns to A or B, so removing them before calling get_operators is safe and reduces the ODE dimension.

        Returns
        -------
        list[str]
        """
        active = {s for r in self.reactions for s in r["reactants"]}
        return [s for s in self.species if s not in active]

    def drop_passive_species(self) -> list:
        """
        Remove passive species from self.species and self.species_map in-place.

        Returns
        -------
        dropped : list[str]
        """
        passive = set(self.get_passive_species())
        if not passive:
            return []
        self.species = [s for s in self.species if s not in passive]
        self.species_map = {s: i for i, s in enumerate(self.species)}
        return sorted(passive)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_lines(self, lines: list) -> None:
        """Parse raw lines from a KIDA reactions file."""
        found_species: set = set()
        parsed_reactions: list = []
        _warned_K = False
        _warned_J = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("*"):
                continue

            padded = line.ljust(150)

            r_slots = [
                padded[0:11].strip(),
                padded[11:22].strip(),
                padded[22:33].strip(),
            ]
            reactants = [r for r in r_slots if r]
            products_field = padded[34:89].strip()
            products = products_field.split() if products_field else []

            all_species = reactants + products

            if any(self._is_mantle(s) for s in all_species):
                if not _warned_K:
                    warnings.warn(
                        "K-prefix (mantle) species detected and skipped.",
                        RuntimeWarning, stacklevel=2,
                    )
                    _warned_K = True
                continue

            if any(s and s[0] == "J" and len(s) > 1 for s in all_species):
                if not _warned_J:
                    warnings.warn(
                        "J-prefix (grain-surface) species detected and skipped.",
                        RuntimeWarning, stacklevel=2,
                    )
                    _warned_J = True
                continue

            params = padded[89:].replace("D", "E").split()
            if len(params) < 10:
                continue

            try:
                alpha = float(params[0])
                beta = float(params[1])
                gamma = float(params[2])
                itype = int(float(params[6]))
                tmin = float(params[7])
                tmax = float(params[8])
                frml = int(float(params[9]))
                rid = int(float(params[10])) if len(params) > 10 else -1
            except Exception as exc:
                warnings.warn(f"Skipping malformed line ({exc}): {line!r}",
                              RuntimeWarning, stacklevel=2)
                continue

            active_reactants = [r for r in reactants
                                 if r not in self.external_fields]
            if len(active_reactants) == 2:
                active_reactants = sorted(active_reactants)
            self.max_order = max(self.max_order, len(active_reactants))

            # Collect species; exclude pseudo-grain species when grains=False
            for s in active_reactants + products:
                if s and s not in self.external_fields:
                    if s in _PSEUDO_GRAIN_SPECIES and not self.grains:
                        continue
                    found_species.add(s)

            parsed_reactions.append(dict(
                reactants=active_reactants,
                products=products,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                itype=itype,
                tmin=tmin,
                tmax=tmax,
                frml=frml,
                id=rid,
            ))

        self.species = sorted(found_species)
        self.species_map = {s: i for i, s in enumerate(self.species)}
        self.reactions = parsed_reactions

        # print(f"Loaded {len(self.reactions)} reactions, "
        #       f"{len(self.species)} species.")

    def _is_mantle(self, s: str) -> bool:
        """Return True if s is a mantle (K-prefix) species name."""
        if not s:
            return False
        # Exclude elemental potassium (K, K+, K-)
        if s in ("K",) or s.startswith("K+") or s.startswith("K-"):
            return False
        return s[0] == "K" and len(s) > 1

    def _select_multirange_entries(self, reactions: list, T: float) -> list:
        """
        For reactions with multiple temperature-range entries, keep only the entry whose [Tmin, Tmax] contains T, or the nearest one if T is out of range.  Single-entry reactions pass through unchanged.
        """
        groups: dict = defaultdict(list)
        for i, rxn in enumerate(reactions):
            key = (tuple(rxn["reactants"]), tuple(rxn["products"]), rxn["itype"])
            groups[key].append(i)

        selected: set = set()
        for indices in groups.values():
            if len(indices) == 1:
                selected.add(indices[0])
                continue
            in_range = [i for i in indices
                        if reactions[i]["tmin"] <= T <= reactions[i]["tmax"]]
            if in_range:
                selected.add(in_range[0])
            else:
                def _dist(i):
                    tmin, tmax = reactions[i]["tmin"], reactions[i]["tmax"]
                    return min(abs(T - tmin), abs(T - tmax))
                selected.add(min(indices, key=_dist))

        return [rxn for i, rxn in enumerate(reactions) if i in selected]

    def _calculate_rate(self, rxn: dict, T: float, nH: float,
                        Av: float, uv_flux: float,
                        Tcap_2body: bool, shield=None, sd=None) -> float:
        """Compute the effective scalar rate coefficient for a single reaction.

        For 2-body reactions nH is already absorbed, so that the contribution to dx/dt is k_eff * x_i * x_j with x in abundance-per-H units.

        ``shield``, when given, is the ``(f_H2, f_CO)`` pair from :func:`shielding.shield_factors`, applied to the matching frml-2 channel.

        ``sd``, when given, is the :class:`pdfac.SplineData` for the local field; frml-2 rates then come from PDfac instead of ``uv_flux``.

        Supported formula types
        -----------------------
        frml 1  — cosmic-ray ionisation / CR-induced photons
        frml 2  — external UV photodissociation
        frml 3  — Kooij: k = α (T/300)^β exp(−γ/T)
        frml 4  — ionpol1
        frml 5  — ionpol2
        frml 0, itype 0   — ion–grain recombination (active when grains=True)
        frml 10 — XH + XH → H₂  (grains=True)
        frml 11 — H → XH  (grains=True)
        """
        frml = rxn["frml"]
        itype = rxn["itype"]
        a, b, g = rxn["alpha"], rxn["beta"], rxn["gamma"]
        tmin, tmax = rxn["tmin"], rxn["tmax"]

        # --- effective temperature (optionally clamped to [Tmin, Tmax]) ---
        def _Teff() -> float:
            if not Tcap_2body:
                return T
            if tmin <= -9000 and tmax >= 9000:
                return T
            return max(tmin, min(T, tmax))

        # --- frml 1: cosmic-ray ionisation / CR-induced photons ---
        if frml == 1:
            return a * zeta_cr

        # --- frml 2: external UV photoreactions ---
        if frml == 2:
            if sd is not None:
                ch = _shield_channel(rxn)
                if shield is not None and ch is not None:
                    # H2SHIELD/COSHIELD overwrite these two: theta only, no PDfac.
                    return a * (shield[0] if ch == "H2" else shield[1])
                return a * float(pdfac.pdfac(sd, pdfac.xval(rxn["reactants"][0], g)))
            k = a * uv_flux
            if self.dust_attenuation:
                k *= np.exp(-g * Av)
            if shield is not None:
                ch = _shield_channel(rxn)
                if ch is not None:
                    k *= shield[0] if ch == "H2" else shield[1]
            return k

        # --- frml 3/4/5: Kooij / ionpol ---
        if frml in (3, 4, 5):
            Teff = _Teff()
            if Teff <= 0.0:
                return 0.0
            if frml == 3:
                kT = a * (Teff / 300.0) ** b * np.exp(-g / Teff)
            elif frml == 4:
                kT = a * b * (0.62 + 0.4767 * g * np.sqrt(300.0 / Teff))
            else:  # frml == 5
                sq = np.sqrt(300.0 / Teff)
                kT = a * b * (1.0 + 0.0967 * g * sq + 28.501 * g * g / Teff)
            # 2-body gas-phase: absorb nH
            if itype in (4, 5, 6, 7, 8, 11):
                return kT * nH
            return kT

        Teff = _Teff()
        if Teff <= 0.0:
            return 0.0

        # --- frml 0: ion–grain recombination (itype 0 is the only frml=0 reaction type present in a gas-phase reactions file) ---
        if frml == 0:
            if itype == 0:
                if self.grains:
                    return a * (Teff / 300.0) ** b * nH
                return 0.0
            raise ValueError(f"Unexpected frml=0 / itype={itype} in gas-phase file")

        # --- frml 10/11: pseudo-grain H₂ formation via XH ---
        if frml in (10, 11):
            if not self.grains:
                return 0.0
            grain_density = grain_gas_ratio * nH
            if frml == 10:
                # XH + XH → H₂  (Teff is gas temperature)
                return 8.64e6 * np.exp(-225.0 / Teff) / grain_density * nH
            # frml == 11: H → XH  (adsorption pseudo-rate)
            return a * (Teff / 300.0) ** b * grain_density 

        raise ValueError(f"Unsupported formula type: frml={frml}")
