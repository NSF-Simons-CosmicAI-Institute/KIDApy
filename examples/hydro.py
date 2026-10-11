"""
Loading of hydrodynamic tracer trajectories.

:class:`HydroData` reads tracer files written by the hydrodynamics simulation and returns, for each tracer, the time series of physical parameters the chemistry needs (H nuclei density, gas and grain temperature, extinction, radiation field).  :func:`mass_per_H_nucleus` gives the factor that converts the simulation's mass density to H nuclei density.
"""

__all__ = ["HydroData", "mass_per_H_nucleus"]

import glob
import re
import warnings
from collections import defaultdict

import numpy as np


#: Default parameter layout.  ``uv_flux`` is the 8-13.6 eV radiation field.
PARAM_NAMES = ["nH", "T", "Tgrain", "Av", "uv_flux"]

#: Layout when ``three_uv_bands=True``.  ``uv_flux`` is renamed ``uv1`` so all three bands are named by their bin: 8-13.6, 3.44-8.0 and 0.41-3.44 eV.  :mod:`pdfac` needs all three.
BAND_PARAM_NAMES = ["nH", "T", "Tgrain", "Av", "uv1", "uv3", "uv4"]

#: Atomic masses [amu] of the elements appearing in the bundled networks.
_ATOMIC_MASS = {"H": 1.008, "He": 4.002602, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Fe": 55.845}

_AMU = 1.66053907e-24          # g

#: Pseudo-species that carry no gas-phase mass.
_NO_MASS = frozenset({"e-", "e", "GRAIN", "GRAIN0", "GRAIN-", "XH"})

#: Longest-first so that ``Cl`` is preferred over ``C``.
_ELEMENT_TOKEN = re.compile("(" + "|".join(sorted(_ATOMIC_MASS, key=len, reverse=True)) + r")(\d*)")


def _composition(path):
    """Element abundances per H nucleus from an ``abundances.in``.

    Returns ``(per_element, unparsed)``; ``unparsed`` maps any species whose formula could not be resolved to its (non-zero) abundance.
    """
    per_element = defaultdict(float)
    unparsed = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].split("!")[0].strip()
            if not line:
                continue
            fields = line.replace("=", " ").split()
            if len(fields) < 2:
                continue
            species = fields[0]
            try:
                abundance = float(fields[1].replace("D", "E").replace("d", "E"))
            except ValueError:
                continue
            if species in _NO_MASS:
                continue

            core = species.rstrip("+-")
            counts, pos = {}, 0
            for match in _ELEMENT_TOKEN.finditer(core):
                if match.start() != pos:          # gap => unrecognised token
                    break
                n = int(match.group(2)) if match.group(2) else 1
                counts[match.group(1)] = counts.get(match.group(1), 0) + n
                pos = match.end()
            if pos != len(core):
                if abundance:
                    unparsed[species] = abundance
                continue

            for element, n in counts.items():
                per_element[element] += n * abundance
    return per_element, unparsed


def mass_per_H_nucleus(abundances_path):
    """Mean gas mass per hydrogen nucleus [g] implied by a network's abundances.

    This is the factor converting the hydro's mass density to the hydrogen nuclei density the chemistry works in::

        nH = rho / mass_per_H_nucleus(abundances_path)

    It is ``sum_i x_i A_i`` over elements, with ``x_i`` the abundance per H nucleus and ``A_i`` the atomic mass.

    Only elemental composition enters, not speciation -- whether carbon starts as C+ or CO is irrelevant -- so the value is constant over a run.

    Parameters
    ----------
    abundances_path : str or Path
        A network's ``abundances.in``.

    Returns
    -------
    float
        Grams per hydrogen nucleus.

    Raises
    ------
    ValueError
        If the file contains no hydrogen.

    Warns
    -----
    RuntimeWarning
        If the H nuclei do not sum to 1 (the result is then normalised by their sum), or if a species' formula cannot be resolved into elements (lumped species such as ``M``); such species are left out of the mass sum.
    """
    per_element, unparsed = _composition(abundances_path)

    n_H = per_element.get("H", 0.0)
    if n_H <= 0.0:
        raise ValueError(f"No hydrogen found in '{abundances_path}'; cannot compute mass per H nucleus")
    if abs(n_H - 1.0) > 0.01:
        warnings.warn(f"'{abundances_path}': H nuclei sum to {n_H:.4f}, not 1.0 — abundances should be relative to total H nuclei.  Normalising.", RuntimeWarning, stacklevel=2)
    if unparsed:
        warnings.warn(f"'{abundances_path}': species with unresolved formulae excluded from the mass sum: {unparsed}", RuntimeWarning, stacklevel=2)

    amu_per_H = sum(_ATOMIC_MASS[el] * n for el, n in per_element.items()) / n_H
    return amu_per_H * _AMU


class HydroData:
    """
    Loads hydrodynamic tracer files and provides tracer-level sampling.

    Typical workflow::

        h = HydroData()
        h.load_hydro_data("data/tracers_*.npy", abundances="networks/nelson/abundances.in")
        indices, trajectories = h.sample_tracers(m=500)

    Attributes
    ----------
    params : np.ndarray, shape (M, 5) or (M, 7) or None
        Flattened parameter array.  Each row is one tracer–timestep pair; columns follow the order in ``param_names``.  Tracer ``j`` occupies rows ``j * n_timesteps`` to ``(j+1) * n_timesteps``.  Set by :meth:`load_hydro_data`.
    param_names : list of str
        Names of the parameter columns: ``["nH", "T", "Tgrain", "Av", "uv_flux"]``, or ``["nH", "T", "Tgrain", "Av", "uv1", "uv3", "uv4"]`` when ``three_uv_bands`` is set.
    """

    def __init__(self, dt_hydro=None, three_uv_bands=False):
        """
        Parameters
        ----------
        dt_hydro : float or None
            Hydrodynamic timestep in seconds. May also be supplied to :meth:`load_hydro_data`.
        three_uv_bands : bool
            Also read the 3.44-8.0 and 0.41-3.44 eV bands (columns 13 and 14), appending ``uv3`` and ``uv4`` to :attr:`param_names`.  Needed by the :mod:`pdfac` rate path, which interpolates between all three bands.  Off by default, so :attr:`params` keeps its 5-column layout.
        """
        self.three_uv_bands = bool(three_uv_bands)
        self.param_names = list(BAND_PARAM_NAMES if self.three_uv_bands else PARAM_NAMES)
        self.params = None               # (M, len(param_names)) flattened
        self._tracer_file_info = None    # [(row_offset, n_tracers, n_timesteps), ...]
        self.dt_hydro = dt_hydro

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_hydro_data(self, pattern, max_tracers=None, dt_hydro=None, abundances=None):
        """
        Load tracer files and extract the hydrodynamic parameters listed in :attr:`param_names`.

        Populates :attr:`params`.

        Parameters
        ----------
        pattern : str
            Glob pattern matching ``.npy`` tracer files (e.g. ``"data/tracers_*.npy"``).  Each file must be a 3-D array of shape ``(n_timesteps, n_tracers, n_cols)``.  The following column indices are read:

            ======  ===========  ==============================================
            Column  Output       Notes
            ======  ===========  ==============================================
            6       ``nH``       Mass density [g cm⁻³], divided by the mass per
                                 H nucleus to give H nuclei cm⁻³
            7       ``T``        Gas temperature [K]
            8       ``Tgrain``   Grain temperature [K]
            9       ``Av``       Visual extinction [mag]
            11      ``uv_flux``  Radiation field, 8.0–13.6 eV band (UV1)
            13      ``uv3``      Radiation field, 3.44–8.0 eV  (three_uv_bands only)
            14      ``uv4``      Radiation field, 0.41–3.44 eV (three_uv_bands only)
            ======  ===========  ==============================================
        max_tracers : int or None, optional
            If provided, load at most this many tracers total across files (files consumed in sorted order; last file truncated along the tracer axis as needed).  Default ``None`` loads all.
        dt_hydro : float or None, optional
            Hydrodynamic timestep in seconds. If supplied, stored on :attr:`dt_hydro`.
        abundances : str or Path
            The network's ``abundances.in``, used to convert column 6 from mass density to hydrogen nuclei density via :func:`mass_per_H_nucleus`.  Required: the conversion depends on the assumed helium abundance, and getting it wrong biases nH by tens of percent.

        Raises
        ------
        FileNotFoundError
            If no files match ``pattern``.
        ValueError
            If any matched file is not a 3-D array, or ``abundances`` is omitted.
        """
        if abundances is None:
            raise ValueError("load_hydro_data() requires 'abundances' (path to the network's abundances.in): column 6 is a mass density and the conversion to nH depends on the assumed composition")
        m_per_H = mass_per_H_nucleus(abundances)
        if dt_hydro is not None:
            self.dt_hydro = dt_hydro

        filenames = sorted(glob.glob(pattern))
        if not filenames:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        arrays = []
        tracer_file_info = []
        row_offset = 0
        tracers_remaining = max_tracers
        for f in filenames:
            if tracers_remaining is not None and tracers_remaining <= 0:
                break
            d = np.load(f)
            if d.ndim != 3:
                raise ValueError(f"Expected 3-D array (n_timesteps, n_tracers, n_cols), got shape {d.shape} in {f}")
            if tracers_remaining is not None and d.shape[1] > tracers_remaining:
                d = d[:, :tracers_remaining, :]
            n_timesteps_f, n_tracers_f = d.shape[0], d.shape[1]
            tracer_file_info.append((row_offset, n_tracers_f, n_timesteps_f))
            row_offset += n_tracers_f * n_timesteps_f
            if tracers_remaining is not None:
                tracers_remaining -= n_tracers_f
            # Transpose to (n_tracers, n_timesteps, n_cols) before flattening so tracer j occupies contiguous rows j*n_timesteps:(j+1)*n_timesteps
            arrays.append(d.transpose(1, 0, 2).reshape(-1, d.shape[2]))

        self._tracer_file_info = tracer_file_info
        data = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]

        nH     = np.squeeze(data[:, 6]) / m_per_H
        T      = np.squeeze(data[:, 7])
        Tgrain = np.squeeze(data[:, 8])
        Av     = np.squeeze(data[:, 9])
        uv     = np.squeeze(data[:, 11])

        cols = [nH, T, Tgrain, Av, uv]
        if self.three_uv_bands:
            cols += [np.squeeze(data[:, 13]), np.squeeze(data[:, 14])]
        self.params = np.column_stack(cols)

        M = self.params.shape[0]
        print(f"Loaded {len(filenames)} file(s), {M:,} samples ({self.n_tracers:,} tracers)")
        for i, name in enumerate(self.param_names):
            col = self.params[:, i]
            print(f"  {name:>8s}: min={col.min():.4e}  max={col.max():.4e}  mean={col.mean():.4e}  std={col.std():.4e}  median={np.median(col):.4e}")

    # ------------------------------------------------------------------
    # Tracer sampling
    # ------------------------------------------------------------------

    @property
    def n_tracers(self):
        """Total number of tracer trajectories across all loaded files."""
        if self._tracer_file_info is None:
            return None
        return sum(n for _, n, _ in self._tracer_file_info)

    @property
    def n_times(self):
        """Number of time points per tracer."""
        if self._tracer_file_info is None:
            return None
        n_ts_per_file = [n_ts for _, _, n_ts in self._tracer_file_info]
        if len(set(n_ts_per_file)) != 1:
            raise RuntimeError(f"Files have different numbers of timesteps {n_ts_per_file}; no single n_times is defined.")
        return n_ts_per_file[0]

    @property
    def n_params(self):
        """Number of hydro parameters stored per tracer time point."""
        return len(self.param_names)

    def sample_tracers(self, m, rng=None, replace=False, indices=None):
        """
        Sample ``m`` complete tracer trajectories uniformly at random, or fetch a specific set of tracer indices.

        Parameters
        ----------
        m : int
            Number of tracers to sample.  Ignored when ``indices`` is given.
        rng : np.random.Generator, int, or None
            Random source.  An ``int`` is used as a seed for ``np.random.default_rng``; ``None`` creates a fresh unseeded generator.  Ignored when ``indices`` is given.
        replace : bool
            Whether to sample with replacement.  Must be ``True`` when ``m`` exceeds :attr:`n_tracers`.  Ignored when ``indices`` is given.
        indices : array-like of int or None
            If provided, fetch exactly these tracer indices instead of sampling randomly.

        Returns
        -------
        tracer_indices : np.ndarray of int, shape (m,)
            Global tracer indices into the full tracer population.
        trajectories : np.ndarray, shape (m, n_timesteps, n_params)
            Raw parameter trajectories for each selected tracer, columns as in :attr:`param_names`.

        Raises
        ------
        RuntimeError
            If all loaded files do not share the same ``n_timesteps`` (uniform timestep count is required to form a rectangular array).
        """
        if self._tracer_file_info is None or self.params is None:
            raise RuntimeError("Call load_hydro_data() first.")

        n_ts_per_file = [n_ts for _, _, n_ts in self._tracer_file_info]
        if len(set(n_ts_per_file)) != 1:
            raise RuntimeError(f"Files have different numbers of timesteps {n_ts_per_file}; cannot return a uniform array.")
        n_timesteps = n_ts_per_file[0]

        if indices is not None:
            chosen = np.asarray(indices, dtype=int)
        else:
            total = self.n_tracers
            if rng is None:
                rng = np.random.default_rng()
            elif isinstance(rng, (int, np.integer)):
                rng = np.random.default_rng(rng)

            if not replace and m > total:
                raise ValueError(f"Requested {m} tracers but only {total} available; use replace=True or reduce m.")

            chosen = rng.choice(total, size=m, replace=replace)

        # With uniform n_timesteps, global tracer i starts at row i * n_timesteps
        row_starts = chosen * n_timesteps
        trajectories = np.stack([self.params[rs : rs + n_timesteps] for rs in row_starts])  # (m, n_timesteps, n_params)

        return chosen, trajectories

