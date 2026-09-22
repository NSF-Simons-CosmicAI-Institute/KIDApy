"""
Radiation-field scaling of FUV photoreaction rates.

Replaces ``k = alpha * exp(-gamma*Av)`` with ``k = alpha * PDfac``, where PDfac is the local radiation field relative to the reference (Draine) field at the wavelength relevant to the reaction.  PDfac = 1 is the unattenuated reference field; PDfac is clamped to ``[0, 1]``.

The local field is sampled in three energy bins.  A natural cubic spline through ``(WL, BNORM)``, where ``BNORM = log10(UV / UVREF)``, gives the ratio at any wavelength.  Each reaction is evaluated at ``XVAL = 550nm / (MULT*gamma)``, with ``gamma`` the third network coefficient and ``MULT`` set by whether the first reactant is an atom or a molecule.  ``AV`` is recovered as ``-ln(FLUXR)``, the ratio at 550 nm.

===== ================ ============= ===========
Bin   Energy [eV]      WL [nm]       UVREF
===== ================ ============= ===========
UV1   8.00 - 13.60     125.44477     0.64282316
UV3   3.44 -  8.00     245.13367     1.1328151
UV4   0.41 -  3.44     1502.4689     8.2202859
===== ================ ============= ===========

``WL`` is the energy-weighted median of the Draine field within each bin; ``UVREF`` is the Draine field integrated over each bin, in the normalised units the hydro reports.


    >>> sd = spline_init(uv1=1e-3, uv3=1e-2, uv4=0.5)
    >>> pdfac(sd, xval("H2O", 3.10))
    np.float64(0.000595...)
"""

from __future__ import annotations

__all__ = [
    "SplineData",
    "spline_init",
    "pdfac",
    "xval",
    "av",
    "to_backend",
]

import re
from typing import NamedTuple

import numpy as np

#: Energy-weighted median wavelength of each bin [m].
WL = (125.44477e-9, 245.13367e-9, 1502.4689e-9)

#: Reference field integrated over each bin, in the hydro's normalised units.
UVREF = (0.64282316, 1.1328151, 8.2202859)

#: Wavelength that ``gamma`` is referenced to [m].
WL_REF = 550.0e-9

#: ``MULT`` for atoms and for molecules.
MULT_ATOM = 1.3284
MULT_MOLECULE = 1.7843

_ELEMENTS = ("He", "Si", "Fe", "Na", "Mg", "Cl",
             "H", "C", "N", "O", "S", "P", "F")

_FORMULA_TOKEN = re.compile(
    "(" + "|".join(sorted(_ELEMENTS, key=len, reverse=True)) + r")(\d*)"
)

#: Species with no element count: electrons, pseudo-grains, external fields.
_NON_MOLECULAR = frozenset({"e-", "e", "GRAIN", "GRAIN0", "GRAIN-",
                            "Photon", "CR", "CRP"})

#: Species with no parseable formula.  XH is grain-surface hydrogen (H=1 in the KIDA species file); Nelson's CHx and OHx stand for CH/CH2/CH3 and OH/H2O; M is a generic metal.
_MSUM_OVERRIDE = {"XH": 1, "CHx": 2, "OHx": 2, "M": 1}

#: Isomer prefixes (cyclic, linear, trans) — structure, not composition.
_ISOMER_PREFIX = re.compile(r"^[clt]-")

_MSUM_CACHE: dict = {}


class SplineData(NamedTuple):
    """Contents of the Fortran ``COMMON /SPLINEDATA/``.

    ``wl`` is shared and increasing; ``bnorm`` and ``y2wl`` are ``(..., n_bands)``, one entry per cell.
    """

    wl: np.ndarray       # (n_bands,) wavelengths [m]
    bnorm: np.ndarray    # (..., n_bands) log10(UV / UVREF)
    y2wl: np.ndarray     # (..., n_bands) spline second derivatives


# ---------------------------------------------------------------------------
# MULT and XVAL
# ---------------------------------------------------------------------------

def msum(species: str) -> int:
    """Number of atoms in a formula, ignoring charge.

    Electrons and pseudo-grain species give 0.

    >>> msum("C"), msum("H2O"), msum("CH3+"), msum("e-")
    (1, 3, 4, 0)
    """
    if species in _MSUM_CACHE:
        return _MSUM_CACHE[species]

    core = species.strip()
    if core in _NON_MOLECULAR:
        _MSUM_CACHE[species] = 0
        return 0

    core = _ISOMER_PREFIX.sub("", core).rstrip("+-")
    if core in _MSUM_OVERRIDE:
        _MSUM_CACHE[species] = _MSUM_OVERRIDE[core]
        return _MSUM_OVERRIDE[core]

    total = 0
    consumed = 0
    for match in _FORMULA_TOKEN.finditer(core):
        if match.start() != consumed:       # gap => unrecognised token
            break
        total += int(match.group(2)) if match.group(2) else 1
        consumed = match.end()
    if consumed != len(core):
        raise ValueError(f"Cannot parse formula {species!r}: unknown element "
                         f"{core[consumed:]!r}")

    _MSUM_CACHE[species] = total
    return total


def mult(species: str) -> float:
    """``MULT_ATOM`` when ``msum == 1``, else ``MULT_MOLECULE``.

    Charge does not count, so ``C+`` is an atom and ``H2+`` a molecule.
    """
    return MULT_ATOM if msum(species) == 1 else MULT_MOLECULE


def xval(species: str, gamma: float) -> float:
    """``550nm / (MULT*gamma)`` [m]; ``inf`` when ``gamma == 0``.

    ``species`` is the first non-photon reactant, which sets ``MULT``.  A zero ``gamma`` has no effective wavelength; :func:`pdfac` treats those reactions as unattenuated.
    """
    if gamma == 0.0:
        return np.inf
    return WL_REF / (mult(species) * gamma)


# ---------------------------------------------------------------------------
# SPLINE_INIT / SPLINE_EVAL
# ---------------------------------------------------------------------------

def _y2wl(x: np.ndarray, y, xp):
    """Natural-cubic-spline second derivatives.

    ``x`` is shared 1-D, ``y`` is ``(..., n)``.  Forward-pass coefficients depend only on ``x`` and stay scalar.
    """
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"need at least 3 bands, got {n}")

    zero = xp.zeros_like(y[..., 0])
    c = [0.0] * n
    u = [zero] * n
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * c[i - 1] + 2.0
        c[i] = (sig - 1.0) / p
        u[i] = (6.0 * ((y[..., i + 1] - y[..., i]) / (x[i + 1] - x[i])
                       - (y[..., i] - y[..., i - 1]) / (x[i] - x[i - 1]))
                / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    y2 = [zero] * n
    for k in range(n - 2, 0, -1):
        y2[k] = c[k] * y2[k + 1] + u[k]
    return xp.stack(y2, axis=-1)


def spline_init(uv1, uv3, uv4, xp=np) -> SplineData:
    """Build the spline from the three band energy densities.

    Parameters
    ----------
    uv1, uv3, uv4 : float or array
        Local energy density in the 8.0-13.6, 3.44-8.0 and 0.41-3.44 eV bins, in the hydro's normalised units.  Arrays must share a shape, which becomes the leading dimensions of the result.
    xp : module
        ``numpy`` (default) or ``jax.numpy``.
    """
    wl = xp.asarray(WL, dtype=np.float64)
    ref = xp.asarray(UVREF, dtype=np.float64)

    uv = xp.stack([xp.asarray(uv1, dtype=np.float64),
                   xp.asarray(uv3, dtype=np.float64),
                   xp.asarray(uv4, dtype=np.float64)], axis=-1)
    bnorm = xp.log10(uv / ref)
    return SplineData(wl=wl, bnorm=bnorm, y2wl=_y2wl(wl, bnorm, xp))


def spline_eval(sd: SplineData, x, xp=np):
    """Field ratio at wavelength ``x`` [m], unclamped.

    Outside the band range the end cubic is extrapolated, as in the Fortran.
    """
    wl, y, y2 = sd.wl, sd.bnorm, sd.y2wl

    klo = xp.clip(xp.searchsorted(wl, x, side="right") - 1, 0, wl.shape[0] - 2)
    khi = klo + 1

    h = wl[khi] - wl[klo]
    a = (wl[khi] - x) / h
    b = (x - wl[klo]) / h
    ylog = (a * y[..., klo] + b * y[..., khi]
            + ((a ** 3 - a) * y2[..., klo]
               + (b ** 3 - b) * y2[..., khi]) * h * h / 6.0)
    return 10.0 ** ylog


def pdfac(sd: SplineData, x, clamp: bool = True, xp=np):
    """Factor scaling ``alpha`` in ``k = alpha * PDfac``.

    Parameters
    ----------
    sd : SplineData
        From :func:`spline_init`; leading dimensions are preserved.
    x : float or array, shape (n_reactions,)
        Effective wavelengths [m] from :func:`xval`.  ``inf`` entries (``gamma == 0``) return 1, i.e. unattenuated.
    clamp : bool
        Restrict the result to ``[0, 1]``.  Default True.
    xp : module
        ``numpy`` (default) or ``jax.numpy``.  Under JAX pass ``sd`` through :func:`to_backend` first.

    Returns
    -------
    array
        Shape ``sd.bnorm.shape[:-1] + x.shape``.
    """
    lam = xp.asarray(x, dtype=np.float64)

    # inf would propagate NaN through the cubic; evaluate at a dummy node and substitute after.
    finite = xp.isfinite(lam)
    out = spline_eval(sd, xp.where(finite, lam, sd.wl[0]), xp=xp)
    out = xp.where(finite, out, 1.0)

    return xp.clip(out, 0.0, 1.0) if clamp else out


# ---------------------------------------------------------------------------
# AV
# ---------------------------------------------------------------------------

def fluxr(sd: SplineData, xp=np):
    """Field ratio at 550 nm."""
    return spline_eval(sd, xp.asarray(WL_REF), xp=xp)


def av(sd: SplineData, xp=np):
    """``AV = -ln(FLUXR)``, floored at 0."""
    return xp.maximum(-xp.log(fluxr(sd, xp=xp)), 0.0)


def to_backend(sd: SplineData, xp) -> SplineData:
    """Move a :class:`SplineData` onto another array back-end (e.g. ``jax.numpy``)."""
    return SplineData(*(xp.asarray(f) for f in sd))
