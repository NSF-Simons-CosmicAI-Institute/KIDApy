"""
H2 and CO self-shielding of the FUV photodissociation rates.

Port of MAGICKAL_lite's ``ml_shield.f`` (INITSS / H2SHIELD / COSHIELD) and the ``COLSET`` column-density prescription in ``ml_physcon.f``.

The rate is written as the unattenuated rate times a set of dimensionless shielding factors theta <= 1:

    k(H2) = alpha * uv_flux * exp(-gamma*Av) * theta_H2(N_H2)

    k(CO) = alpha * uv_flux * exp(-gamma*Av) * theta_CO(N_CO) * theta_H2CO(N_H2)

theta_H2 is H2 self-shielding; theta_CO is CO self-shielding; theta_H2CO is shielding of the CO bands by overlapping H2 Lyman-Werner lines. 

All three are tabulated in ``networks/lee1996_shielding.dat``.

The Lee tables carry a fourth curve, theta_dust(Av), for continuum extinction of the CO band.  It is loaded but not applied: the KIDA network already supplies its own ``exp(-gamma*Av)`` dust term for the same reaction, and applying both would attenuate twice.  


Interpolation
-------------
Both columns of every table are taken to log10 and fitted with a natural cubic spline, as in ``INITSS``/``SPLINT``. 

"""

from __future__ import annotations

__all__ = [
    "load_tables",
    "column_densities",
    "shield_factors",
    "ShieldTables",
]

import os
import re
from typing import NamedTuple

import numpy as np

#: Default location of the Lee et al. (1996) tables.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "networks", "lee1996_shielding.dat")

#: Floor applied to column densities and Av before taking log10.
_FLOOR = 1.0e-60

#: Index below which theta is clamped to 1, per table.  These are the CDTAB1(1) / CDTAB2(1) / CDTAB3(2) / CDTAB4(3) thresholds of H2SHIELD and COSHIELD, converted to 0-based.
_CLAMP = dict(h2=0, co=0, h2_co=1, dust=2)


class ShieldTables(NamedTuple):
    """Spline coefficients for the four Lee et al. (1996) curves.

    Each field is a ``(3, n)`` array: row 0 is log10 of the abscissa, row 1 is log10 of theta, row 2 the natural-cubic-spline second derivatives of row 1.  Abscissae are strictly increasing.
    """

    h2: np.ndarray        # theta_H2(N_H2)      — H2 self-shielding
    co: np.ndarray        # theta_CO(N_CO)      — CO self-shielding
    h2_co: np.ndarray     # theta_H2CO(N_H2)    — H2 cross-shielding of CO
    dust: np.ndarray      # theta_dust(Av)      — continuum extinction of CO band


# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

#: Header labels marking the start of each block, in file order.
_BLOCK_HEADERS = (
    (r"N\(H2\)\s+Theta\[", "h2"),
    (r"N\(CO\)\s+Theta1\[", "co"),
    (r"N\(H2\)\s+Theta2\[", "h2_co"),
    (r"Av\s+Theta3\[", "dust"),
)

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[QqDdEe][-+]?\d+)?")

_TABLE_CACHE: dict = {}


def _spline_second_derivs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Natural-cubic-spline second derivatives; port of ``SPLINE``.

    Includes the flat-segment guard of the Fortran (``RTG addition``): where three consecutive ordinates are equal the segment is forced linear, which stops the spline ringing across the plateaus in these tables.
    """
    n = x.size
    y2 = np.zeros(n)
    u = np.zeros(n)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (6.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i])
                       - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
                / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
        if y[i] == y[i - 1] and y[i] == y[i + 1]:
            y2[i] = 0.0
            u[i] = 0.0
    y2[n - 1] = 0.0
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]
    return y2


def _parse_number(tok: str) -> float:
    """Convert a Fortran quad/double literal (``1.000Q-60``) to float."""
    return float(tok.replace("Q", "E").replace("q", "E")
                    .replace("D", "E").replace("d", "E"))


def load_tables(path: str | None = None) -> ShieldTables:
    """Read the Lee et al. (1996) shielding tables.

    The blocks are located by their column headers rather than by the hardcoded blank-line counts and row counts that ``INITSS`` uses, so editing the data file cannot silently misalign the read.

    Parameters
    ----------
    path : str, optional
        Table file.  Defaults to ``networks/lee1996_shielding.dat``.

    Returns
    -------
    ShieldTables

    Raises
    ------
    FileNotFoundError
        If the table file is missing.
    ValueError
        If a block header is absent or a block has fewer than two rows.
    """
    path = path or DATA_PATH
    if path in _TABLE_CACHE:
        return _TABLE_CACHE[path]

    if not os.path.exists(path):
        raise FileNotFoundError(f"Shielding table not found: '{path}'")
    with open(path) as fh:
        lines = fh.readlines()

    # Locate each block header, in file order.
    starts = []
    for pattern, name in _BLOCK_HEADERS:
        rx = re.compile(pattern)
        begin = starts[-1][0] if starts else 0
        for i in range(begin, len(lines)):
            if rx.search(lines[i]):
                starts.append((i, name))
                break
        else:
            raise ValueError(f"Shielding table '{path}': no header for block "
                             f"'{name}' (pattern {pattern!r})")

    blocks = {}
    for b, (start, name) in enumerate(starts):
        stop = starts[b + 1][0] if b + 1 < len(starts) else len(lines)
        rows = []
        for line in lines[start + 1:stop]:
            toks = _NUM.findall(line)
            # Data rows are "x  theta"; the first row of each block carries two extra legacy columns, which are ignored.
            if len(toks) >= 2:
                rows.append((_parse_number(toks[0]), _parse_number(toks[1])))
        if len(rows) < 2:
            raise ValueError(f"Shielding table '{path}': block '{name}' has "
                             f"{len(rows)} row(s), need >= 2")
        arr = np.asarray(rows, dtype=np.float64).T          # (2, n)
        arr = np.log10(np.maximum(arr, _FLOOR))
        if np.any(np.diff(arr[0]) <= 0.0):
            raise ValueError(f"Shielding table '{path}': block '{name}' "
                             "abscissa is not strictly increasing")
        blocks[name] = np.vstack([arr, _spline_second_derivs(arr[0], arr[1])])

    tables = ShieldTables(**blocks)
    _TABLE_CACHE[path] = tables
    return tables


# ---------------------------------------------------------------------------
# Column densities (port of COLSET)
# ---------------------------------------------------------------------------

# Gong et al. (2017) broken power laws, N = 10**c * Av**p, with the break at Av = 1 where both branches agree.
_LOG_N_H2_1MAG = 20.837
_LOG_N_CO_1MAG = 16.215
_P_H2_LOW, _P_H2_HIGH = 1.261, 1.019
_P_CO_LOW, _P_CO_HIGH = 4.329, 2.264


def column_densities(Av, xp=np):
    """Shielding column densities N(H2), N(CO) [cm^-2] from Av.

    Broken power laws fitted by Gong et al. (2017), as used by MAGICKAL_lite's ``COLSET``: the steeper exponent applies below Av = 1, the shallower above.

    Parameters
    ----------
    Av : float or array
        Visual extinction [mag].  Traced JAX values are fine when ``xp`` is ``jax.numpy``.
    xp : module
        ``numpy`` (default) or ``jax.numpy``.

    Returns
    -------
    N_H2, N_CO : same type as Av
        Column densities [cm^-2], floored at 1e-50 as in ``COLSET``.
    """
    Avs = xp.maximum(Av, 0.0)
    low = Avs <= 1.0
    # Guard the power against Av == 0 (0**p is fine, but the log-space path below and any reverse-mode derivative are not).
    Avp = xp.maximum(Avs, _FLOOR)
    N_H2 = 10.0 ** _LOG_N_H2_1MAG * xp.where(low, Avp ** _P_H2_LOW,
                                             Avp ** _P_H2_HIGH)
    N_CO = 10.0 ** _LOG_N_CO_1MAG * xp.where(low, Avp ** _P_CO_LOW,
                                             Avp ** _P_CO_HIGH)
    return xp.maximum(N_H2, 1e-50), xp.maximum(N_CO, 1e-50)


# ---------------------------------------------------------------------------
# Shielding factors
# ---------------------------------------------------------------------------

def _theta(x, table, xp, iclamp=0):
    """Evaluate a log-log shielding curve at abscissa ``x`` (linear units).

    Cubic-spline evaluation, port of ``SPLINT``.  At or below ``X[iclamp]`` theta is 1; above the last point it holds the final tabulated value.
    """
    X, Y, Y2 = table[0], table[1], table[2]
    lx = xp.log10(xp.maximum(x, _FLOOR))
    lxc = xp.clip(lx, X[0], X[-1])            # hold the end values, never extrapolate

    j = xp.clip(xp.searchsorted(X, lxc) - 1, 0, X.shape[0] - 2)
    h = X[j + 1] - X[j]
    a = (X[j + 1] - lxc) / h
    b = (lxc - X[j]) / h
    ly = (a * Y[j] + b * Y[j + 1]
          + ((a ** 3 - a) * Y2[j] + (b ** 3 - b) * Y2[j + 1]) * h * h / 6.0)
    return xp.where(lx <= X[iclamp], 1.0, 10.0 ** ly)


def shield_factors(Av, tables=None, xp=np, dust=False):
    """H2 and CO shielding factors at extinction ``Av``.

    Parameters
    ----------
    Av : float or array
        Visual extinction [mag].
    tables : ShieldTables, optional
        Pre-loaded tables.  Loaded (and cached) from disk if omitted.  Under JAX, pass tables already converted with :func:`to_backend`.
    xp : module
        ``numpy`` (default) or ``jax.numpy``.
    dust : bool
        Fold ``theta_dust(Av)`` into ``f_CO``, as ``COSHIELD`` does.  Off by default and handled by parser.py. 

    Returns
    -------
    f_H2, f_CO : same type as Av
        Multiplicative factors in [0, 1] applied on top of the network's ``alpha * uv_flux * exp(-gamma*Av)`` photodissociation rate.
    """
    if tables is None:
        tables = load_tables()

    N_H2, N_CO = column_densities(Av, xp=xp)

    f_H2 = _theta(N_H2, tables.h2, xp, _CLAMP["h2"])
    f_CO = (_theta(N_CO, tables.co, xp, _CLAMP["co"])
            * _theta(N_H2, tables.h2_co, xp, _CLAMP["h2_co"]))
    if dust:
        f_CO = f_CO * _theta(Av, tables.dust, xp, _CLAMP["dust"])

    return f_H2, f_CO


def to_backend(tables: ShieldTables, xp) -> ShieldTables:
    """Move :class:`ShieldTables` onto another array back-end (e.g. jax.numpy)."""
    return ShieldTables(*(xp.asarray(t) for t in tables))
