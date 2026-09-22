# Gas-phase KIDA parser and solver

Modules for building and integrating gas-phase astrochemical reaction networks from KIDA-format input files.

The parser exposes the explicit polynomial structure of the ODE right-hand side as sparse matrices $A \in \mathbb R^{N \times N}$ and $B \in \mathbb R^{N \times N^2}$, rather than a plain callable $f(x)$, making it directly amenable to intrusive model reduction.  The solver
exploits this structure to assemble an analytic Jacobian
$$\partial f/\partial x = A + B(x, \cdot) + B(\cdot, x)$$, enabling faster and reliable convergence of the stiff integrator's implicit Newton steps. 

The current implementation only supports chemical networks that contain unimolecular and bimolecular interactions, so that the right-hand side of the ODE is at most quadratic in $x$.

## Files

| File | Contents |
|------|----------|
| `parser.py` | `Network`, `load_abundances` |
| `solver.py` | `QuadraticSolver`, `QuadraticSolverTracer` |
| `shielding.py` | H₂ and CO self-shielding of the FUV photodissociation rates |
| `pdfac.py` | Three-band radiation-field scaling of FUV photoreaction rates |
| `networks/kida.uva.2024/` | KIDA uva 2024 gas-phase reactions and initial abundances |
| `networks/nelson/` | Nelson gas-phase reactions and initial abundances |
| `networks/osu_09_2008/` | OSU 09/2008 gas-phase reactions and initial abundances |
| `networks/lee1996_shielding.dat` | Lee et al. (1996) H₂ and CO shielding tables |
| `examples/kida_uva_2024_point.py` | Point integration of the KIDA uva 2024 network |
| `examples/kida_uva_2024_tracer.py` | Tracer integration of the KIDA uva 2024 network |
| `examples/nelson_point.py` | Point integration of the Nelson network |
| `examples/nelson_tracer.py` | Tracer integration of the Nelson network |
| `examples/nelson_shielding.py` | Effect of CO self-shielding on the Nelson network |

---

## Dependencies

- Python 3.9+
- numpy
- scipy
- matplotlib 

---

## parser.py

### `Network`

Reads a KIDA-format reactions file and builds the sparse ODE matrices
$A$ ($N \times N$) and $B$ ($N \times N^2$) for the system

$$\frac{dx}{dt} = A x + B(x \otimes x)$$

where $x$ is the vector of species abundances per H nucleus. $A$ encodes unimolecular reaction rate coefficients and $B$ encodes bimolecular reaction rate coefficients.

**Constructor**

```python
Network(grains=False, self_shielding=False, dust_attenuation=False, pd_fac=False)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `grains` | bool | Activate pseudo-grain reactions (H₂ formation via XH, ion–grain recombination via GRAIN-/GRAIN0). Default: `False`. |
| `self_shielding` | bool | Apply Lee et al. (1996) H₂ and CO shielding factors to the H₂ and CO photodissociation rates. See `shielding.py`. Default: `False`. |
| `dust_attenuation` | bool | Include the $\exp(-\gamma A_v)$ dust term in frml-2 photoreaction rates. Off by default, on the assumption that `uv_flux` is already attenuated. Default: `False`. |
| `pd_fac` | bool | Evaluate frml-2 photoreaction rates from the three-band radiation field (`uv1`, `uv3`, `uv4`) via PDfac. Mutually exclusive with `dust_attenuation`; when enabled, $A_v$ is derived from the radiation field. Default: `False`. |

**Methods**

```python
net.load_from_disk(filepath)
A, B = net.get_operators(env)
passive = net.get_passive_species()
dropped = net.drop_passive_species()
```

**Environment dict** passed to `get_operators`:

```python
env = dict(
    T       = 10.0,   # gas temperature [K]
    nH      = 1e4,    # total H number density [cm⁻³]
    Av      = 1.0,    # visual extinction [mag]
    uv_flux = 1.0,    # FUV field scaling (1 = standard Draine field)
)
```

For the three-band UV treatment, construct the network with `pd_fac=True` and pass the three band energy densities instead of a single `uv_flux`:

```python
net = Network(pd_fac=True)
env = dict(
    T   = 10.0,
    nH  = 1e4,
    uv1 = 1.0,    # 8.00-13.60 eV band; synonym for uv_flux in the one-band path
    uv3 = 1.0,    # 3.44-8.00 eV band
    uv4 = 1.0,    # 0.41-3.44 eV band
)
```

With `pd_fac=True`, `Av` is computed from the radiation field and any `Av` supplied in `env` is ignored.

**Temperature clamping.** Each KIDA reaction entry carries a validity window $[T_{\min}, T_{\max}]$ over which its rate law was fit.  When `Tcap_2body=True` (the default), the effective temperature used to evaluate bimolecular rate
coefficients is clamped to $[T_{\min}, T_{\max}]$, avoiding extrapolation of the fit outside its tabulated range.  Set `Tcap_2body=False` to disable clamping and use the raw gas temperature for every reaction.

Reactions that list multiple $[T_{\min}, T_{\max}]$ ranges are resolved at load time: the range containing $T$ is selected, or the nearest range if $T$ falls outside all of them.

**Supported formula types**

| frml | Description |
|------|-------------|
| 1 | Cosmic-ray ionisation / CR-induced photons |
| 2 | UV photodissociation |
| 3 | Kooij: $k = \alpha \left(T/300\right)^\beta \exp(-\gamma/T)$ |
| 4 | ionpol1 |
| 5 | ionpol2 |
| 0 | Ion–grain recombination (`grains=True`) |
| 10 | $\mathrm{XH} + \mathrm{XH} \to \mathrm{H_2}$ (`grains=True`) |
| 11 | $\mathrm{H} \to \mathrm{XH}$ (`grains=True`) |

**Pseudo-grain species** (`grains=False`): XH, GRAIN-, and GRAIN0 are excluded from the species list.

**Physical constants** used internally by the parser:

| Constant | Value | Description |
|----------|-------|-------------|
| `grain_radius` | $1.0 \times 10^{-5}\ \mathrm{cm}$ | Grain radius |
| `grain_gas_ratio` | $5.0 \times 10^{-12}$ | Grain number density per H nucleus, $n_\mathrm{gr}/n_\mathrm{H}$ |
| `zeta_cr` | $1.6 \times 10^{-17}\ \mathrm{s}^{-1}$ | Cosmic-ray ionisation rate |

### `load_abundances`

```python
abund = load_abundances(path)
```

Reads a two-column text file (`species  abundance`) and returns a `dict[str, float]`. Lines starting with `#` are treated as comments.

---

## solver.py

### `QuadraticSolver`

Integrates $dx/dt = Ax + B(x \otimes x)$ with fixed $A$ and $B$ using `scipy.integrate.solve_ivp`.

**Methods**

```python
t, y = solver.solve(A, B, t_span, x0, atol, rtol,
                    method="BDF", t_eval=None, scale=None)

out, header = solver.save(path, t, y, col_names=None)

fig, ax = solver.plot(t, y, indices, labels=None, outpath=None)
```

**`solve` parameters**

| Parameter | Description |
|-----------|-------------|
| `A`, `B` | Sparse matrices from `Network.get_operators` |
| `t_span` | $(t_0, t_1)$ in seconds |
| `x0` | Initial state, shape $(N,)$ |
| `atol`, `rtol` | Solver tolerances |
| `method` | scipy integrator, default `"BDF"` |
| `t_eval` | Output times |
| `scale` | Positive array $(N,)$; solve in scaled coordinates $z = x/\texttt{scale}$ |

Returns $t$ of shape $(N_t,)$ and $y$ of shape $(N, N_t)$. Raises `RuntimeError` on solver failure.

**`save` parameters**

| Parameter | Description |
|-----------|-------------|
| `path` | Base path; `.npy` and `.csv` are written automatically |
| `t` | Time array, shape $(N_t,)$ |
| `y` | Solution array, shape $(N, N_t)$ |
| `col_names` | List of $N$ column names for the CSV header |

Output array has shape $(N_t,\ 1+N)$: first column is $t$, then one column per
state variable.

**`plot` parameters**

| Parameter | Description |
|-----------|-------------|
| `t` | Time array |
| `y` | Solution array, shape $(N, N_t)$ |
| `indices` | Row indices of $y$ to plot |
| `labels` | Legend labels (defaults to index numbers) |
| `outpath` | Save figure to this path if provided |

---

### `QuadraticSolverTracer`

Integrates the non-autonomous system

$$\frac{dx}{dt} = A(p(t))\, x + B(p(t))(x \otimes x)$$

where $p(t)$ is the physical environment of a tracer particle recorded by a hydrodynamics simulation. Chemistry is post-processed along the trajectory by solving the ODE at each hydrodynamic snapshot.

Three schemes are available for reconstructing $p(t)$ between the discrete hydrodynamic snapshots:

| `interpolation` | Description |
|-----------------|-------------|
| `"piecewise_constant"` | $A$ and $B$ are assembled once per hydro step and held fixed over the step interval. The ODE is autonomous within each step. |
| `"cubic_spline"` | $p(t)$ is interpolated continuously with a [natural cubic spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html). |
| `"pchip"` | $p(t)$ is interpolated with a [PCHIP spline](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html), which preserves monotonicity of each parameter profile and avoids spurious oscillations near sharp transitions. |

**Methods**

```python
t, y = tracer.solve(dt_hydro, pt, get_tensors, x0, atol, rtol, min_scale,
                    method="BDF", interpolation="piecewise_constant",
                    use_scaling=False, verbose=False, log_cols=None,
                    t_eval=None, jac_structure=None)

results = tracer.solve_multiple(dt_hydro, tracers, get_tensors, x0, atol, rtol,
                                min_scale, method="BDF",
                                interpolation="piecewise_constant",
                                use_scaling=False, log_cols=None, t_eval=None,
                                jac_structure=None, n_workers=1, verbose=False,
                                return_failures=False)

out, header = tracer.save_data(t, y, pt, species, save_path="run",
                               save_csv=True, dt_hydro=None,
                               interpolation="piecewise_constant",
                               log_cols=None)
```

**`solve` parameters**

| Parameter | Description |
|-----------|-------------|
| `dt_hydro` | Hydrodynamic timestep in seconds |
| `pt` | Physical trajectory, shape $(M, 5)$ with columns $[\mathrm{nH},\, T,\, T_\mathrm{grain},\, A_v,\, \mathrm{uv\_flux}]$, or shape $(M, 7)$ with columns $[\mathrm{nH},\, T,\, T_\mathrm{grain},\, A_v,\, \mathrm{uv1},\, \mathrm{uv3},\, \mathrm{uv4}]$ for `pd_fac=True` |
| `get_tensors` | Callable `get_tensors(env) -> (A, B)`; typically `net.get_operators` |
| `x0` | Initial species abundances, shape $(N,)$ |
| `atol`, `rtol` | Solver tolerances |
| `min_scale` | Floor used when constructing the scale vector for scaled solves |
| `method` | scipy integrator, default `"BDF"` |
| `interpolation` | `"piecewise_constant"`, `"cubic_spline"`, or `"pchip"` |
| `use_scaling` | Solve in scaled coordinates $z = x/\mathrm{scale}$ if `True` |
| `log_cols` | Parameter columns to interpolate in log space (e.g. `["nH", "Av", "uv_flux"]`) |
| `t_eval` | Optional output time grid over the full trajectory |

For `"piecewise_constant"`, returns `(list_of_t, list_of_y)` — one entry per hydrodynamic segment. For `"cubic_spline"` and `"pchip"`, returns continuous $(t, y)$ arrays.

**Pre-equilibration.** Unlike `QuadraticSolver.solve`, the tracer solver does not begin the trajectory from `x0` directly. It first integrates for a hardcoded $10^4$ yr at the fixed conditions of `pt[0]`, seeded with `x0`, and uses that relaxed state as the initial condition for the trajectory. The returned `y[:, 0]` is therefore the equilibrated state, not the `x0` passed in.

---

## Quick start

```python
from parser import Network, load_abundances
from solver import QuadraticSolver
import numpy as np

net = Network(grains=True)
net.load_from_disk("networks/kida.uva.2024/gas_reactions_kida.uva.2024.in")
net.drop_passive_species()

abund = load_abundances("networks/kida.uva.2024/abundances.in")
x0 = np.zeros(len(net.species))
for name, val in abund.items():
    if name in net.species_map:
        x0[net.species_map[name]] = val

env = dict(T=10.0, nH=1e4, Av=10.0, uv_flux=1.0)
A, B = net.get_operators(env)

YEAR = 3600 * 24 * 365.25
t_eval = np.logspace(0, np.log10(1e6 * YEAR), 300)

solver = QuadraticSolver()
t, y = solver.solve(A, B, (t_eval[0], t_eval[-1]), x0,
                    atol=1e-20, rtol=1e-6, t_eval=t_eval)

solver.save("output", t, y, col_names=net.species)
```

Self-contained examples are in the `examples/` directory. Run from the
repository root:

```
python examples/kida_uva_2024_point.py
python examples/kida_uva_2024_tracer.py
python examples/nelson_point.py
python examples/nelson_tracer.py
python examples/nelson_shielding.py
```

---

## Author

Arjun Vijaywargiya
[![ORCID](https://img.shields.io/badge/ORCID-0000--0001--5391--5560-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0000-0001-5391-5560)
The University of Texas at Austin

---

## References

Wakelam, V., Gratier, P., Loison, J.-C., Hickson, K. M., Penguen, J., & Mechineau, A. (2024).
The 2024 KIDA network for interstellar chemistry.
*A&A*, 689, A63.
https://doi.org/10.1051/0004-6361/202450606

Gong, M., Ostriker, E. C., & Wolfire, M. G. (2017).
A Simple and Accurate Network for Hydrogen and Carbon Chemistry in the Interstellar Medium.
*ApJ*, 843, 38.
https://doi.org/10.3847/1538-4357/aa7561

Wakelam, V., Herbst, E., Loison, J.-C., et al. (2012).
A Kinetic Database for Astrochemistry (KIDA).
*ApJS*, 199, 21.
https://doi.org/10.1088/0067-0049/199/1/21

Lee, H.-H., Herbst, E., Pineau des Forêts, G., Roueff, E., & Le Bourlot, J. (1996).
Photodissociation of H₂ and CO and time dependent chemistry in inhomogeneous interstellar clouds.
*A&A*, 311, 690.
