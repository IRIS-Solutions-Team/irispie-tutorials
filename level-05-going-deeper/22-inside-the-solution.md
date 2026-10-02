---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 22 · Inside the solution

**45 minutes** · after *21 · Parameter variants* · next: *23 · Inside the
data contract*

`solve_first_order` has been called in every tutorial since the first one,
and tutorial 8 described what it produces: the matrices `T`, `K`, `P`, `Z`,
`D` and `H` that step the model forward. This tutorial opens the step
itself. What the equations look like before they are solved, what the
solving consists of, and which of the intermediate objects are worth reading
when something is wrong.

```{code-cell} ipython3
from collections import Counter

import numpy as np
from irispie import Simultaneous, Databox, Series, qq

SOURCE = """

!transition-variables
    y, pi, i, q, y_w, pi_w, i_w

!transition-shocks
    shk_y, shk_pi, shk_i, shk_q, shk_y_w, shk_pi_w, shk_i_w

!measurement-variables
    obs_pi, obs_i, obs_q

!measurement-shocks
    shk_obs_pi, shk_obs_i, shk_obs_q

!parameters
    a1, a2, a3, a4, b1, b2, b3, c1, c2, c3
    pi_tar, r_ss, beta, psi
    d1, d2, d3, pi_w_ss, r_w_ss

!transition-equations
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + a3*q + a4*y_w + shk_y;

    pi - pi_tar = b1*(pi{-1} - pi_tar) + beta*(1-b1)*(pi{+1} - pi_tar)
                + b2*y + b3*q + shk_pi;

    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

    q = q{+1} - ((i - pi{+1}) - (i_w - pi_w{+1}))/4 - psi*q + shk_q;

    y_w = d1*y_w{-1} + shk_y_w;
    pi_w = d2*pi_w{-1} + (1-d2)*pi_w_ss + shk_pi_w;
    i_w = d3*i_w{-1} + (1-d3)*(pi_w_ss + r_w_ss) + shk_i_w;

!measurement-equations
    obs_pi = pi + shk_obs_pi;
    obs_i = i + shk_obs_i;
    obs_q = q + shk_obs_q;

"""

CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2,
             c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99, psi=0.1,
             d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

STDS = dict(std_shk_y=0.4, std_shk_pi=0.3, std_shk_i=0.2, std_shk_q=0.5,
            std_shk_y_w=0.3, std_shk_pi_w=0.2, std_shk_i_w=0.2,
            std_shk_obs_pi=0.1, std_shk_obs_i=0.05, std_shk_obs_q=0.2)


def build(source=SOURCE, calib=CALIB, **solve_kwargs):
    model = Simultaneous.from_string(source, linear=True, flat=True)
    model.assign_strict(**calib)
    model.assign(**STDS)
    model.steady()
    model.solve_first_order(**solve_kwargs)
    return model


m = build()
```

## The system before it is solved

`systemize` returns the model as it stands after differentiation and before
anything is solved. Every equation has been written as a linear relation
between the variables dated now, the variables dated next period, the
constants and the shocks:

```{code-cell} ipython3
system = m.systemize()

for name in ("A", "B", "C", "D"):
    print(f"  transition  {name}: {np.asarray(getattr(system, name)).shape}")
for name in ("F", "G", "H", "J"):
    print(f"  measurement {name}: {np.asarray(getattr(system, name)).shape}")
```

The transition block is `A @ xi(t+1) = B @ xi(t) + C + D @ u(t)` and the
measurement block is `F @ y(t) = G @ xi(t) + H + J @ w(t)`. Nothing has been
inverted and nothing has been classified; this is the model as written,
arranged into matrices.

The dimension to notice is ten. The model has seven transition variables,
and the system vector has ten elements, because a variable that appears with
a lead has to be carried separately from the same variable dated now.

## Why the solving needs a generalized decomposition

```{code-cell} ipython3
A = np.asarray(system.A)
B = np.asarray(system.B)

print("A is", A.shape, "with rank", np.linalg.matrix_rank(A))
print("B is", B.shape, "with rank", np.linalg.matrix_rank(B))
print()
print("A can be inverted:", np.linalg.matrix_rank(A) == A.shape[0])
```

`A` is singular. Some rows of the system contain no next-period term at all
— a static relation, or an equation whose forward variable was eliminated —
so the system cannot be rearranged into `xi(t+1) = inv(A) @ B @ xi(t)` and
the ordinary eigenvalue problem is unavailable.

What is used instead is the generalized eigenvalue problem for the pair
`(A, B)`, solved by a QZ decomposition. It produces the same classification
without ever inverting `A`, and it is why one of the eigenvalues below is
not a finite number.

## The eigenvalues

```{code-cell} ipython3
solution = m.get_solution()
eigenvalues = np.asarray(solution.eigenvalues)
kinds = [str(k).split(".")[-1] for k in solution.eigenvalues_stability]

print("eigenvalues:", len(eigenvalues))
for value, kind in sorted(zip(np.abs(eigenvalues), kinds)):
    print(f"  modulus {value:>22.4f}   {kind}")
print()
print(" ", Counter(kinds))
```

Ten eigenvalues for seven variables, and the last of them has a modulus of
about `5e17`. That is the infinite generalized eigenvalue produced by the
singularity of `A`, reported as a very large number rather than as infinity.
It counts as unstable, which is correct: a root at infinity is as far
outside the unit circle as a root can be.

Seven are stable and three are unstable. Tutorial 11 established that this
model has three variables appearing with a lead, so the Blanchard–Kahn
condition is three against three and the solution is unique. That count is
performed here, on this list.

## Two forms of the same solution

The solution is stored twice, in two coordinate systems.

```{code-cell} ipython3
print("square form, in the model's own variables")
for name in ("T", "K", "P", "Z", "D", "H"):
    print(f"  {name}: {np.asarray(getattr(solution, name)).shape}")
print()
print("triangular form, in transformed coordinates")
for name in ("Ta", "Ka", "Pa", "Za", "Ua"):
    print(f"  {name}: {np.asarray(getattr(solution, name)).shape}")
```

The square form is the one tutorial 8 used and the one every simulation
reports: `T` maps the model's variables onto themselves, so its rows and
columns are `y`, `pi`, `i` and the rest, in the order
`get_solution_vectors` reports.

The triangular form is the same system after a change of coordinates. The
transformed vector is usually written `alpha`, the transformation is `Ua`,
and the two forms are related by a similarity:

```{code-cell} ipython3
T = np.asarray(solution.T)
Ta = np.asarray(solution.Ta)
Ua = np.asarray(solution.Ua)

print("Ua @ Ta @ inv(Ua) equals T:",
      np.allclose(Ua @ Ta @ np.linalg.inv(Ua), T, atol=1e-8))
print()
print("num_xi   ", solution.num_xi, " the model's transition variables")
print("num_alpha", solution.num_alpha, " the transformed vector")
print("num_stable", solution.num_stable)
print("num_unit_roots", solution.num_unit_roots)
```

The point of the transformation is that `Ta` is arranged so the stable part
of the system is separated from the rest. That separation is what makes the
stable sub-blocks below meaningful, and it is why the solver works in these
coordinates and converts back at the end.

## The stable sub-block

On this model every element of the transformed vector is stable, so the
sub-blocks are the whole matrices. A model with a unit root is the
interesting case:

```{code-cell} ipython3
RANDOM_WALK = SOURCE.replace("y_w = d1*y_w{-1} + shk_y_w;",
                             "y_w = y_w{-1} + shk_y_w;")
walking = build(RANDOM_WALK)
walking_solution = walking.get_solution()

print(f"{'':12} {'stationary':>12} {'unit root':>11}")
for label, attribute in (("num_alpha", "num_alpha"),
                         ("num_stable", "num_stable"),
                         ("num_unit_roots", "num_unit_roots")):
    print(f"{label:14} {getattr(solution, attribute):>10} "
          f"{getattr(walking_solution, attribute):>11}")
print()
print("Ta_stable, stationary:", np.asarray(solution.Ta_stable).shape)
print("Ta_stable, unit root :", np.asarray(walking_solution.Ta_stable).shape)
print()
print(" ", Counter(str(k).split(".")[-1]
                   for k in walking_solution.eigenvalues_stability))
```

Making foreign output a random walk converts one stable root into a unit
root. The transformed vector still has seven elements, but only six of them
are stable, and `Ta_stable` shrinks to a six-by-six block. The unit-root
direction is held separately, which is what tutorial 16's `fixed_unknown`
initialisation was treating as an unknown constant.

## The forward expansion

A model with leads responds to shocks that have not happened yet, which
tutorial 12 used for anticipated shocks. The machinery is an expansion of
the solution forward, computed on demand:

```{code-cell} ipython3
fresh = build()
fresh_solution = fresh.get_solution()

print("before:", len(fresh_solution.square_expansion), "expansion matrices")

fresh_solution.expand_square_solution(3)

print("after :", len(fresh_solution.square_expansion), "expansion matrices")
for position, matrix in enumerate(fresh_solution.square_expansion):
    print(f"  horizon {position+1}: shape {np.asarray(matrix).shape}"
          f"   largest entry {np.abs(np.asarray(matrix)).max():.4f}")
```

Each matrix says how a shock known to arrive that many periods ahead affects
the variables today. The entries fall as the horizon lengthens, which is the
same statement as tutorial 12's finding that a warning given further in
advance changes today's outcome less.

The list is empty until something asks for it. A simulation with anticipated
shocks expands it to the horizon it needs.

## Terminal conditions

A model with leads needs an assumption about what happens after the last
simulated period, because the equation in that period refers to the one
after it. The first-order solution supplies that assumption, which is why
`stacked_time` requires a solved model even though it never uses the
solution to compute the path.

`simulate` takes a `terminal` argument with two settings. The default,
`"first_order"`, recomputes the terminal columns from `T` and `K` at every
iteration, which places the model on its stable path after the span.
`"data"` takes whatever the input databox holds in those columns and keeps
it:

```{code-cell} ipython3
SPAN = qq(2025,1) >> qq(2026,4)
AFTER = qq(2027,1)

for terminal in ("first_order", "data"):
    db = Databox.steady(m, SPAN)
    db["shk_y"][qq(2025,1)] = 1.0
    db["pi"][AFTER] = 6.0
    run = m.simulate(db, SPAN, method="stacked_time", terminal=terminal)
    print(f"{terminal:12} pi:", np.round(np.asarray(run["pi"][SPAN]).ravel()[:6], 4))
```

The `6.0` placed one period past the end of the span is nonsense, and the
two settings treat it differently. Under the default it is overwritten
before the first iteration and inflation returns to the target of two.
Under `"data"` it is a condition the solver must satisfy, and the whole
path bends towards it, reaching `5.09` by the sixth quarter.

`"data"` exists for the case where the terminal values are a genuine
assumption rather than an accident, which is rare. Everywhere else the
default is correct, and the fact that it silently replaces the contents of
those columns is why tutorial 23 can leave them out of its diagnostics.

## clip_small

`solve_first_order` takes a `clip_small` argument that sets negligible
entries of the solution matrices to exact zeros. On a well-conditioned model
it does nothing at all:

```{code-cell} ipython3
plain = build()
clipped = build(clip_small=True)

for name in ("T", "K", "P"):
    before = np.asarray(getattr(plain.get_solution(), name))
    after = np.asarray(getattr(clipped.get_solution(), name))
    print(f"  {name}: exact zeros {int((before==0).sum()):3} -> "
          f"{int((after==0).sum()):3} of {before.size:3}"
          f"   largest change {np.abs(before-after).max():.2e}")
```

Nothing was removed. The argument earns its place on a model with a channel
that is present in the equations and numerically absent:

```{code-cell} ipython3
DISCONNECTED = dict(CALIB)
DISCONNECTED["a4"] = 1e-14          # foreign demand barely reaches output

loose = build(calib=DISCONNECTED)
loose_clipped = build(calib=DISCONNECTED, clip_small=True)

before = np.asarray(loose.get_solution().T)
after = np.asarray(loose_clipped.get_solution().T)
removed = np.abs(before[(before != 0) & (after == 0)])

print("entries set to zero:", removed.size)
print("largest of them    :", f"{removed.max():.3e}" if removed.size else "none")
```

One entry of about `3e-16` was removed, which is the size of accumulated
rounding error rather than of any economic channel. Clipping is a
presentational convenience: it makes a matrix readable by removing entries
that are indistinguishable from zero, and it does not change what a
simulation produces.

## tolerance

The classification of an eigenvalue as stable, unstable or a unit root is a
comparison of its modulus against one, and `tolerance` is the width of the
band around one that counts as a unit root.

```{code-cell} ipython3
moduli = np.sort(np.abs(eigenvalues))
print("moduli nearest to one:", np.round(moduli[moduli < 10][-4:], 4))
print()

for value in (None, 0.1, 0.2):
    try:
        trial = build(tolerance=value)
        counts = Counter(str(k).split(".")[-1]
                         for k in trial.get_solution().eigenvalues_stability)
        print(f"  tolerance={str(value):6} {dict(counts)}")
    except Exception as error:
        print(f"  tolerance={str(value):6} {type(error).__name__}: {error}")
```

The unstable pair has a modulus of `1.1427`, so a tolerance of `0.1` leaves
it unstable and the solution is unchanged. A tolerance of `0.2` brings it
inside the band, the model is then asked to treat two roots as unit roots
when the counting says they are not, and the solver refuses.

The advice in that message is to increase the tolerance, which is what
caused the failure. Read it as a report that the classification is
inconsistent, and change the model or the calibration rather than the
tolerance.

## ⚠️ Break it

**Reading the eigenvalues off the diagonal of `Ta`.**

`Ta` is triangular, the diagonal of a triangular matrix holds its
eigenvalues, and the stable eigenvalues are the ones that govern how fast
the model returns to its steady state. Putting those three facts together
gives the wrong answer:

```{code-cell} ipython3
stable_moduli = np.sort(np.abs(
    [value for value, kind in zip(eigenvalues, kinds) if kind == "STABLE"]))

print("stable moduli, as reported :", np.round(stable_moduli, 4))
print("sorted |diagonal of Ta|    :", np.round(np.sort(np.abs(np.diag(Ta))), 4))
```

Two of the seven are wrong. The reported modulus is `0.5126` and the
diagonal says `0.4891`.

`Ta` is quasi-triangular rather than triangular. A real decomposition cannot
put a complex pair on the diagonal, so it leaves a two-by-two block instead:

```{code-cell} ipython3
print("largest entry below the diagonal:",
      f"{np.abs(np.tril(Ta, -1)).max():.4f}")
print("found at:", np.argwhere(np.abs(np.tril(Ta, -1)) > 1e-8).tolist())
print()
block = Ta[2:4, 2:4]
print("the block:")
print(np.round(block, 4))
print()
print("its eigenvalues:", np.round(np.linalg.eigvals(block), 4))
print("their modulus  :", round(float(abs(np.linalg.eigvals(block)[0])), 4))
```

The block carries a complex pair, `0.4891 ± 0.1534j`, whose modulus is
`0.5126`. Its diagonal entries are the real part, and reading them alone
loses the imaginary part entirely. The modulus is understated, and the
cycling the pair produces disappears from the description altogether.

Nothing raises, and the numbers are plausible. Use `get_eigenvalues`, which
reports the pair correctly, and treat `Ta` as a computational
intermediate rather than as a table to read.

## What you did

```{code-cell} ipython3
# the system as written, before anything is solved
system = m.systemize()
system.A, system.B, system.C, system.D

# the eigenvalues and how each was classified
solution = m.get_solution()
solution.eigenvalues
solution.eigenvalues_stability

# the two coordinate systems and the transformation between them
solution.T, solution.Ta, solution.Ua

# the stable part, and how much of the vector it covers
solution.Ta_stable
solution.num_stable, solution.num_unit_roots, solution.num_alpha

# the forward expansion used by anticipated shocks
solution.expand_square_solution(3)

# what happens after the last simulated period
m.simulate(Databox.steady(m, SPAN), SPAN, method="stacked_time",
           terminal="first_order")
```

## Things to remember

1. **`systemize` returns the model before it is solved**, as `A`, `B`, `C`,
   `D` for the transition block and `F`, `G`, `H`, `J` for measurement.
2. **The system vector is larger than the variable vector**, because a
   variable appearing with a lead is carried separately.
3. **`A` is singular**, so the solver uses a generalized decomposition and
   one eigenvalue is reported as an enormous number standing for infinity.
4. **The solution is stored twice**, in the model's own coordinates and in
   transformed ones, related by `Ua @ Ta @ inv(Ua) == T`.
5. **The stable sub-blocks shrink when a unit root appears.** A random walk
   in one equation took `Ta_stable` from seven by seven to six by six.
6. **The forward expansion is computed on demand** and is what anticipated
   shocks use.
7. **The solution also supplies the terminal condition.** `terminal`
   defaults to `"first_order"`, which overwrites the columns past the end
   of the span at every iteration; `"data"` keeps whatever was in them.
8. **`clip_small` is presentational.** It removes entries the size of
   rounding error and leaves simulations unchanged.
9. **`Ta` is quasi-triangular.** Its diagonal is not the list of
   eigenvalues when any of them are complex.

## Exercise

Expand the forward solution to sixty horizons and record the largest entry
at horizons 10, 20, 30, 40, 50 and 60. Work out the average rate at which
the entries shrink per horizon, and compare it with the eigenvalues.

Before running it, predict what sets that rate.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**The rate is the reciprocal of the smallest unstable eigenvalue modulus,
`1 / 1.1427 = 0.8751`.**

| horizon | largest entry |
|---|---|
| 10 | `0.477975` |
| 20 | `0.146282` |
| 30 | `0.017063` |
| 40 | `0.005402` |
| 50 | `0.002728` |
| 60 | `0.000572` |

Falling from `0.477975` to `0.000572` over fifty horizons is an average
factor of `0.8741` per horizon, against `0.8751` from the eigenvalue.

The reason is the structure of the solution. The unstable roots are the ones
that had to be removed to obtain a unique path, and removing them is what
produces the forward-looking terms: the effect of a shock expected `k`
periods ahead is carried back through `k` divisions by the unstable roots.
The smallest unstable modulus therefore dominates, because dividing by the
number closest to one shrinks the result least.

Taking the ratio of consecutive horizons rather than the average gives an
oscillating sequence rather than a constant. The unstable pair is complex,
`1.1427` in modulus with an imaginary part, so the expansion rotates as well
as decays and individual steps shrink by more or less than the average.

The practical reading is the one tutorial 12 arrived at from the other
direction. Announcing a shock further ahead changes today by less, and how
much less is a property of the model's unstable roots rather than of the
shock. On this calibration an announcement twenty quarters out still moves
today's solution by a fifth of what the same shock does on impact, which
is why the expansion is computed to the horizon a simulation needs rather
than to some fixed length.

</details>

## Next

**Tutorial 23 · Inside the data contract** follows the other half of a
simulation: how a `Databox` becomes the array the solvers actually operate
on. `Dataslate` is that array, `slatable_for_simulate` describes what a
method will ask for, and the initial and terminal columns are where a
simulation that silently returns nothing usually goes wrong.
