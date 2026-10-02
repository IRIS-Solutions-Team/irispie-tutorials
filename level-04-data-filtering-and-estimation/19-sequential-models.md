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

# 19 · Sequential models

**40 minutes** · after *18 · Stochastic simulation* · next: *20 · Vector
autoregressions*

Every model so far has been a `Simultaneous` object: a set of equations that
have to be solved jointly, because each one depends on the others and on
expectations of the future. That structure is necessary for a core model and
expensive for everything attached to it.

A great deal of what a forecasting team produces is not simultaneous.
Unemployment follows from output, a price level follows from inflation, a
fiscal balance follows from both, and real variables follow from nominal
ones by definition. These are computed in order: each equation uses results
the preceding equations have already produced, and nothing refers back. The
`Sequential` class is for exactly that, and it evaluates such a block
directly rather than solving it.

```{code-cell} ipython3
import numpy as np
import irispie as ip
from irispie import Simultaneous, Sequential, Databox, Series, qq

CORE_SOURCE = """

!transition-variables
    y, pi, i, q, y_w, pi_w, i_w

!transition-shocks
    shk_y, shk_pi, shk_i, shk_q, shk_y_w, shk_pi_w, shk_i_w

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

"""

CORE_CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2,
                  c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99,
                  psi=0.1, d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

core = Simultaneous.from_string(CORE_SOURCE, linear=True, flat=True)
core.assign_strict(**CORE_CALIB)
core.steady()
core.solve_first_order()

FCAST = qq(2025,1) >> qq(2027,4)
```

## Writing a sequential block

The source uses `!equations` and `!parameters`. There are no transition
variables, no shocks and no measurement section, because a sequential block
declares nothing: every name on a left-hand side is computed, and every
other name must arrive from outside.

```{code-cell} ipython3
SATELLITE = """

!parameters
    u_ss, okun, rho_u, gamma_pi, gamma_u, w_ss

!equations

    "Unemployment, Okun's law with persistence"
    u = rho_u*u{-1} + (1-rho_u)*(u_ss - okun*y);

    "Unemployment gap"
    u_gap === u - u_ss;

    "Nominal wage growth, per cent per year"
    dw = gamma_pi*pi - gamma_u*u_gap + w_ss;

    "Real wage growth, per cent per year"
    dw_real === dw - pi;

"""

satellite = Sequential.from_string(SATELLITE)
satellite.assign(u_ss=5.0, okun=0.4, rho_u=0.6,
                 gamma_pi=0.8, gamma_u=0.5, w_ss=2.0)

print("equations        :", satellite.num_equations)
print("computed here    :", list(satellite.lhs_names))
print("required from    :", list(satellite.rhs_only_names))
print("parameters       :", list(satellite.parameter_names))
print("missing values   :", satellite.check_missing_parameters())
```

Two of the four equations use `=` and two use `===`. The distinction is the
central one in this tutorial.

An equation written with `=` is **behavioural**: it is an approximation that
could be wrong, and IrisPie attaches a residual to it so that it can be
adjusted. An equation written with `===` is an **identity**: it is true by
construction, and it gets no residual, because there is nothing about a
definition that could require adjustment.

```{code-cell} ipython3
print("behavioural equations:", list(satellite.nonidentity_index))
print("identities           :", list(satellite.identity_index))
print("residuals created    :", list(satellite.residual_names))
print()
for explanatory in satellite.iter_explanatories():
    kind = "identity   " if explanatory.is_identity else "behavioural"
    print(f"  {explanatory.lhs_name:9} {kind}  residual {explanatory.residual_name}")
```

Two behavioural equations produce two residuals, named `res_u` and `res_dw`.
Note that neither residual appears anywhere in the source above. **IrisPie
adds the residual term; it must not be written by hand.** The Break it
section shows what happens when it is.

## The incidence matrix

A block is sequential if its equations can be evaluated in a single pass.
The condition is a statement about which equation uses which result, and
that is what the incidence matrix records:

```{code-cell} ipython3
incidence = np.asarray(satellite.incidence_matrix).astype(int)

print("    " + " ".join(f"{n:>8}" for n in satellite.lhs_names))
for row, name in zip(incidence, satellite.lhs_names):
    print(f"{name:9} " + " ".join(f"{v:>8}" for v in row))
print()
print("is_sequential:", satellite.is_sequential)
```

Row `i` marks the left-hand side names that equation `i` uses. The matrix is
**lower triangular**: every equation uses only names computed at or before
its own position, and nothing uses a result produced later. That is the
definition of a sequential block, and `is_sequential` performs exactly this
check.

Reading the rows in order describes the block. `u` depends on nothing
computed here. `u_gap` depends on `u`. `dw` depends on `u_gap`. `dw_real`
depends on `dw`. The chain runs from output to real wages in four steps.

## Running it on the core model

A sequential block takes a databox and adds columns to it. The natural
source for that databox is the core model:

```{code-cell} ipython3
core_in = Databox.steady(core, FCAST)
core_in["shk_y"][qq(2025,1)] = 1.0
core_out = core.simulate(core_in, FCAST)

satellite_in = core_out.copy()
satellite_in["u"] = Series(start=qq(2024,4), values=np.array([5.0]))

result = satellite.simulate(satellite_in, FCAST)

SHOW = qq(2025,1) >> qq(2025,4)
for name in ("y", "pi", "u", "u_gap", "dw", "dw_real"):
    print(f"  {name:8}", np.round(np.asarray(result[name][SHOW]).ravel(), 4))
```

The core model supplied `y` and `pi`; the block produced the other four
series. The demand shock raises output by `0.9057` on impact, which lowers
unemployment to `4.8551` through Okun's law, opens an unemployment gap of
`-0.1449`, and raises nominal wage growth to `3.9913`.

Note the initial condition. `u` has a lag, so the block needs a value for
the quarter before the simulation starts, and it is supplied in the input
databox exactly as initial conditions were in tutorial 11. The three other
variables have no lags and need nothing.

## From a gap to a level

The satellite block above produced more gaps. The other common use is the
opposite direction: turning the model's gaps into the levels that are
actually published. Everything in the core model is a deviation, and nobody
publishes a deviation.

This is a block of pure identities, so it has no residuals at all:

```{code-cell} ipython3
LEVELS = """

!parameters
    g_pot

!equations

    "Log potential output, index"
    lgdp_pot === lgdp_pot{-1} + g_pot/4;

    "Log output, index"
    lgdp === lgdp_pot + y;

"""

levels = Sequential.from_string(LEVELS)
levels.assign(g_pot=2.0)

print("computed here :", list(levels.lhs_names))
print("required from :", list(levels.rhs_only_names))
print("identities    :", list(levels.identity_index))
print("residuals     :", list(levels.residual_names))
```

Potential output grows at `g_pot` per cent a year, which is `g_pot/4` per
quarter, and output is potential plus the gap. Neither statement could be
wrong, so neither gets a residual.

A longer span shows what the two look like together:

```{code-cell} ipython3
WIDE = qq(2015,1) >> qq(2029,4)

wide_rng = np.random.default_rng(3)
wide_in = Databox.steady(core, WIDE)
for name in ("shk_y", "shk_pi", "shk_i", "shk_q",
             "shk_y_w", "shk_pi_w", "shk_i_w"):
    wide_in[name] = Series(start=WIDE[0],
                           values=wide_rng.normal(0, 0.4, len(WIDE)))

wide_core = core.simulate(wide_in, WIDE)

levels_in = wide_core.copy()
levels_in["lgdp_pot"] = Series(start=WIDE[0]-1, values=np.array([100.0]))

with_levels = levels.simulate(levels_in, WIDE)

figure = ip.make_subplots(
    (2, 1),
    figure_title="Output, potential output and the gap between them",
    subplot_titles=["Log output and log potential, index",
                    "Output gap, per cent of potential"],
    figure_height=660,
    show_legend=True,
)

with_levels["lgdp"].plot(
    span=WIDE, figure=figure, subplot=0, show_figure=False,
    legend=["log output"],
    update_traces={"mode": "lines", "line": {"width": 3}})
with_levels["lgdp_pot"].plot(
    span=WIDE, figure=figure, subplot=0, show_figure=False,
    legend=["log potential"],
    update_traces={"mode": "lines",
                   "line": {"width": 2, "dash": "dash", "color": "black"}})
with_levels["y"].plot(
    span=WIDE, figure=figure, subplot=1, show_figure=False,
    legend=["output gap"],
    update_traces={"mode": "lines", "line": {"width": 3}})

figure.add_hline(y=0, line_width=1, line_color="grey", row=2, col=1)

figure
```

The top panel is the picture most people have in mind when an output gap is
discussed, and it is worth looking at the two panels together.

Output runs from an index of about `100` to about `130` over fifteen years.
Potential runs alongside it as a straight line. The gap in the lower panel
is the vertical distance between them, and it stays within about one and a
half per cent of zero throughout.

That difference in scale is the whole difficulty. The gap is a small
residual between two large numbers, one of which is never observed at all.
An error of half a per cent in the trend — which is nothing over fifteen
years of growth — is a large error in the gap, and tutorial 16 estimated
that gap with a standard error of `0.3160` for exactly this reason.

The lower panel also shows why the trend in this block is an assumption
rather than a result. `g_pot` was set to `2.0` and potential is a straight
line by construction. A real exercise estimates the trend and the gap
together, and the two are then separated only by what the model says about
each, which is what the Kalman filter in tutorial 16 was doing.

## Residuals as a lever

The residuals exist to be used. A forecaster who believes unemployment will
be higher than Okun's law implies, for a reason the equation does not
contain, writes that belief into `res_u`:

```{code-cell} ipython3
judgement = satellite_in.copy()
judgement["res_u"] = Series(start=FCAST[0],
                            values=np.full(len(FCAST), 0.5))

adjusted = satellite.simulate(judgement, FCAST)

print("  u, as the equation implies:",
      np.round(np.asarray(result["u"][SHOW]).ravel(), 4))
print("  u, with res_u = 0.5       :",
      np.round(np.asarray(adjusted["u"][SHOW]).ravel(), 4))
```

The adjustment accumulates, because `u` is persistent: half a point added in
every quarter raises unemployment by exactly `0.5` in the first quarter, and
by `1.088` by the fourth. This is the standard mechanism for imposing judgement
on a forecast, and it is available only on the behavioural equations. There
is no `res_u_gap`, because an unemployment gap that differed from `u - u_ss`
would not be an unemployment gap.

## Order, and fixing it

The equations above were written in a workable order. Nothing requires that,
and a block assembled from several people's files usually is not:

```{code-cell} ipython3
shuffled = Sequential.from_string(SATELLITE)
shuffled.assign(u_ss=5.0, okun=0.4, rho_u=0.6,
                gamma_pi=0.8, gamma_u=0.5, w_ss=2.0)

shuffled.reorder_equations([3, 2, 1, 0])
print("after reordering   :", list(shuffled.lhs_names))
print("is_sequential      :", shuffled.is_sequential)
print()
print(np.asarray(shuffled.incidence_matrix).astype(int))
```

The matrix is now upper triangular, which is the same information arranged
backwards: every equation uses a result that will not be computed until
later. `sequentialize` finds an order that works and applies it:

```{code-cell} ipython3
shuffled.sequentialize()

print("after sequentialize:", list(shuffled.lhs_names))
print("is_sequential      :", shuffled.is_sequential)
print()
print("same answers as before:",
      np.allclose(np.asarray(shuffled.simulate(satellite_in, FCAST)["dw_real"][SHOW]),
                  np.asarray(result["dw_real"][SHOW])))
```

The order is restored and the results are identical. `reorder_equations`
takes an explicit permutation when a particular order is wanted for
presentation; `sequentialize` works one out from the incidence matrix.

## When no order exists

Some sets of equations cannot be arranged sequentially, because they are
genuinely simultaneous:

```{code-cell} ipython3
CIRCULAR = """
!equations
    a = b + 1;
    b = a + 1;
"""

circular = Sequential.from_string(CIRCULAR)

print("is_sequential:", circular.is_sequential)
print(np.asarray(circular.incidence_matrix).astype(int))
print()
try:
    circular.sequentialize()
except Exception as error:
    print(f"sequentialize() raises {type(error).__name__}")
    print(f"  {error}")
```

The incidence matrix is full: each equation uses the other's result, so no
ordering puts both below the diagonal. No arrangement of the rows can make
such a matrix triangular, and the block has to be solved jointly rather than
evaluated.

The error message refers to a permutation of integers and does not state the
actual problem, which is that the equations are mutually dependent. Check
`is_sequential` before calling `sequentialize`, and treat a `False` there as
the real diagnosis. A block in this condition belongs in a `Simultaneous`
model.

## ⚠️ Break it

**Writing the residual yourself.**

The residual is added automatically, and an equation that already contains
one is not corrected:

```{code-cell} ipython3
DOUBLED = SATELLITE.replace(
    "u = rho_u*u{-1} + (1-rho_u)*(u_ss - okun*y);",
    "u = rho_u*u{-1} + (1-rho_u)*(u_ss - okun*y) + res_u;")

doubled = Sequential.from_string(DOUBLED)
doubled.assign(u_ss=5.0, okun=0.4, rho_u=0.6,
               gamma_pi=0.8, gamma_u=0.5, w_ss=2.0)

print("residual names are unchanged:", list(doubled.residual_names))
print()
print("as written by IrisPie:")
print("  correct :", satellite.equation_strings[0])
print("  doubled :", doubled.equation_strings[0])
```

The residual appears twice in the stored equation. The model loads, the name
list is identical, and nothing reports a problem. The consequence appears
only when the residual is given a value:

```{code-cell} ipython3
wrong = doubled.simulate(judgement, FCAST)

print("  res_u = 0.5, residual added once :",
      np.round(np.asarray(adjusted["u"][SHOW]).ravel(), 4))
print("  res_u = 0.5, residual written in :",
      np.round(np.asarray(wrong["u"][SHOW]).ravel(), 4))
```

Every adjustment is applied twice. A forecaster who adds half a point of
judgement gets a full point, and the only indication is that the numbers are
further from the equation than intended.

This is easy to do, because writing the residual is correct in a
`Simultaneous` model, where shocks are declared and must appear in the
equations that use them. In a `Sequential` model the residual is created
from the left-hand side name and inserted automatically. Leave it out, and
confirm with `equation_strings` if there is any doubt.

## What you did

```{code-cell} ipython3
# a block of equations evaluated in order, not solved
block = Sequential.from_string(SATELLITE)
block.assign(u_ss=5.0, okun=0.4, rho_u=0.6,
             gamma_pi=0.8, gamma_u=0.5, w_ss=2.0)

# what it computes, and what it needs supplied
block.lhs_names
block.rhs_only_names

# = is behavioural and gets a residual; === is an identity and does not
block.nonidentity_index
block.identity_index
block.residual_names

# whether it can be evaluated in one pass, and how to make it so
block.is_sequential
block.sequentialize()

# run it on a databox the core model produced
block.simulate(satellite_in, FCAST)
```

## Things to remember

1. **`Sequential` evaluates, `Simultaneous` solves.** Use it for anything
   that follows from the core model rather than being determined with it.
2. **`=` is behavioural and receives a residual; `===` is an identity and
   does not.** The distinction determines where judgement can be applied.
3. **Never write the residual into the equation.** It is added
   automatically, and writing it applies the adjustment twice.
4. **A block is sequential when its incidence matrix is lower triangular**,
   which is what `is_sequential` tests.
5. **`sequentialize` derives a working order; `reorder_equations` imposes a
   specific one.** The results are unaffected by the order, provided one
   exists.
6. **Check `is_sequential` before `sequentialize`.** A mutually dependent
   block raises an error about permutations rather than reporting the
   dependency.
7. **Lagged left-hand side variables need initial conditions** in the input
   databox, exactly as in a `Simultaneous` simulation.

## Exercise

Set `res_u` to `1.0` over the whole forecast, then set `res_dw` to `1.0`
instead, and compare the effect of each on real wage growth.

Before running it, predict which has the larger effect in the first quarter,
and whether the ranking is the same four quarters later.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**`res_dw` has the larger effect on impact, and `res_u` overtakes it by the
fourth quarter.**

| change in `dw_real` | 2025Q1 | 2025Q2 | 2025Q3 | 2025Q4 |
|---|---|---|---|---|
| `res_dw = 1.0` | `+1.000` | `+1.000` | `+1.000` | `+1.000` |
| `res_u = 1.0` | `-0.500` | `-0.800` | `-0.980` | `-1.088` |

The two residuals reach real wage growth by different routes. `res_dw`
enters the wage equation directly and with a coefficient of one, and
`dw_real` is `dw` minus inflation, so the effect is exactly one in every
quarter and does not change.

`res_u` enters the unemployment equation, which has a coefficient of one on
the residual, and then reaches wages through `gamma_u = 0.5`. The impact
effect is therefore half as large and of the opposite sign. It does not stay
at a half, because unemployment is persistent at `rho_u = 0.6`: a residual
applied every quarter accumulates, and the effect on wage growth grows with
it, passing the direct effect between the third and fourth quarters.

The general point is that the position of an equation in the chain
determines how an adjustment propagates. A residual on the first equation of
a block is felt by everything after it, scaled by the coefficients along the
path and shaped by any persistence it meets. A residual on the last equation
affects that equation alone.

</details>

## Next

**Tutorial 20 · Vector autoregressions** closes Level 4 with `RedVAR`, which
is estimated from data rather than written down. The subjects are
`estimate`, the Minnesota and mean priors through `MinnesotaPriorObs` and
`MeanPriorObs`, the companion matrix and the stability it determines, and
`resample` for Monte Carlo, bootstrap and wild bootstrap.
