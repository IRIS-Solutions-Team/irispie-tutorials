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

# 18 · Stochastic simulation

**45 minutes** · after *17 · Estimation* · next: *19 · Sequential models*

Every simulation in Level 3 answered the same question: what happens if
these shocks occur? The shocks were specified by the user. A forecast is
different, because the shocks have not yet occurred, and the appropriate
description of them is a distribution rather than a single value.

This tutorial draws the shocks from that distribution, runs the model many
times in a single call, and summarises the result as a fan chart.

```{code-cell} ipython3
import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, Series, qq

SOURCE = """

!transition-variables
    "Output gap, % of potential"              y
    "Inflation, % per year"                   pi
    "Policy rate, % per year"                 i
    "Real exchange rate gap, %"               q
    "Foreign output gap, %"                   y_w
    "Foreign inflation, % per year"           pi_w
    "Foreign policy rate, % per year"         i_w

!transition-shocks
    shk_y, shk_pi, shk_i, shk_q
    shk_y_w, shk_pi_w, shk_i_w

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

CALIB = dict(a1=0.7, a2=0.2, a3=0.1, a4=0.3, b1=0.1, b2=0.3, b3=0.2,
             c1=0.5, c2=1.5, c3=0.5, pi_tar=2, r_ss=1, beta=0.99, psi=0.1,
             d1=0.8, d2=0.7, d3=0.8, pi_w_ss=2, r_w_ss=1)

STDS = dict(std_shk_y=0.4, std_shk_pi=0.3, std_shk_i=0.2, std_shk_q=0.5,
            std_shk_y_w=0.3, std_shk_pi_w=0.2, std_shk_i_w=0.2)

SHOCKS = ("shk_y", "shk_pi", "shk_i", "shk_q",
          "shk_y_w", "shk_pi_w", "shk_i_w")

m = Simultaneous.from_string(SOURCE, linear=True, flat=True)
m.assign_strict(**CALIB)
m.assign(**STDS)
m.steady()
m.solve_first_order()

FCAST = qq(2025,1) >> qq(2029,4)
DRAWS = 1000
```

## What the shocks look like

The standard deviations assigned in tutorial 16 define a covariance matrix,
which the model reports directly:

```{code-cell} ipython3
cov = np.asarray(m.get_cov_transition_shocks())

print("shape:", cov.shape)
print("diagonal only?", np.allclose(cov, np.diag(np.diag(cov))))
print("standard deviations:", np.round(np.sqrt(np.diag(cov)), 3))
```

The matrix is diagonal and remains so. The model language provides a means
of specifying the size of each shock but none for specifying that two shocks
move together: `assign` accepts `std_shk_y`, and there is no corresponding
name for a correlation.

This restriction may not be appropriate for a given application. If a demand
shock and a risk-premium shock are correlated in the data, the correlation
must be imposed by constructing the covariance matrix explicitly and drawing
from it, rather than by obtaining one from the model:

```{code-cell} ipython3
wanted = cov.copy()
wanted[0, 3] = wanted[3, 0] = 0.4 * 0.5 * 0.6     # corr(shk_y, shk_q) = 0.6

print("still a valid covariance matrix?",
      bool(np.all(np.linalg.eigvals(wanted) > 0)))
```

The draws below use the diagonal matrix reported by the model.
Substituting `wanted` requires no other change to the method.

## A thousand runs in one call

A *variant* is a parallel copy of the model holding its own values.
`expand_num_variants` creates a thousand of them, `Databox.steady` then
constructs shock series with a thousand columns each, and a single
`simulate` call evaluates all of them. Variants are used here only to carry
draws; tutorial 21 is about them properly, including the way every method
in the package changes the shape of its return value once a model holds
more than one:

```{code-cell} ipython3
many = m.copy()
many.expand_num_variants(DRAWS)

rng = np.random.default_rng(11)
drawn = rng.multivariate_normal(np.zeros(len(SHOCKS)), cov,
                                size=(len(FCAST), DRAWS))

db = Databox.steady(many, FCAST)
for position, name in enumerate(SHOCKS):
    db[name] = Series(start=FCAST[0], values=drawn[:, :, position])

out = many.simulate(db, FCAST)

print("model variants :", many.num_variants)
print("y variants     :", out["y"].num_variants)
print("y data shape   :", out["y"].get_data().shape)
```

The array passed to `Series` has shape `(periods, variants)`, which is the
shape `get_data` returns. A thousand simulations of a seven-variable model
over twenty quarters completes in well under a second, because it requires
one model solution followed by a thousand matrix recursions rather than a
thousand complete solutions.

## From draws to a fan

A thousand individual paths cannot be presented directly. Percentiles taken
across the variants can be:

```{code-cell} ipython3
bands = ip.percentile(out["y"], (10, 30, 50, 70, 90))

print("variants after percentile:", bands.num_variants)
print("draws still intact       :", out["y"].num_variants)
print()
print("          p10     p30     p50     p70     p90")
for period in (qq(2025,1), qq(2025,2), qq(2025,4), qq(2027,4)):
    row = np.asarray(bands[period]).ravel()
    print(f"  {str(period):8} " + " ".join(f"{v:7.4f}" for v in row))
```

Use the function `ip.percentile(series, ...)` rather than the method
`series.percentile(...)`. **The method performs the same calculation in
place and returns `None`**: it replaces the thousand draws with five values
per period, and the draws cannot be recovered. The functional form returns a
new series and leaves the original unchanged, which is why the second line
of output still reports a thousand variants.

```{code-cell} ipython3
figure = ip.make_subplots(
    (1, 1),
    figure_title="Output gap: a thousand draws, as a fan",
    figure_height=520,
    show_legend=True,
)

invisible = {"mode": "lines", "line": {"width": 0}}


def add_band(low, high, colour, label):
    """Shade between two percentiles. The lower bound must be drawn first."""
    ip.percentile(out["y"], (low,)).plot(
        span=FCAST, figure=figure, subplot=0, show_figure=False, legend=["_"],
        update_traces=dict(invisible, showlegend=False, hoverinfo="skip"))
    ip.percentile(out["y"], (high,)).plot(
        span=FCAST, figure=figure, subplot=0, show_figure=False, legend=[label],
        update_traces=dict(invisible, fill="tonexty", fillcolor=colour))


add_band(10, 90, "rgba(99,110,250,0.18)", "10 to 90 per cent")
add_band(30, 70, "rgba(99,110,250,0.34)", "30 to 70 per cent")

ip.percentile(out["y"], (50,)).plot(
    span=FCAST, figure=figure, subplot=0, show_figure=False, legend=["median"],
    update_traces={"mode": "lines", "line": {"width": 3, "color": "rgb(40,50,140)"}})

acov = np.asarray(m.get_acov())[0]
rows = [str(name) for name in m.get_acov_dimension_names().rows]
sd_y = float(np.sqrt(np.diag(acov))[rows.index("y")])

for sign in (1, -1):
    figure.add_hline(y=sign * 1.28155 * sd_y, line_width=1,
                     line_dash="dot", line_color="black")

figure
```

`fill="tonexty"` fills down to the preceding trace, so each band requires
two plot calls: the lower bound with no line and no legend entry, then the
upper bound carrying the fill. The bands must be added widest first, or the
narrower band is obscured.

## The fan stops widening

The shape of the fan is the point of interest. It widens for approximately
four quarters and then maintains a constant width for the remaining sixteen,
lying between the two dotted lines.

Those lines are not drawn from the simulation. They come from `get_acov` in
tutorial 10, which computes the model's unconditional covariance
analytically, and they mark the 10th and 90th percentiles of a normal
distribution with that standard deviation. The simulated fan converges to
them:

```{code-cell} ipython3
uncond = np.sqrt(np.diag(acov))

print(f"{'variable':9} {'simulated, last 12q':>20} {'analytic':>10}")
for name in ("y", "pi", "i", "q"):
    simulated = float(np.std(out[name].get_data()[-12:, :]))
    print(f"{name:9} {simulated:20.4f} {uncond[rows.index(name)]:10.4f}")
```

These are two routes to the same quantity: a thousand random paths, and a
matrix equation solved once. They agree to within two per cent, which is the
expected level of agreement at a thousand draws. The window must be taken
from the later part of the span, because the early quarters are still
widening and would bias the simulated figure downwards.

This also explains the shape. **A fan chart for a stationary model does not
widen indefinitely.** Uncertainty about the next quarter amounts to a single
shock, whereas uncertainty five years ahead is the full unconditional
distribution, and nothing lies beyond it. A fan that continues to widen to
the edge of the chart indicates a unit root in the model, whether intended
or not.

## Volatility that changes over the forecast

Before the standard deviations are changed, the source of the randomness in
this tutorial should be stated precisely. It does not come from `simulate`:

```{code-cell} ipython3
quiet = m.copy()
quiet.expand_num_variants(5)
nothing = Databox.steady(quiet, FCAST)      # every shock left at zero

print("spread across five variants:",
      float(np.std(quiet.simulate(nothing, FCAST)["y"].get_data())))
```

`simulate` uses the shocks it is supplied with and draws none of its own.
The standard deviations assigned to the model therefore have no effect on
it; they are used by the Kalman filter, and by the user when constructing
the draws, and nowhere else. All of the randomness above originates in
`rng`.

Time-varying volatility is consequently a property of the draws.
`vary_stds` constructs the standard deviations period by period, from a
multiplier applied to the assigned values, a direct replacement, or both:

```{code-cell} ipython3
multiplier = Databox()
multiplier["std_shk_y"] = Series(start=qq(2025,1),
                                 values=np.array([3.0, 3.0, 3.0, 3.0]))

varying = m.vary_stds(multiplier_db=multiplier, span=FCAST)

print("std_shk_y :", np.round(np.asarray(varying["std_shk_y"].get_data()).ravel()[:8], 3))
print("std_shk_pi:", np.round(np.asarray(varying["std_shk_pi"].get_data()).ravel()[:8], 3))
```

Four quarters of elevated volatility followed by a return to the assigned
value, with every other shock unchanged. To apply it, draw standard normal
variates and scale them period by period:

```{code-cell} ipython3
def fan_with(sd_matrix, seed=11):
    """A thousand draws whose standard deviations may change each period."""
    model = m.copy()
    model.expand_num_variants(DRAWS)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(FCAST), DRAWS, len(SHOCKS)))
    box = Databox.steady(model, FCAST)
    for position, name in enumerate(SHOCKS):
        scaled = z[:, :, position] * sd_matrix[:, position][:, None]
        box[name] = Series(start=FCAST[0], values=scaled)
    return model.simulate(box, FCAST)


loud = np.column_stack([np.asarray(varying["std_" + name].get_data()).ravel()
                        for name in SHOCKS])
calm = np.tile(np.sqrt(np.diag(cov)), (len(FCAST), 1))

for label, matrix in (("as calibrated", calm), ("2025 tripled", loud)):
    band = ip.percentile(fan_with(matrix)["y"], (10, 90))
    widths = [float(np.diff(np.asarray(band[p]).ravel())[0])
              for p in (qq(2025,2), qq(2025,4), qq(2026,2))]
    print(f"  {label:14} 10-90 width  "
          f"2025Q2 {widths[0]:6.4f}   2025Q4 {widths[1]:6.4f}   2026Q2 {widths[2]:6.4f}")
```

The scaling requires one line. The same structure accommodates correlated
draws if the standard normal variates are replaced by a per-period Cholesky
factor.

Note the construction of that `Series`. `np.array([3.0, 3.0, 3.0, 3.0])`
produces four periods. A plain Python list containing the same four values
**produces one period**:

```{code-cell} ipython3
from_list = Series(start=qq(2025,1), values=[3.0, 2.0, 1.0])
from_tuple = Series(start=qq(2025,1), values=(3.0, 2.0, 1.0))
from_array = Series(start=qq(2025,1), values=np.array([3.0, 2.0, 1.0]))

print("list :", from_list.get_data().shape, np.asarray(from_list.get_data()).ravel())
print("tuple:", from_tuple.get_data().shape, np.asarray(from_tuple.get_data()).ravel())
print("array:", from_array.get_data().shape, np.asarray(from_array.get_data()).ravel())
```

A list retains the first value and discards the remainder, with no warning;
a tuple containing the same values behaves correctly. A single-element list
such as `[1.0]` is therefore unaffected, which is why the pattern appears
without incident in earlier tutorials. Use `np.array` for any sequence
longer than one element.

## ⚠️ Break it

**Shock variants the model does not have.**

Constructing the draws is the substantive step, and expanding the model can
appear to be a formality. Omitting it produces the following:

```{code-cell} ipython3
plain = m.copy()                     # one variant, never expanded

forgot = Databox.steady(plain, FCAST)
for position, name in enumerate(SHOCKS):
    forgot[name] = Series(start=FCAST[0], values=drawn[:, :, position])

lonely = plain.simulate(forgot, FCAST)

print("shock variants supplied :", forgot["shk_y"].num_variants)
print("output variants returned:", lonely["y"].num_variants)
```

A thousand columns of shocks were supplied and one column of output was
returned, with no error and no warning. The simulation that was performed is
itself valid:

```{code-cell} ipython3
first_draw = Databox.steady(plain, FCAST)
for position, name in enumerate(SHOCKS):
    first_draw[name] = Series(start=FCAST[0], values=drawn[:, 0, position])

print("identical to draw number one:",
      np.allclose(lonely["y"].get_data(), plain.simulate(first_draw, FCAST)["y"].get_data()))
```

The first draw was used and the remaining nine hundred and ninety-nine were
discarded. Every subsequent step still executes: `ip.percentile` applied to
a one-variant series returns that series unchanged, so the fan chart is
produced, the bands coincide exactly with the median, and the result has the
appearance of a forecast made with complete certainty.

A single check placed before the percentiles prevents this, and should be
included as a matter of routine:

```{code-cell} ipython3
print("variants in the output:", out["y"].num_variants)
assert out["y"].num_variants == DRAWS, "the model was not expanded"
```

## What you did

```{code-cell} ipython3
# the shocks the model thinks it has
cov = np.asarray(m.get_cov_transition_shocks())

# one copy of the model per draw
many = m.copy()
many.expand_num_variants(DRAWS)

# draws as (periods, variants) arrays, one series per shock
db = Databox.steady(many, FCAST)
for position, name in enumerate(SHOCKS):
    db[name] = Series(start=FCAST[0], values=drawn[:, :, position])

out = many.simulate(db, FCAST)

# percentiles across variants - the function, not the method
bands = ip.percentile(out["y"], (10, 50, 90))

# standard deviations that change over the span
varying = m.vary_stds(multiplier_db=multiplier, span=FCAST)
```

## Things to remember

1. **One simulation, many variants.** `expand_num_variants` on the model,
   `(periods, variants)` arrays in the shock series, a single `simulate`
   call.
2. **The covariance matrix is diagonal and cannot be made otherwise.**
   Correlated shocks must be constructed and drawn from an explicit matrix.
3. **`ip.percentile` returns a new series; `series.percentile` modifies the
   original in place** and returns `None`.
4. **Add bands widest first**, lower bound before upper bound, because
   `fill="tonexty"` fills to the preceding trace.
5. **Check the fan against `get_acov`.** A thousand draws and the analytic
   covariance should agree once the fan has stopped widening; here they
   agree to within two per cent.
6. **The fan of a stationary model stops widening.** A fan that does not
   indicates a unit root.
7. **`simulate` draws no shocks.** The standard deviations on the model are
   used by the filter and when constructing draws; `vary_stds` supplies a
   path by which to scale those draws.
8. **A list of numbers is not a series of numbers.** `values=[1.0, 2.0]`
   retains the first value without warning; use `np.array`.
9. **Variant counts do not broadcast.** Shocks with more variants than the
   model are silently truncated to the first.

## Exercise

The cell above tripled the demand shock for the four quarters of 2025 and
reported the 10-to-90 width at three dates. Compute the two ratios: how much
wider the band is in 2025Q2, during the period of elevated volatility, and
how much wider it remains in 2026Q2, two quarters after that period ends.

Before examining the output, predict whether the band in 2025Q2
approximately triples.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**It very nearly triples on impact, and almost all of the increase has
dissipated within two quarters.**

| 10-to-90 width | as calibrated | 2025 tripled | ratio |
|---|---|---|---|
| 2025Q2 | `1.0887` | `3.1533` | `2.90` |
| 2025Q4 | `1.1908` | `3.2184` | `2.70` |
| 2026Q2 | `1.2908` | `1.5063` | `1.17` |

The near-tripling has a straightforward explanation. On impact the output
gap consists largely of its own shock: the remaining six shocks reach it
only through the interest rate, the exchange rate and foreign demand, and
each enters with a small coefficient. Tripling the dominant shock therefore
almost triples the spread.

The speed of the return is the more informative result. By 2026Q2, one
quarter after the period of elevated volatility ends, the band is only 17
per cent wider than it would otherwise have been. Output persistence is
`a1 = 0.7`, so one quarter of additional volatility is reduced to half its
size within two quarters and to a fifth within five, and the policy rule
acts against it throughout.

This is the same property as the convergence of the fan to a fixed width
earlier in the tutorial, observed from the opposite direction. A stationary
model returns to its unconditional distribution, and the rate at which it
does so is a property of the model rather than of the shock applied to
it.

</details>

## Next

**Tutorial 19 · Sequential models** sets `Simultaneous` aside.
`Sequential` models are solved equation by equation in a fixed order rather
than jointly, which is the appropriate structure for reporting blocks,
identities and satellite equations. `sequentialize` determines the order,
`reorder_equations` overrides it, and the incidence matrix shows why one
ordering is valid and another is not.
