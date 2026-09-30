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

# 10 · Model properties

**35 minutes** · after *9 · When it will not solve* · next: *11 · Your first
simulation*

Tutorial 9 asked whether the model works. This one asks what it is like.

A solved model already implies a great deal about the economy it describes —
how volatile inflation will be, how tightly the policy rate follows it, how
many quarters a shock takes to wash out. None of that needs data, and none of
it needs a simulation. It follows from the solution matrices and the size of
the shocks, and IrisPie will compute it in one call.

These are the numbers you compare against the real world when you want to
know whether a calibration is any good.

```{code-cell} ipython3
import numpy as np
import irispie as ip
from irispie import Simultaneous, Databox, qq

CALIB = dict(a1=0.7, a2=0.2, b1=0.6, b2=0.3, c1=0.5,
             c2=1.5, c3=0.5, pi_tar=2, r_ss=1)

SOURCE = """

!transition-variables

    "Output gap, % of potential"          y
    "Inflation, % per year"               pi
    "Policy rate, % per year"             i

!transition-shocks

    "Demand shock"                        shk_y
    "Cost-push shock"                     shk_pi
    "Monetary policy shock"               shk_i

!parameters

    a1, a2, b1, b2, c1, c2, c3, pi_tar, r_ss

!transition-equations

    "Aggregate demand"
    y = a1*y{-1} - a2*(i - pi{+1} - r_ss) + shk_y;

    "Phillips curve"
    pi = b1*pi{-1} + (1-b1)*pi{+1} + b2*y + shk_pi;

    "Policy rule"
    i = c1*i{-1} + (1-c1)*(pi_tar + r_ss + c2*(pi - pi_tar) + c3*y) + shk_i;

"""

def build(**changes):
    """The running model, with any parameters or shock sizes overridden."""
    model = Simultaneous.from_string(SOURCE, linear=True, flat=True)
    calibration = dict(CALIB)
    calibration.update(changes)
    model.assign_strict(**calibration)
    model.steady()
    model.solve_first_order()
    return model

m = build()
```

## How volatile the model is

```{code-cell} ipython3
m.get_acov()
```

One matrix, wrapped in a tuple — the tuple is for higher orders, which come
later in this tutorial. It is the **asymptotic covariance**: what the
variances and covariances of the variables settle down to when the model is
hit by its shocks forever.

The diagonal holds the variances. Standard deviations are easier to read,
because they are in the same units as the variables themselves:

```{code-cell} ipython3
acov = np.asarray(m.get_acov()[0])

for position, name in enumerate(("y", "pi", "i")):
    print(f"{name:3s} variance {acov[position, position]:8.4f}"
          f"   std dev {np.sqrt(acov[position, position]):7.4f}")
```

Now compare those against what goes in:

```{code-cell} ipython3
m.get_stds()
```

**Every shock has a standard deviation of one**, and out of that come
standard deviations of 1.68, 3.43 and 4.60. The model amplifies. A one-point
shock to demand does not produce a one-point economy — it feeds through the
Phillips curve into the policy rule and back again, and the policy rate ends
up the most volatile of the three.

Those are numbers you can hold against reality. A policy rate with a standard
deviation of 4.6 percentage points is a lot; if the country you are modelling
has never moved its rate that much, the calibration is too aggressive
somewhere.

## Reading it by name

Position 0 is `y` because you wrote `y` first. That is fine for three
variables and hopeless for eighty, so the ordering is available:

```{code-cell} ipython3
m.get_acov_dimension_names()
```

`DimensionNames` carries the row and column labels, and it will do the
slicing for you:

```{code-cell} ipython3
names = m.get_acov_dimension_names()

names.select(acov, ("pi", "i"))
```

The 2×2 block for inflation and the policy rate, pulled out by name with no
index arithmetic. Rows and columns can differ — pass a pair of tuples:

```{code-cell} ipython3
names.select(acov, (("pi",), ("y", "i")))
```

One row, two columns: the covariance of inflation with output and with the
policy rate.

## What moves together

Covariances depend on units, which makes them hard to judge. Correlations do
not:

```{code-cell} ipython3
np.round(np.asarray(m.get_acorr()[0]), 4)
```

Ones down the diagonal, and the interesting numbers off it:

```{code-cell} ipython3
acorr = np.asarray(m.get_acorr()[0])

print(f"y  and pi : {acorr[0, 1]:+.4f}")
print(f"y  and i  : {acorr[0, 2]:+.4f}")
print(f"pi and i  : {acorr[1, 2]:+.4f}")
```

**Inflation and the policy rate correlate at 0.92.** That is the policy rule
doing exactly what it was written to do — `i` responds to `pi` with a
coefficient of `c2 = 1.5`, so the two move almost in lockstep.

Output is the loose one. It correlates only weakly with the policy rate,
because it is pushed around by its own demand shock as much as by policy.

This is the sort of number that tells you whether a calibration is plausible
before you have fitted anything. If inflation and the policy rate correlate
at 0.92 in your model and at 0.3 in your data, something in the policy rule
needs to change.

## How long a shock lasts

`up_to_order=` asks for the same thing at a lag. Order 1 relates today to
last quarter, order 2 to two quarters ago, and so on:

```{code-cell} ipython3
lagged = m.get_acorr(up_to_order=6)

print("lag:  ", "  ".join(f"{k:>7d}" for k in range(7)))
for position, name in enumerate(("y", "pi", "i")):
    row = [np.asarray(x)[position, position] for x in lagged]
    print(f"{name:4s}: ", "  ".join(f"{v:+7.4f}" for v in row))
```

Each row is a variable's correlation with its own past — its
**autocorrelation function**, and the cleanest summary of persistence there
is.

Read down the columns. At one quarter, output is at `0.69` while the policy
rate is still at `0.87`: a demand shock has started to fade before the
central bank has finished reacting to it. Output drops below `0.5` between
lags 1 and 2; inflation and the policy rate take until lags 2 to 3.

Now read further along, because something changes sign:

```{code-cell} ipython3
np.round([np.asarray(x)[0, 0] for x in lagged], 4)
```

**Negative from lag 4 onwards.** Output above trend today means output
*below* trend in a year's time. That is not persistence any more — it is a
cycle, and it comes straight from tutorial 9. The eigenvalues included a
complex pair, `0.7706 ± 0.3399i`, and a complex root is exactly what
oscillation looks like in the algebra. Here is the same fact in a form you
could take to data.

## Where the numbers come from

Everything above depends on the shocks. They arrive as an identity matrix:

```{code-cell} ipython3
np.asarray(m.get_cov_transition_shocks())
```

Unit variances, no correlation between shocks — the default until you say
otherwise. Scale all of them at once:

```{code-cell} ipython3
doubled_shocks = build()
doubled_shocks.rescale_stds(2.0)

ratio = np.diag(np.asarray(doubled_shocks.get_acov()[0])) / np.diag(acov)
print("variance ratio after doubling every shock:", np.round(ratio, 6))
```

**Exactly four.** Variance is quadratic in shock size, so doubling the shocks
quadruples every variance and leaves every *correlation* untouched. Scaling
all shocks together changes the loudness and nothing else.

Changing one shock is a different matter:

```{code-cell} ipython3
larger_cost_push = build()
larger_cost_push.assign(std_shk_pi=3.0)
larger_cost_push.solve_first_order()

print("baseline   y-pi correlation:", round(float(acorr[0, 1]), 4))
print("std_shk_pi = 3              :",
      round(float(np.asarray(larger_cost_push.get_acorr()[0])[0, 1]), 4))
```

From `+0.3602` to `-0.0117`. The equations did not move — `b2` is still 0.3,
the policy rule is untouched — and yet output and inflation have stopped
being related.

**A model's correlations are not a property of its equations alone.** They
are a property of the equations *and* the mix of shocks hitting them. Make
cost-push shocks loud enough and the demand-driven relationship between
output and inflation disappears underneath them. This is worth knowing before
you compare a model correlation against a data correlation and conclude the
transmission is wrong.

## Measurement variables

If the model has a measurement side, it appears here too:

```{code-cell} ipython3
OBSERVED = SOURCE.rstrip() + """

!measurement-variables

    "Observed inflation"                  obs_pi

!measurement-shocks

    "Inflation measurement error"         mshk_pi

!measurement-equations

    "Inflation is observed with error"
    obs_pi = pi + mshk_pi;

"""

observed = Simultaneous.from_string(OBSERVED, linear=True, flat=True)
observed.assign_strict(**CALIB)
observed.steady()
observed.solve_first_order()

print(observed.get_acov_dimension_names())
print()
print("variances:", np.round(np.diag(np.asarray(observed.get_acov()[0])), 4))
```

Four variables now, and the arithmetic is worth checking by hand:

```{code-cell} ipython3
variances = np.diag(np.asarray(observed.get_acov()[0]))

print("pi     :", round(float(variances[1]), 4))
print("obs_pi :", round(float(variances[3]), 4))
print("gap    :", round(float(variances[3] - variances[1]), 6))
```

Exactly `1.0` — the variance of the measurement error, which has a standard
deviation of one. `obs_pi = pi + mshk_pi` with two independent pieces, so the
variances add. Observed inflation is noisier than inflation, by precisely the
amount of noise you put in.

## ⚠️ Break it

**Asking for the variance of something that does not have one.**

Tutorial 9 built a random walk and found a unit root. Build it again, and ask
for its covariance:

```{code-cell} ipython3
RANDOM_WALK = SOURCE.replace(
    '    "Policy rate, % per year"             i',
    '    "Policy rate, % per year"             i\n'
    '    "Potential growth, % per year"        g',
).replace(
    '    "Monetary policy shock"               shk_i',
    '    "Monetary policy shock"               shk_i\n'
    '    "Potential growth shock"              shk_g',
) + """
!transition-equations

    "Potential growth is a random walk"
    g = g{-1} + shk_g;

"""

random_walk = Simultaneous.from_string(RANDOM_WALK, linear=True, flat=True)
random_walk.assign_strict(**CALIB)
random_walk.steady()
random_walk.solve_first_order()

print("stability:", random_walk.get_variable_stability())
print("variances:", np.round(np.diag(np.asarray(random_walk.get_acov()[0])), 4))
```

**`nan`** in the last position, and no warning anywhere.

It is the correct answer. A random walk has no asymptotic variance — shocks
accumulate instead of dying away, so the variance grows without limit and
there is nothing finite to report. `get_acov` says so in the only way a float
can.

The trap is not the `nan` itself but what it does next. Feed that matrix into
anything — a weighted average, a covariance-based estimator, a correlation —
and the `nan` spreads silently through everything it touches. The other three
variables have perfectly good variances, and the matrix as a whole is
unusable.

**Run `get_variable_stability()` before `get_acov()`**, and drop the
non-stationary variables, or transform them into something stationary first.
That is what `get_variable_stability()` is actually for.

## What you did

```{code-cell} ipython3
# variances and covariances the model implies, with no data
m.get_acov()

# the same thing scaled to correlations
m.get_acorr()

# row and column labels, and slicing by name
m.get_acov_dimension_names().select(acov, ("pi", "i"))

# persistence: correlation with own past, lag by lag
m.get_acorr(up_to_order=6)

# what drives all of it
m.get_stds()
```

## Things to remember

1. **These are properties of the solved model, not of any data.** They exist
   as soon as `solve_first_order()` has run.
2. **`get_acov` gives covariances, `get_acorr` gives correlations.** Read
   standard deviations off the diagonal of the first, since they are in the
   variables' own units.
3. **`up_to_order=` turns either one into an autocorrelation function**,
   which is the most compact description of persistence available.
4. **A negative autocorrelation is a cycle**, and it traces back to a complex
   pair in the eigenvalues.
5. **Scaling every shock changes variances and no correlations.** Variance
   goes with the square, so doubling the shocks quadruples it.
6. **Changing one shock changes the correlations.** What the model implies
   about co-movement depends on the mix of shocks, not the equations alone.
7. **A non-stationary variable returns `nan`**, correctly, and contaminates
   anything the matrix is used for. Check `get_variable_stability()` first.

## Exercise

Make the central bank noisier: set `std_shk_i` to 3, leaving everything else
alone.

Before you run it, predict what happens to the correlation between inflation
and the policy rate. It is `+0.92` at the baseline.

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

It **falls**, from `+0.9206` to `+0.7857`.

The instinct is that a louder policy shock should make the policy rate more
important and tie the two more tightly together. The opposite happens, and
the reason is what the shock *is*: `shk_i` is the part of the policy rate
that is **not** a response to inflation. Make it bigger and more of the
policy rate's movement has nothing to do with inflation, so the correlation
between them weakens.

Two more results in the same output are worth noticing.

The correlation of output with the policy rate collapses from `+0.166` to
`+0.032`, for the same reason. And the policy rate becomes **less
persistent** — its lag-one autocorrelation drops from `0.867` to `0.770` —
because an independent, one-off shock dilutes the smooth, inertial part that
`c1` produces.

All three variances rise, which is the only part that behaves as expected.

</details>

## Next

That is the end of Level 2. You can write a model, solve its steady state,
steer it, turn it into a state-space solution, check that the solution is
sound, and read off what it implies.

**Tutorial 11 · Your first simulation** opens Level 3 and starts using it.
`simulate` has been appearing since tutorial 1 with no explanation: how the
input databox is built, what initial conditions the model needs and where it
looks for them, how the simulation span works, what `return_info` reports,
and how to read what comes back.
