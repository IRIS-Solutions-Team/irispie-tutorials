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

# 23 · Inside the data contract

**40 minutes** · after *22 · Inside the solution* · next: *24 · Building
models from code*

Tutorial 22 opened the solution. This one opens the other half of a
simulation. A `Databox` is a dictionary of time series and the solvers work
on a rectangular array of floats, so something has to turn one into the
other, decide what to do about names that are absent, and work out how many
periods of padding the model needs on each side.

That object is a `Dataslate`, and the description of what a method will ask
for is a `Slatable`. Neither appears in ordinary use. They are worth knowing
because a simulation that returns nothing but `nan` has almost always failed
here rather than in the solver.

```{code-cell} ipython3
import numpy as np
from irispie import Simultaneous, Databox, Series, qq
from irispie.dataslates.main import Dataslate

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

m = Simultaneous.from_string(SOURCE, linear=True, flat=True)
m.assign_strict(**CALIB)
m.steady()
m.solve_first_order()

SPAN = qq(2025,1) >> qq(2026,4)
VARIABLES = ("y", "pi", "i", "q", "y_w", "pi_w", "i_w")
```

## What a method says it needs

Every method that consumes data publishes a description of what it will look
for. For simulation that description is `slatable_for_simulate`:

```{code-cell} ipython3
slatable = m.slatable_for_simulate(parameters_from_data=False,
                                   shocks_from_data=True,
                                   stds_from_data=False)

print("names it will look for :", len(slatable.databox_names))
print("names with a fallback  :", len(slatable.fallbacks))
print("parameters it overwrites:", len(slatable.overwrites))
print("validators             :", len(slatable.databox_validators))
print()
print("max_lag  :", slatable.max_lag)
print("max_lead :", slatable.max_lead)
```

Four pieces of information, and each one determines something a user runs
into.

**The names** are everything the method might read: the variables, the
shocks, their anticipated counterparts, the measurement shocks and the
parameters. Fifty-six names for a seven-variable model.

**The fallbacks** are what happens when a name is absent. Seventeen names
have one, and they are all shocks:

```{code-cell} ipython3
with_fallback = sorted(slatable.fallbacks)
without = [n for n in slatable.databox_names if n not in slatable.fallbacks]

print("with a fallback:", with_fallback[:6], "...")
print("the fallback value:", slatable.fallbacks["shk_y"])
print()
print("without one    :", without[:10], "...")
```

A shock that is not supplied is zero. A variable that is not supplied is
nothing at all, and that asymmetry is the subject of the Break it below.

**The overwrites** are the parameters, pushed into the array from the model
rather than read from the databox. That is why `parameters_from_data=False`
is the default: the calibration on the model object wins.

**`max_lag` and `max_lead`** say how far outside the simulation span the
method needs to see. This model has a one-period lag and a one-period lead.

## Building the array

```{code-cell} ipython3
db = Databox.steady(m, SPAN)
dataslate = Dataslate.from_databox_for_slatable(slatable, db, SPAN)

array = np.asarray(dataslate.get_data_variant())

print("array shape (names, periods):", array.shape)
print()
print("simulation span   :", len(SPAN), "periods, from", SPAN[0], "to", SPAN[-1])
print("dataslate span    :", dataslate.num_periods, "periods, from",
      dataslate.start, "to", dataslate.end)
print()
print("num_initials :", dataslate.num_initials)
print("num_terminals:", dataslate.num_terminals)
print("base_columns :", dataslate.base_columns)
```

Eight quarters were requested and the array has ten columns. One was added
at the front because the model has a lag, and one at the back because it has
a lead. `base_columns` records which columns are the simulation itself, so
the solver can write to the middle and leave the padding alone.

This is the arithmetic behind an observation first made in tutorial 11: the
number of periods in a simulation output is often one more than the number
asked for. The extra column is the initial condition, and it is in the
output because it was in the array.

## The initial column is what the lag reads

The column before the span is not decoration. It holds `xi(t-1)` for the
first period, and changing it changes everything downstream:

```{code-cell} ipython3
for value in (0.0, 1.0):
    box = Databox.steady(m, SPAN)
    box["y"] = Series(start=qq(2024,4), values=np.array([value]))
    out = m.simulate(box, SPAN)
    print(f"  y in 2024Q4 = {value}  ->  y = "
          f"{np.round(np.asarray(out['y'][SPAN]).ravel()[:4], 4)}")
```

An output gap of one in the quarter before the simulation decays to `0.634`,
`0.3591`, `0.1849`. The first of those is the top-left entry of `T` from
tutorial 8, which is what it means for the initial column to be the thing
the lag reads.

## The terminal column

The column after the span holds the terminal condition for the
forward-looking variables. For a first-order solution it is the steady
state, which is why it needs no attention:

```{code-cell} ipython3
names = list(slatable.databox_names)
print("pi across every column of the array:")
print(" ", np.round(array[names.index("pi"), :], 4))
print()
print("the last column is", dataslate.end, "which is one quarter past",
      SPAN[-1])
```

Every column is `2`, the steady state, including the terminal one. A
stacked-time simulation uses the same column differently, and tutorial 13's
`stacked_time` method is where the terminal condition does real work.

## ⚠️ Break it

**A simulation that returns nothing, and does not complain.**

The fallback asymmetry is the cause of most of these. Shocks default to
zero; variables do not default to anything:

```{code-cell} ipython3
full = Databox.steady(m, SPAN)

shocks_only = Databox()
for name in full.get_names():
    if str(name).startswith("shk"):
        shocks_only[name] = full[name]

out = m.simulate(shocks_only, SPAN)

print("the call returned a", type(out).__name__, "as usual")
print("y:", np.asarray(out["y"][SPAN]).ravel()[:5])
```

No exception, no warning, and a databox of the right shape full of `nan`.
The missing variables became `nan` rows in the array, and `nan` propagates
through every subsequent calculation.

The validators do not catch it, because they check the type of what was
supplied rather than its contents:

```{code-cell} ipython3
name, (_, message) = next(iter(slatable.databox_validators.items()))
print(f"validator for {name!r}: {message!r}")
print()

nan_input = Databox.steady(m, SPAN)
nan_input["y"] = Series(start=SPAN[0], values=np.full(len(SPAN), np.nan))

print("y:", np.asarray(m.simulate(nan_input, SPAN)["y"][SPAN]).ravel()[:4])
```

A series full of `nan` is still a series, so the validator passes it. The
check is that the input is a time series at all, not that it contains
numbers.

The diagnosis is to build the array directly and inspect it, rather than
attributing the failure to the solver:

```{code-cell} ipython3
def report_missing(model, databox, span, **kwargs):
    """Which of the names a simulation needs arrive as all-nan."""
    slat = model.slatable_for_simulate(parameters_from_data=False,
                                       shocks_from_data=True,
                                       stds_from_data=False, **kwargs)
    slate = Dataslate.from_databox_for_slatable(slat, databox, span)
    data = np.asarray(slate.get_data_variant())
    rows = list(slat.databox_names)
    return [rows[k] for k in range(data.shape[0]) if np.isnan(data[k, :]).all()]


print("missing from the shocks-only databox:")
print(" ", report_missing(m, shocks_only, SPAN))
```

Ten names, which are the seven transition variables and the three
measurement variables. That list is the answer to the question the
simulation refused to ask.

## What you did

```{code-cell} ipython3
# what a method will look for, and how it will treat absences
slatable = m.slatable_for_simulate(parameters_from_data=False,
                                   shocks_from_data=True,
                                   stds_from_data=False)
slatable.databox_names
slatable.fallbacks
slatable.max_lag, slatable.max_lead

# the array the solver actually receives
dataslate = Dataslate.from_databox_for_slatable(slatable, db, SPAN)
dataslate.get_data_variant()
dataslate.num_initials, dataslate.num_terminals
dataslate.base_columns

# which names arrived empty
report_missing(m, shocks_only, SPAN)
```

## Things to remember

1. **A `Slatable` is the declaration and a `Dataslate` is the array.** The
   first says what a method needs; the second is what it receives.
2. **The array is wider than the span.** `max_lag` adds columns at the front
   and `max_lead` at the back, which is why an output often has one period
   more than was requested.
3. **The initial column is the lag.** It is where an initial condition is
   read from, and it determines the whole path.
4. **The terminal column is the terminal condition**, the steady state for a
   first-order solution.
5. **Shocks fall back to zero; variables do not fall back at all.** A
   missing variable becomes a row of `nan`.
6. **`nan` propagates silently.** The call succeeds, the output has the
   right shape, and every number in it is `nan`.
7. **The validators check the type, not the contents.** A series of `nan`
   passes.
8. **Build the slate yourself to diagnose it.** All-`nan` rows name exactly
   what was missing.

## Exercise

Five input databoxes are offered to the same simulation. Work out which
produce numbers and which produce `nan`:

1. everything from `Databox.steady`
2. the shocks only
3. the seven transition variables and no shocks
4. six of the seven variables, with `y` left out
5. all seven variables, but `y` starting at `2025Q1` rather than `2024Q4`

```{code-cell} ipython3
# your turn
```

<details>
<summary><b>Show the answer</b></summary>

<br>

**Cases 1 and 3 work. Cases 2, 4 and 5 return `nan`.**

| input | result |
|---|---|
| everything | `-0.0000` |
| shocks only | `nan` |
| seven variables, no shocks | `-0.0000` |
| six variables, `y` left out | `nan` |
| `y` starting at `2025Q1` | `nan` |

Case 3 is the one that shows the fallback rule. Omitting every shock is
harmless, because each one falls back to zero and a simulation with no
shocks is a perfectly ordinary request.

Case 5 is the one worth remembering, because the variable is present and the
simulation still fails. `y` was supplied over the simulation span but not
for `2024Q4`, which is the column the lag reads. The array therefore has a
`nan` in its first column, the first period of the simulation inherits it,
and it spreads from there.

This is the most common version of the failure in practice. A databox
assembled from published data naturally starts where the data starts, and a
model with a lag needs one period before that. The fix is to extend the
series backwards by `max_lag` periods, and the check is to confirm that the
input covers `span[0] - 1` for every variable with a lag.

Nothing in cases 2, 4 or 5 raises. All three return a databox of the right
shape, and the only way to tell them apart from a successful run is to look
at the numbers.

</details>

## Next

**Tutorial 24 · Building models from code** constructs a model without
writing a source file. `ModelSource.from_lists` assembles the pieces
directly, the portable format allows a model to be edited as data, and
`Stacker` exposes the stacked-time system that tutorial 13 solved.
